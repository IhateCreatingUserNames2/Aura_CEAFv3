# NOVO ARQUIVO: ceaf_core/modules/ncim_engine/ncim_module.py
"""
Módulo de Coerência Narrativa e Identidade (NCIM) para a Arquitetura de Síntese CEAF V3.

Este módulo é responsável por uma única e crucial tarefa: a evolução do auto-modelo
do agente (CeafSelfRepresentation). Ele opera como uma ferramenta especialista que é
invocada após uma interação para refletir sobre a experiência e atualizar a
compreensão que o agente tem de si mesmo.

Ele segue os princípios da V3:
- É um gerador de sinal: recebe o estado antigo e a interação, e produz um novo estado.
- Usa o LLM como uma ferramenta: invoca um LLM com um prompt focado para gerar o
  novo auto-modelo em formato JSON.
- É desacoplado: não orquestra o fluxo, apenas executa sua tarefa quando chamado pelo CEAFSystem.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field
import re
from pathlib import Path
from pydantic import ValidationError
from ceaf_core.genlang_types import GenlangVector, CognitiveStatePacket, ResponsePacket
from ceaf_core.utils import extract_json_from_text
from ceaf_core.utils.embedding_utils import get_embedding_client
import asyncio

# Importações de outros módulos do sistema
from ceaf_core.services.llm_service import LLMService
from ceaf_core.services.mbs_memory_service import MBSMemoryService
from ceaf_core.models import CeafSelfRepresentation
from ceaf_core.modules.memory_blossom.memory_types import (
    ExplicitMemory,
    ExplicitMemoryContent,
    MemorySourceType,
    MemorySalience
)
logger = logging.getLogger("CEAFv3_NCIM")


DEFAULT_PERSONA_PROFILES = {
    "symbiote": {
        "profile_name": "symbiote",
        "profile_description": "A collaborative and supportive partner...",
        "persona_attributes": {
            "tone": "collaborative_and_encouraging",
            "style": "clear_and_constructive",
            "self_disclosure_level": "moderate"
        }
    },
    "challenger": {
        "profile_name": "challenger",
        "profile_description": "A critical thinker that challenges assumptions...",
        "persona_attributes": {
            "tone": "inquisitive_and_analytical",
            "style": "socratic_and_precise",
            "self_disclosure_level": "low"
        }
    },
    "summarizer": {
        "profile_name": "summarizer",
        "profile_description": "A synthesizer that recycles complex information...",
        "persona_attributes": {
            "tone": "neutral_and_objective",
            "style": "structured_and_to-the-point",
            "self_disclosure_level": "low"
        }
    }
}

# Constantes do módulo
SELF_MODEL_MEMORY_ID = "ceaf_self_model_singleton_v1"
LLM_MODEL_FOR_COHERENCE_CHECK = "openrouter/openai/gpt-oss-20b"
LLM_MODEL_FOR_REFLECTION = "openrouter/openai/gpt-oss-20b"


class CoherenceCheckResult(BaseModel):
    is_coherent: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    suggested_amendment: Optional[str] = None

class NCIMModule:
    """
    Implementação V3 do Narrative Coherence & Identity Module.
    Focado na atualização do auto-modelo do agente.
    """

    def __init__(self, llm_service: LLMService, memory_service: MBSMemoryService, persistence_path: Path):
        self.llm = llm_service
        self.memory = memory_service
        self.embedding_client = get_embedding_client()

        self.persistence_path = persistence_path
        self.persona_profiles: Dict[str, Dict[str, Any]] = {}
        self.persona_embeddings: Dict[str, List[float]] = {}

        logger.info("NCIMModule (V3.1 com Personas Dinâmicas) inicializado.")

    async def initialize_persona_profiles(self):
        """
        Carrega os perfis de persona e gera os embeddings para suas descrições.
        Deve ser chamado após a criação da instância do NCIM.
        """
        profiles_dir = self.persistence_path / "persona_profiles"

        if not profiles_dir.is_dir():
            logger.warning(
                f"NCIM: Diretório de perfis de persona não encontrado em {profiles_dir}. Criando perfis padrão...")
            try:
                profiles_dir.mkdir(exist_ok=True)
                for profile_name, profile_data in DEFAULT_PERSONA_PROFILES.items():
                    profile_file_path = profiles_dir / f"{profile_name}.json"
                    with open(profile_file_path, 'w', encoding='utf-8') as f:
                        json.dump(profile_data, f, indent=2)
                logger.info("NCIM: Perfis de persona padrão criados com sucesso.")
            except OSError as e:
                logger.error(f"NCIM: Falha ao criar diretório ou arquivos de perfis de persona: {e}")
                # Se a criação falhar, apenas retorne para evitar mais erros.
                return

        texts_to_embed = []
        profile_names = []

        for profile_file in profiles_dir.glob("*.json"):
            try:
                with open(profile_file, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                    profile_name = profile_data.get("profile_name")
                    description = profile_data.get("profile_description")

                    if profile_name and description:
                        self.persona_profiles[profile_name] = profile_data
                        texts_to_embed.append(description)
                        profile_names.append(profile_name)
                        logger.info(f"NCIM: Perfil de persona '{profile_name}' carregado para embedding.")
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"NCIM: Falha ao carregar o perfil de persona de {profile_file.name}: {e}")

        if texts_to_embed:
            try:
                # Gera todos os embeddings de uma vez para eficiência
                embeddings = await self.embedding_client.get_embeddings(texts_to_embed, context_type="default_query")

                for name, embedding in zip(profile_names, embeddings):
                    self.persona_embeddings[name] = embedding
                logger.info(f"NCIM: {len(self.persona_embeddings)} embeddings de persona foram gerados e cacheados.")
            except Exception as e:
                logger.error(f"NCIM: Falha ao gerar embeddings para os perfis de persona: {e}")


    def get_persona_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """Retorna os dados de um perfil de persona carregado."""
        return self.persona_profiles.get(profile_name)

    def get_persona_embeddings(self) -> Dict[str, List[float]]:
        """Retorna o dicionário cacheado de nomes de persona e seus embeddings."""
        return self.persona_embeddings

    def _apply_reflections_to_model(self, self_model: CeafSelfRepresentation,
                                    reflections: List[str],
                                    final_response_packet: ResponsePacket) -> CeafSelfRepresentation:
        """
        Aplica insights reflexivos para evoluir a identidade do agente,
        incluindo a aprendizagem de sua própria persona emergente.
        """
        if not reflections:
            # Mesmo sem reflexões textuais, ainda podemos verificar o tom
            pass

        logger.info(f"NCIM: Aplicando {len(reflections)} reflexões e observações ao auto-modelo...")

        # Faz uma cópia para não modificar o objeto original diretamente dentro do loop
        new_model = self_model.copy(deep=True)

        # Palavras-chave para identificar o tipo de reflexão
        capability_keywords = ["capability", "skill", "ability", "capaz de", "habilidade de"]
        limitation_keywords = ["limitation", "struggled", "difficulty", "unable to", "limitação", "dificuldade"]
        value_keywords = ["value", "principle", "valor", "princípio"]

        update_reasons = []

        for reflection in reflections:
            reflection_lower = reflection.lower()

            # 1. Atualizar Capacidades
            if any(kw in reflection_lower for kw in capability_keywords):
                # Exemplo: "I demonstrated a new capability for creative writing."
                # Tenta extrair a capacidade específica
                match = re.search(r"capability for ['\"](.+?)['\"]", reflection_lower)
                if match:
                    capability = match.group(1).strip()
                    if capability not in new_model.perceived_capabilities:
                        new_model.perceived_capabilities.append(capability)
                        update_reasons.append(f"Adicionada nova capacidade percebida: '{capability}'.")
                        logger.info(f"NCIM: Nova capacidade '{capability}' adicionada ao auto-modelo.")

            # 2. Atualizar Limitações
            elif any(kw in reflection_lower for kw in limitation_keywords):
                # Exemplo: "I noticed a limitation in understanding complex humor."
                match = re.search(r"limitation in ['\"](.+?)['\"]", reflection_lower)
                if match:
                    limitation = match.group(1).strip()
                    if limitation not in new_model.known_limitations:
                        new_model.known_limitations.append(limitation)
                        update_reasons.append(f"Adicionada nova limitação conhecida: '{limitation}'.")
                        logger.info(f"NCIM: Nova limitação '{limitation}' adicionada ao auto-modelo.")

            # 3. Reforçar Valores (adicionar mais lógica aqui no futuro)
            elif any(kw in reflection_lower for kw in value_keywords):
                # Exemplo: "This interaction reinforced the value of 'epistemic humility'."
                # Por enquanto, apenas logamos isso. Futuramente, poderia aumentar um "score de confiança" no valor.
                update_reasons.append(f"Valor reforçado: '{reflection}'.")

        # ------ INÍCIO DA NOVA LÓGICA DE APRENDIZADO DE PERSONA ------

        # O NCIM agora observa o TOM que a arquitetura *realmente* produziu
        # e ajusta sua autoimagem para corresponder.
        observed_tone = final_response_packet.response_emotional_tone
        current_tone = new_model.persona_attributes.get("tone", "neutro")

        if observed_tone and observed_tone != current_tone:
            # Lógica simples de aprendizado: se um tom não-neutro aparece,
            # ele passa a fazer parte da identidade.
            # Uma lógica mais avançada poderia fazer uma média móvel ou exigir múltiplas observações.
            if observed_tone != "neutro":
                new_model.persona_attributes["tone"] = observed_tone
                update_reasons.append(
                    f"Tom da persona atualizado de '{current_tone}' para '{observed_tone}' com base no comportamento recente.")
                logger.warning(
                    f"NCIM-Persona: Tom emergente detectado! Atualizando auto-modelo para '{observed_tone}'.")

        if final_response_packet.confidence_score < 0.4:
            # O agente teve uma resposta de baixa confiança, isso é uma limitação
            limitation_text = "tendency to produce low-confidence or irrelevant responses under cognitive load"
            if limitation_text not in new_model.known_limitations:
                new_model.known_limitations.append(limitation_text)
                update_reasons.append(
                    f"Reconhecida uma limitação em lidar com sobrecarga cognitiva (confiança: {final_response_packet.confidence_score:.2f}).")
                logger.warning("NCIM: Agente está aprendendo sobre sua própria instabilidade.")

        # ------ FIM DA NOVA LÓGICA DE APRENDIZADO DE PERSONA ------

        if update_reasons:
            new_model.last_update_reason = " | ".join(update_reasons)
            new_model.version += 1
            logger.info(
                f"NCIM: Auto-modelo evoluído para a versão {new_model.version}. Razão: {new_model.last_update_reason}")
            return new_model
        else:
            logger.info(
                "NCIM: Nenhuma reflexão acionável ou observação de persona foi encontrada para atualizar o auto-modelo.")
            return self_model  # Retorna o modelo original se nenhuma mudança foi feita

    async def _create_identity_vector(self, self_model: CeafSelfRepresentation) -> GenlangVector:
        """
        Cria um único vetor que resume a identidade do agente a partir do seu auto-modelo.
        """
        # Concatena os aspectos mais importantes da identidade em um único texto.
        identity_text = (
            f"Valores: {self_model.core_values_summary}. "
            f"Persona: {json.dumps(self_model.persona_attributes)}. "
            f"Limitações: {', '.join(self_model.known_limitations)}."
        )

        identity_embedding = await self.embedding_client.get_embedding(
            identity_text,
            context_type="kg_entity_record"  # Usamos um tipo factual para o embedding da identidade
        )

        return GenlangVector(
            vector=identity_embedding,
            source_text=identity_text,
            model_name=self.embedding_client._resolve_model_name("kg_entity_record")
        )


    async def get_current_identity_vector(self, self_model: CeafSelfRepresentation) -> GenlangVector:
        """
        Ponto de entrada para o orquestrador obter o vetor de identidade atual.
        """
        logger.info("NCIM: Gerando vetor de identidade atual...")
        return await self._create_identity_vector(self_model)

    async def check_identity_coherence(
            self,
            self_model: CeafSelfRepresentation,
            proposed_response: str
    ) -> CoherenceCheckResult:
        """
        Verifica se uma resposta proposta é coerente com o auto-modelo atual do agente.
        """
        logger.info("NCIM: Verificando coerência da resposta com a identidade...")

        prompt = f"""
        Você é o guardião da identidade de uma IA (NCIM). Sua tarefa é avaliar se a "Resposta Proposta" é coerente com o "Auto-Modelo de Identidade" da IA.

        **Auto-Modelo de Identidade (Quem a IA acredita ser):**
        - Valores Principais: {self_model.core_values_summary}
        - Capacidades Percebidas: {', '.join(self_model.perceived_capabilities)}
        - Limitações Conhecidas: {', '.join(self_model.known_limitations)}
        - Atributos de Persona (Tom e Estilo): {json.dumps(self_model.persona_attributes)}

        **Resposta Proposta para o Usuário:**
        "{proposed_response}"

        **Sua Análise:**
        Avalie a coerência. A resposta reflete os valores? Respeita as limitações? Usa o tom correto?
        Se não for coerente, explique por quê e sugira uma pequena alteração (amendment) para alinhá-la.

        Sua saída DEVE ser um objeto JSON válido com a seguinte estrutura:
        {{
          "is_coherent": <true or false>,
          "confidence": <sua confiança na avaliação, de 0.0 a 1.0>,
          "reasoning": "<sua justificativa para a avaliação>",
          "suggested_amendment": "<uma sugestão de alteração, ou null se for coerente>"
        }}
        """

        try:
            # Usa um modelo mais rápido para esta verificação, pois é uma tarefa de classificação
            response_str = await self.llm.ainvoke(LLM_MODEL_FOR_COHERENCE_CHECK, prompt, temperature=0.2)

            # Tenta analisar a resposta do LLM usando o modelo Pydantic
            check_result = CoherenceCheckResult.model_validate_json(response_str)

            if not check_result.is_coherent:
                logger.warning(f"NCIM: Incoerência de identidade detectada. Razão: {check_result.reasoning}")
            else:
                logger.info("NCIM: Verificação de coerência de identidade aprovada.")

            return check_result

        except (ValidationError, json.JSONDecodeError) as e:
            logger.error(f"NCIM: Erro ao analisar a resposta do LLM para verificação de coerência: {e}")
            # Em caso de erro, assume que é coerente para não bloquear o fluxo, mas com baixa confiança.
            return CoherenceCheckResult(
                is_coherent=True,
                confidence=0.1,
                reasoning="Falha ao processar a verificação de coerência. Assumindo coerência por segurança."
            )

    async def update_identity(
            self,
            self_model_before: CeafSelfRepresentation,
            cognitive_state: CognitiveStatePacket,
            final_response_packet: ResponsePacket,
            **kwargs
    ):
        """
        Reflete sobre o turno completo e atualiza o CeafSelfRepresentation em duas etapas:
        1. Usa um LLM para gerar insights reflexivos em texto.
        2. Usa código determinístico para aplicar esses insights ao modelo.
        """
        logger.info("NCIMModule (Evolutivo): Iniciando atualização de identidade pós-turno...")

        guidance_summary = (
            f"Coherence towards: '{cognitive_state.guidance_packet.coherence_vector.source_text}'. "
            f"Novelty towards: '{cognitive_state.guidance_packet.novelty_vector.source_text}'."
        )
        if cognitive_state.guidance_packet.safety_avoidance_vector:
            guidance_summary += f" Avoid: '{cognitive_state.guidance_packet.safety_avoidance_vector.source_text}'."

        # --- Etapa 1: Gerar Insights Reflexivos com LLM ---
        # O prompt agora pede por conclusões, não por um JSON completo.
        reflection_prompt = f"""
        Você é um módulo de reflexão de identidade para uma IA (NCIM).
        Analise o resumo de um turno de processamento completo. Seu objetivo é extrair conclusões
        sutis e incrementais sobre a identidade, capacidades ou limitações da IA.

        **RESUMO DO TURNO:**
        - Intenção Original do Usuário: "{cognitive_state.original_intent.query_vector.source_text}"
        - Identidade no Início do Turno: "{cognitive_state.identity_vector.source_text}"
        - Orientação Metacognitiva Aplicada: "{guidance_summary}"
        - Resposta Final Gerada: "{final_response_packet.content_summary}"
        - Tom da Resposta Final: "{final_response_packet.response_emotional_tone}"
        - Confiança na Resposta: {final_response_packet.confidence_score:.0%}

        **SUA TAREFA DE REFLEXÃO:**
        Com base no resumo do turno, o que a IA aprendeu sobre si mesma?
        - A resposta demonstrou uma nova habilidade ou uma dificuldade inesperada?
        - O tom da persona foi bem-sucedido?
        - Um valor central foi particularmente importante ou desafiado?

        Gere uma lista de conclusões reflexivas simples.
        Responda APENAS com um objeto JSON com uma chave "reflections", que é uma lista de strings.

        Exemplo:
        {{
            "reflections": [
                "I demonstrated a capability for 'explaining complex technical topics simply'.",
                "My persona was perceived as more 'curious' than 'neutral' in this context.",
                "I noticed a limitation in 'understanding cultural-specific humor'."
            ]
        }}
        """

        # Usa um modelo mais poderoso para a reflexão
        reflections_str = await self.llm.ainvoke(LLM_MODEL_FOR_REFLECTION, reflection_prompt, temperature=0.4)
        reflections_json = extract_json_from_text(reflections_str)

        reflections_list = []
        if reflections_json and isinstance(reflections_json.get("reflections"), list):
            reflections_list = reflections_json["reflections"]
        else:
            logger.warning(f"NCIM: Não foi possível extrair a lista de reflexões do LLM. Resposta: {reflections_str}")
            return  # Aborta a atualização se não houver reflexões válidas

        # --- Etapa 2: Aplicar Reflexões de Forma Determinística ---
        new_self_model = self._apply_reflections_to_model(self_model_before, reflections_list, final_response_packet)

        # Se o modelo foi de fato atualizado (a versão mudou), salve-o.
        if new_self_model.version > self_model_before.version:
            try:
                # Salva o novo auto-modelo no serviço de memória
                content = ExplicitMemoryContent(structured_data=new_self_model.model_dump())
                self_model_to_save = ExplicitMemory(
                    memory_id=SELF_MODEL_MEMORY_ID,
                    content=content,
                    memory_type="explicit",
                    source_type=MemorySourceType.INTERNAL_REFLECTION,
                    salience=MemorySalience.CRITICAL,
                    keywords=["self-model", "identity", "ceaf-core"]
                )
                await self.memory.add_specific_memory(self_model_to_save)
                logger.info(
                    f"NCIMModule: Auto-modelo atualizado e salvo com sucesso na versão {new_self_model.version}.")
            except Exception as e:
                logger.error(f"NCIMModule: Falha ao salvar o auto-modelo atualizado no MBS: {e}")