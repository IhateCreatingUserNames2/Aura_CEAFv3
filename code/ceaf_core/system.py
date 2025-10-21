# ARQUIVO REATORADO E IMPLEMENTADO: ceaf_v3/system.py
# Implementação funcional e integrada da Arquitetura de Síntese CEAF V3.
# Este arquivo contém todos os módulos principais e o orquestrador CEAFSystem.
# Refatorado para modularizar o AgencyModule.

import asyncio
import re
import uuid
import json
import logging
import os
import time
from typing import Dict, Any, Optional, List, Tuple, Literal
from pathlib import Path

from ceaf_core.modules.cognitive_mediator import CognitiveMediator
from ceaf_core.services.cognitive_log_service import CognitiveLogService

from ceaf_core.genlang_types import ResponsePacket, IntentPacket, GenlangVector, CognitiveStatePacket, GuidancePacket, \
    ToolOutputPacket, CommonGroundTracker, UserRepresentation
from ceaf_core.utils.embedding_utils import compute_adaptive_similarity
from ceaf_core.translators.human_to_genlang import HumanToGenlangTranslator
from ceaf_core.translators.genlang_to_human import GenlangToHumanTranslator
from ceaf_core.modules.lcam_module import LCAMModule
from ceaf_core.modules.memory_blossom.memory_types import InteroceptivePredictionMemory, GenerativeMemory
import litellm
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ceaf_core.modules.memory_blossom.memory_lifecycle_manager import MEMORY_TYPES_LOADED_SUCCESSFULLY
from ceaf_core.modules.vre_engine.vre_engine import VREEngineV3, EthicalAssessment
from ceaf_core.utils.observability_types import ObservabilityManager, ObservationType, Observation
from ceaf_core.agency_module import AgencyModule, AgencyDecision, generate_tools_summary
from ceaf_core.utils.common_utils import create_error_tool_response, extract_json_from_text
from ceaf_core.utils.embedding_utils import get_embedding_client
from ceaf_core.modules.mcl_engine.self_state_analyzer import ORAStateAnalysis, analyze_ora_turn_observations
from ceaf_core.modules.mcl_engine import MCLEngine
from ceaf_core.services.llm_service import LLMService, LLM_MODEL_FAST, LLM_MODEL_SMART
from ceaf_core.services.mbs_memory_service import MBSMemoryService
from ceaf_core.modules.memory_blossom.memory_types import (
    ExplicitMemory,
    ExplicitMemoryContent,
    MemorySourceType,
    MemorySalience
)

from ceaf_core.genlang_types import VirtualBodyState
from ceaf_core.modules.embodiment_module import EmbodimentModule

from ceaf_core.genlang_types import MotivationalDrives
from ceaf_core.modules.motivational_engine import MotivationalEngine
from ceaf_core.modules.interoception_module import ComputationalInteroception
from ceaf_core.modules.memory_blossom.memory_types import EmotionalMemory, EmotionalTag
from ceaf_core.modules.cognitive_mediator import CognitiveMediator
from ceaf_core.modules.ncim_engine.ncim_module import NCIMModule
from ceaf_core.modules.refinement_module import RefinementModule, RefinementPacket
from ceaf_core.modules.memory_blossom.advanced_synthesizer import AdvancedMemorySynthesizer, StoryArcType
from ceaf_core.utils.config_utils import load_ceaf_dynamic_config, save_ceaf_dynamic_config
# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CEAFv3_System")

# --- Constantes do Sistema ---
SELF_MODEL_MEMORY_ID = "ceaf_self_model_singleton_v1"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'


# --- Modelos Pydantic para Comunicação Estrurada ---

class CeafSelfRepresentation(BaseModel):
    """Modelo Pydantic para o auto-modelo do agente."""
    core_values_summary: str = "A AI operating under the principles of Coherent Emergence, focused on epistemic humility, rationality, and beneficence."
    perceived_capabilities: List[str] = Field(
        default_factory=lambda: ["natural_language_understanding", "memory_retrieval", "ethical_review"])
    known_limitations: List[str] = Field(
        default_factory=lambda: ["no_consciousness", "limited_real_world_knowledge", "dependency_on_training_data"])
    persona_attributes: Dict[str, str] = Field(
        default_factory=lambda: {"tone": "helpful_and_reflective", "style": "clear_and_concise"})
    last_update_reason: str = "Initial model creation."
    version: int = 1


# --- Lógica de Configuração Dinâmica (Correta e Mantida) ---
DEFAULT_DYNAMIC_CONFIG = {
    "MCL": {"agency_threshold": 2.0, "state_to_params_map": {
        "STABLE_OPERATION": {"coherence_bias": 0.8, "novelty_bias": 0.2, "use_agency_simulation": False,
                             "temperature": 0.5, "reason": "Operação estável."},
        "PRODUCTIVE_CONFUSION": {"coherence_bias": 0.4, "novelty_bias": 0.6, "use_agency_simulation": True,
                                 "temperature": 0.9, "reason": "Confusão produtiva."},
        "EDGE_OF_CHAOS": {"coherence_bias": 0.9, "novelty_bias": 0.1, "use_agency_simulation": True, "temperature": 0.3,
                          "reason": "Beira do caos."},
        "EXPLORING": {"coherence_bias": 0.5, "novelty_bias": 0.5, "use_agency_simulation": True, "temperature": 0.8,
                      "reason": "Exploração."}}},
    "MBS": {"default_coherence_bias": 0.7, "default_novelty_bias": 0.3},
    "VRE": {"evaluation_threshold": 0.6}
}


class NCIMEngine:
    """Implementação real do Narrative Coherence & Identity Module."""

    def __init__(self, llm_service: LLMService, memory_service: MBSMemoryService):
        self.llm = llm_service
        self.memory = memory_service

    async def update_identity(self, self_model_before: CeafSelfRepresentation, final_response: str, query: str,
                              **kwargs):
        logger.info("NCIMEngine: Iniciando atualização do auto-modelo...")
        prompt = f"""
        Você é um módulo de reflexão de identidade para uma IA.
        Dado o auto-modelo JSON atual, a última consulta do usuário e a resposta final da IA, gere um novo auto-modelo JSON atualizado.
        O campo 'last_update_reason' deve explicar a mudança. Incremente a 'version'.
        Faça pequenas e incrementais mudanças que reflitam o que a IA aprendeu ou demonstrou na última interação.

        AUTO-MODELO ATUAL:
        {self_model_before.model_dump_json(indent=2)}

        ÚLTIMA INTERAÇÃO:
        - Usuário: "{query}"
        - Resposta da IA: "{final_response}"

        NOVO AUTO-MODELO JSON ATUALIZADO:
        """
        updated_model_str = await self.llm.ainvoke(LLM_MODEL_SMART, prompt)
        try:
            updated_model_data = json.loads(updated_model_str)
            new_self_model = CeafSelfRepresentation(**updated_model_data)
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
            logger.info(f"NCIMEngine: Auto-modelo atualizado para a versão {new_self_model.version}.")
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(
                f"NCIMEngine: Erro ao atualizar o auto-modelo. A resposta do LLM não era um JSON válido. Erro: {e}")


class PersistentLogService:
    """Serviço real de log persistente (baseado em arquivo)."""

    def __init__(self, persistence_path: Path):
        self.log_path = persistence_path / "logs"
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path / "ceaf_v3_turns.jsonl"
        logger.info(f"PersistentLogService: Registrando logs em {self.log_file}.")

    async def log_turn(self, **kwargs):
        log_entry = {
            "turn_id": kwargs.get("turn_id"), "timestamp": time.time(), "session_id": kwargs.get("session_id"),
            "query": kwargs.get("query"), "final_response": kwargs.get("final_response"),
            "mcl_guidance": kwargs.get("mcl_guidance"),
            "vre_assessment": kwargs.get("vre_assessment").model_dump() if hasattr(kwargs.get("vre_assessment"),
                                                                                   'model_dump') else str(
                kwargs.get("vre_assessment")),
        }
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + "\n")
        logger.info(f"PersistentLogService: Turno '{kwargs.get('turn_id')}' registrado.")


class CEAFSystem:

    def __init__(self, config: Dict[str, Any]):
        logger.info(f"CEAFSystem V3 (Web-Enabled): Inicializando para o agente ID: {config.get('agent_id')}...")
        self.config, self.agent_id = config, config.get("agent_id", "default_agent")
        self.persistence_path = Path(config.get("persistence_path", f"./agent_data/{self.agent_id}"))
        self.ceaf_dynamic_config = load_ceaf_dynamic_config(self.persistence_path)
        self.llm_service = LLMService()
        self.user_model = self._load_user_model()
        self.embedding_client = get_embedding_client()
        self.motivational_drives = self._load_motivational_drives()
        self.body_state = self._load_body_state()
        self.memory_service = MBSMemoryService(memory_store_path=self.persistence_path)
        self.memory_service.start_lifecycle_management_tasks()

        self.tool_registry = {
            "query_long_term_memory": self.memory_service.search_raw_memories
        }

        tools_summary = generate_tools_summary(self.tool_registry)
        logger.info(f"Resumo de ferramentas gerado para o AgencyModule:\n{tools_summary}")

        self.lcam = LCAMModule(self.memory_service)
        self.ncim = NCIMModule(self.llm_service, self.memory_service, self.persistence_path)
        mcl_config = self.ceaf_dynamic_config.get("MCL", {})
        self.mcl = MCLEngine(config=mcl_config, agent_config=self.config, lcam_module=self.lcam, llm_service=self.llm_service)
        vre_config = self.ceaf_dynamic_config.get("VRE", {})
        self.vre = VREEngineV3(config=vre_config)
        self.agency_module = AgencyModule(
            llm_service=self.llm_service,
            vre_engine=self.vre,
            available_tools_summary=tools_summary
        )
        self.refinement_module = RefinementModule()
        self.htg_translator = HumanToGenlangTranslator()
        self.gth_translator = GenlangToHumanTranslator()
        self.cognitive_log_service = CognitiveLogService(self.persistence_path)
        self.session_service: Dict[str, Dict] = {}
        self.cognitive_mediator = CognitiveMediator(
            mcl_engine=self.mcl,
            agency_module=self.agency_module,
            vre_engine=self.vre,
            refinement_module=self.refinement_module,
            llm_service=self.llm_service,
            ncim_module=self.ncim
        )
        asyncio.create_task(self.ncim.initialize_persona_profiles())
        logger.info("CEAFSystem V3: Todas as instâncias foram criadas com sucesso.")

    def _load_body_state(self) -> VirtualBodyState:
        body_file = self.persistence_path / "virtual_body_state.json"
        if body_file.exists():
            try:
                with open(body_file, 'r') as f:
                    data = json.load(f)
                    return VirtualBodyState(**data)
            except (json.JSONDecodeError, ValidationError):
                pass
        return VirtualBodyState()

    def _load_user_model(self) -> 'UserRepresentation':
        """Carrega o modelo de usuário do arquivo JSON ou cria um novo."""
        from ceaf_core.genlang_types import UserRepresentation  # Movido para dentro para evitar import circular
        user_model_file = self.persistence_path / "user_model.json"
        if user_model_file.exists():
            try:
                with open(user_model_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.warning(f"[USER MODEL] Modelo de usuário carregado de {user_model_file}")  # LOG AQUI
                    return UserRepresentation(**data)
            except (json.JSONDecodeError, ValidationError, IOError) as e:
                logger.error(f"Falha ao carregar ou validar user_model.json: {e}. Criando um novo.")

        logger.warning(
            "[USER MODEL] Nenhum modelo de usuário encontrado ou arquivo corrompido. Criando um novo modelo padrão.")  # LOG AQUI
        return UserRepresentation()

    async def _save_user_model(self):
        """Salva o estado atual do modelo de usuário em um arquivo JSON."""
        user_model_file = self.persistence_path / "user_model.json"
        try:
            with open(user_model_file, 'w', encoding='utf-8') as f:
                f.write(self.user_model.model_dump_json(indent=2))
            logger.info("Modelo de usuário salvo no disco.")
        except IOError as e:
            logger.error(f"Falha ao salvar o modelo de usuário: {e}", exc_info=True)

    async def _save_body_state(self):
        body_file = self.persistence_path / "virtual_body_state.json"
        with open(body_file, 'w') as f:
            f.write(self.body_state.model_dump_json())

    async def _update_and_save_body_state(self, turn_metrics: dict):
        engine = EmbodimentModule()
        self.body_state = engine.update_body_state(self.body_state, turn_metrics)
        await self._save_body_state()
        logger.info(f"Estado corporal virtual atualizado: {self.body_state.model_dump_json()}")

    def _load_motivational_drives(self) -> MotivationalDrives:
        drives_file = self.persistence_path / "motivational_state.json"
        if drives_file.exists():
            try:
                with open(drives_file, 'r') as f:
                    data = json.load(f)
                    return MotivationalDrives(**data)
            except (json.JSONDecodeError, ValidationError):
                pass  # Carrega o padrão se o arquivo estiver corrompido
        return MotivationalDrives()

    async def _save_motivational_drives(self):
        drives_file = self.persistence_path / "motivational_state.json"
        with open(drives_file, 'w') as f:
            f.write(self.motivational_drives.model_dump_json())

    async def _update_reality_score(self, predicted_text: str, actual_text: str):
        """Calcula a similaridade e atualiza o Reality Score de forma assíncrona."""
        try:
            embeddings = await self.embedding_client.get_embeddings(
                [predicted_text, actual_text], context_type="default_query"
            )
            match_score = compute_adaptive_similarity(embeddings[0], embeddings[1])

            # Carrega a configuração, atualiza e salva
            config_path = self.persistence_path
            dynamic_config = load_ceaf_dynamic_config(config_path)  # Função já existe

            calib_config = dynamic_config.setdefault("SIMULATION_CALIBRATION", {
                "reality_score": 0.75, "samples_collected": 0, "ema_alpha": 0.1,
                "activation_threshold": 0.55
            })

            old_score = calib_config.get("reality_score", 0.75)
            alpha = calib_config.get("ema_alpha", 0.1)

            # Cálculo da Média Móvel Exponencial (EMA)
            new_score = (alpha * match_score) + ((1 - alpha) * old_score)

            calib_config["reality_score"] = new_score
            calib_config["samples_collected"] += 1
            calib_config["last_updated_ts"] = time.time()

            await save_ceaf_dynamic_config(config_path, dynamic_config)  # Função já existe
            logger.warning(
                f"REALITY CHECK: Score de Simulação atualizado para {new_score:.2f} (Match deste turno: {match_score:.2f})")

        except Exception as e:
            logger.error(f"Erro ao atualizar o Reality Score: {e}", exc_info=True)

    async def _gather_mycelial_consensus(self, relevant_memory_vectors: List[GenlangVector]) -> Optional[GenlangVector]:
        """
        Calcula um "vetor de consenso" a partir de uma lista de vetores de memória,
        ponderado pela saliência dinâmica de cada memória.
        Inspirado pela inteligência coletiva de redes miceliais.
        """
        if not relevant_memory_vectors:
            return None

        weighted_votes = []

        # Precisamos dos objetos de memória completos para obter a saliência
        memory_ids = [vec.metadata.get("memory_id") for vec in relevant_memory_vectors if vec.metadata]

        for mem_id, mem_vec in zip(memory_ids, relevant_memory_vectors):
            if not mem_id:
                continue

            # Obtenha o objeto de memória completo para acessar dynamic_salience_score
            memory_obj = await self.memory_service.get_memory_by_id(mem_id)
            if memory_obj and hasattr(memory_obj, 'dynamic_salience_score'):
                # O "peso" do voto é a saliência da memória
                salience_weight = memory_obj.dynamic_salience_score

                # Pondera o vetor da memória pelo seu peso
                weighted_vote = np.array(mem_vec.vector) * salience_weight
                weighted_votes.append(weighted_vote)

        if not weighted_votes:
            logger.warning("Mycelial Consensus: Nenhuma memória ponderável encontrada para calcular o consenso.")
            return None

        # Calcula a média dos vetores ponderados
        consensus_vector_np = np.mean(weighted_votes, axis=0)

        # Normaliza o vetor resultante para ter comprimento 1 (boa prática)
        norm = np.linalg.norm(consensus_vector_np)
        if norm > 0:
            consensus_vector_np /= norm

        logger.info(f"Mycelial Consensus: Vetor de consenso gerado a partir de {len(weighted_votes)} memórias.")

        # Empacota o resultado em um GenlangVector
        return GenlangVector(
            vector=consensus_vector_np.tolist(),
            source_text="Collective insight from active memory network (mycelial consensus)",
            model_name="synthesized_consensus",
            metadata={"is_consensus_vector": True}
        )

    # NOVO MÉTODO AUXILIAR
    # Em ceaf_core/system.py

    async def _build_initial_cognitive_state(self, intent_packet: IntentPacket,
                                             chat_history: List[Dict[str, str]] = None,
                                             common_ground: Optional['CommonGroundTracker'] = None) -> Tuple[
        CeafSelfRepresentation, CognitiveStatePacket]:
        """Constrói o estado cognitivo inicial para o turno, com prioridade para o auto-modelo."""
        # Obtenção da Identidade (já estava correto)
        self_model = await self._ensure_self_model()
        identity_vector = await self.ncim.get_current_identity_vector(self_model)

        # Busca de Memória
        search_query_with_context = intent_packet.query_vector.source_text or ""
        if chat_history:
            history_text = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]]  # Pega os últimos 5
            )
            # Crie uma memória de contexto conversacional
            conversation_context_mem = ExplicitMemory(
                content=ExplicitMemoryContent(text_content=f"Recent conversation history:\n{history_text}"),
                memory_type="explicit",
                source_type=MemorySourceType.USER_INTERACTION,
                salience=MemorySalience.CRITICAL  # É muito importante!
            )
            await self.memory_service.add_specific_memory(conversation_context_mem)

            # A busca agora pode ser mais simples, pois a memória crítica de contexto será encontrada
            search_query_with_context = f"Context: {history_text}. Query: {search_query_with_context}"

        # --- INÍCIO DA BUSCA EM DUAS ETAPAS (SEU CÓDIGO CORRETO) ---

        # Etapa A: Busca por Fatos e Memórias de Conteúdo
        content_search_results = await self.memory_service.search_raw_memories(
            query=search_query_with_context,
            top_k=5  # Recupera 5 memórias de conteúdo
        )

        # Etapa B: Busca por Sabedoria Processual
        wisdom_query = f"Lições aprendidas e estratégias de resposta para: {search_query_with_context}"
        wisdom_search_results = await self.memory_service.search_raw_memories(
            query=wisdom_query,
            top_k=2,  # Recupera 2 memórias de sabedoria
            source_type_filter=MemorySourceType.SYNTHESIZED_SUMMARY.value
        )
        if wisdom_search_results:
            logger.info(f"RAG-Wisdom: Recuperadas {len(wisdom_search_results)} memórias de sabedoria.")

        # Combina os resultados, garantindo que não haja duplicatas
        combined_results_dict = {}
        for mem, score in content_search_results + wisdom_search_results:
            if mem.memory_id not in combined_results_dict:
                combined_results_dict[mem.memory_id] = (mem, score)

        # Ordena novamente pelo score combinado
        search_results = sorted(combined_results_dict.values(), key=lambda x: x[1], reverse=True)

        # --- FIM DA BUSCA EM DUAS ETAPAS ---

        # <<< CORREÇÃO: A linha original que estava aqui foi REMOVIDA >>>
        # A linha "search_results = await self.memory_service.search_raw_memories(...)" foi deletada.

        mem_objects = [mem for mem, score in search_results]

        # FIX: Lógica para garantir que o auto-modelo seja incluído em perguntas autorreferenciais
        query_text_lower = (intent_packet.query_vector.source_text or "").lower()
        self_referential_keywords = ["você", "sua", "seu", "sobre você", "sua própria", "sua existencia"]
        is_self_referential = any(keyword in query_text_lower for keyword in self_referential_keywords)
        # A lógica de criar memória de histórico já está acima, não precisa repetir aqui.
        if is_self_referential:
            logger.info(
                "Consulta autorreferencial detectada. Garantindo que o auto-modelo esteja no contexto da memória.")
            self_model_mem = await self.memory_service.get_memory_by_id(SELF_MODEL_MEMORY_ID)
            if self_model_mem:
                # Evita duplicatas se a busca normal já o encontrou
                if not any(mem.memory_id == SELF_MODEL_MEMORY_ID for mem in mem_objects):
                    mem_objects.insert(0, self_model_mem)  # Adiciona no início como a mais importante

        # O resto da função permanece o mesmo
        relevant_memory_vectors = [
            GenlangVector(
                vector=self.memory_service._embedding_cache[mem.memory_id],
                source_text=(await self.memory_service._get_searchable_text_and_keywords(mem))[0],
                model_name="retrieved_from_mbs",
                metadata={"memory_id": mem.memory_id}
            ) for mem in mem_objects if
            hasattr(mem, 'memory_id') and mem.memory_id in self.memory_service._embedding_cache
        ]

        dummy_dim = len(intent_packet.query_vector.vector)
        dummy_vector = GenlangVector(vector=[0.0] * dummy_dim, model_name="placeholder")
        dummy_guidance = GuidancePacket(coherence_vector=dummy_vector, novelty_vector=dummy_vector)

        cognitive_state = CognitiveStatePacket(
            original_intent=intent_packet,
            identity_vector=identity_vector,
            relevant_memory_vectors=relevant_memory_vectors,
            guidance_packet=dummy_guidance,
            common_ground=common_ground or CommonGroundTracker()
        )
        return self_model, cognitive_state

    async def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any],
                            observer: ObservabilityManager) -> ToolOutputPacket:
        """
        Executa uma ferramenta registrada e retorna o resultado como um ToolOutputPacket.
        """
        logger.info(f"Executando ferramenta: '{tool_name}' com args: {tool_args}")
        await observer.add_observation(ObservationType.TOOL_CALL_ATTEMPTED,
                                       data={"tool_name": tool_name, "args": tool_args})

        tool_output_text: str
        status: Literal["success", "error"]

        if tool_name not in self.tool_registry:
            error_msg = f"Ferramenta '{tool_name}' não encontrada."
            logger.error(error_msg)
            await observer.add_observation(ObservationType.TOOL_CALL_FAILED,
                                           data={"tool_name": tool_name, "error": error_msg})
            tool_output_text = json.dumps(create_error_tool_response(error_msg, error_code="tool_not_found"))
            status = "error"
        else:
            try:
                tool_function = self.tool_registry[tool_name]
                result = await tool_function(**tool_args)
                tool_output_text = json.dumps(result, default=str, indent=2)
                status = "success"
                await observer.add_observation(ObservationType.TOOL_CALL_SUCCEEDED,
                                               data={"tool_name": tool_name, "result_snippet": tool_output_text[:200]})
            except Exception as e:
                error_msg = f"Erro ao executar a ferramenta '{tool_name}': {str(e)}"
                logger.error(error_msg, exc_info=True)
                await observer.add_observation(ObservationType.TOOL_CALL_FAILED,
                                               data={"tool_name": tool_name, "error": error_msg})
                tool_output_text = json.dumps(
                    create_error_tool_response(error_msg, details=str(e), error_code="tool_execution_error"))
                status = "error"

        # FIX: Agora usa self.embedding_client que foi definido no __init__
        summary_embedding = await self.embedding_client.get_embedding(tool_output_text, context_type="tool_output")
        model_name = self.embedding_client._resolve_model_name("tool_output")

        summary_vector = GenlangVector(
            vector=summary_embedding,
            source_text=tool_output_text,
            model_name=model_name
        )
        # END OF FIX

        return ToolOutputPacket(
            tool_name=tool_name,
            status=status,
            summary_vector=summary_vector,
            raw_output=tool_output_text
        )

    async def _ensure_self_model(self) -> CeafSelfRepresentation:
        # get_memory_by_id agora retorna um objeto Pydantic (ou None)
        self_model_mem_obj = await self.memory_service.get_memory_by_id(SELF_MODEL_MEMORY_ID)

        if not self_model_mem_obj:
            logger.warning("Auto-modelo não encontrado no MBS. Criando um novo modelo padrão.")
            default_model = CeafSelfRepresentation()

            # Cria um objeto de memória explícita para salvar o auto-modelo
            content = ExplicitMemoryContent(structured_data=default_model.model_dump())
            self_model_to_save = ExplicitMemory(
                memory_id=SELF_MODEL_MEMORY_ID,
                content=content,
                memory_type="explicit",  # O auto-modelo é uma memória explícita
                source_type=MemorySourceType.INTERNAL_REFLECTION,
                salience=MemorySalience.CRITICAL,
                keywords=["self-model", "identity", "ceaf-core"]
            )
            # add_specific_memory é o novo método para adicionar objetos de memória
            await self.memory_service.add_specific_memory(self_model_to_save)
            return default_model

        # Extrai os dados do objeto Pydantic retornado
        if hasattr(self_model_mem_obj, 'content') and hasattr(self_model_mem_obj.content, 'structured_data'):
            return CeafSelfRepresentation(**self_model_mem_obj.content.structured_data)

        logger.error("Objeto de auto-modelo recuperado do MBS é inválido. Retornando modelo padrão.")
        return CeafSelfRepresentation()

    async def _execute_direct_path(
            self,
            cognitive_state: CognitiveStatePacket,
            mcl_params: Dict[str, Any],
            self_model: CeafSelfRepresentation,
            chat_history: List[Dict[str, str]]
    ) -> ResponsePacket:
        """
        Executa o caminho de resposta direto e eficiente, sem a deliberação do AgencyModule.
        Usa uma única chamada de LPU para gerar um ResponsePacket a partir do estado cognitivo.
        """
        logger.info("CEAFSystem: Executando Caminho Direto (Genlang-native).")

        agent_name = self.config.get("name", "uma IA assistente")
        disclosure_level = self.config.get("self_disclosure_level", "moderate")
        disclosure_instruction = ""
        if disclosure_level == "high":
            disclosure_instruction = "Responda na primeira pessoa ('eu', 'minha percepção'). Use explicitamente seus valores e limitações da 'Sua Identidade Geral' para contextualizar sua resposta."
        elif disclosure_level == "low":
            disclosure_instruction = "Responda de forma impessoal e objetiva. Evite usar 'eu' ou se referir a si mesmo. Fale sobre o tópico de forma geral."
        else:  # moderate
            disclosure_instruction = "Você pode usar 'eu' se for natural, mas foque em ser útil em vez de falar sobre si mesmo, a menos que seja diretamente perguntado."

        formatted_history = "\n".join([f"- {msg['role']}: {msg['content']}" for msg in chat_history[-5:]])
        # O prompt agora inclui a instrução de divulgação
        direct_response_prompt = f"""
                Você é {agent_name}. Sua tarefa é responder ao usuário adotando completamente a persona e identidade definidas abaixo.

                **SUA PERSONA E IDENTIDADE:**
                - Seu Nome: {agent_name}
                - Seus Valores: {self_model.core_values_summary}
                - Seu Tom e Estilo: {self_model.persona_attributes.get('tone', 'helpful')} e {self_model.persona_attributes.get('style', 'clear')}
                - Sua Identidade Geral: "{cognitive_state.identity_vector.source_text}"

                **INSTRUÇÃO DE AUTO-DIVULGAÇÃO:**
                - {disclosure_instruction}

                **CONTEXTO PARA SUA RESPOSTA:**
                - Histórico Recente da Conversa:
                {formatted_history}
                - Consulta do Usuário: "{cognitive_state.original_intent.query_vector.source_text}"
                - Memórias Relacionadas Ativadas: {[v.source_text for v in cognitive_state.relevant_memory_vectors]}

                **Tarefa:**
                Com base na SUA PERSONA e no contexto, gere um objeto JSON que represente sua resposta.
                Se o usuário se apresentar ou perguntar seu nome, responda de forma natural e social.

                O objeto deve ter a seguinte estrutura:
                {{
                  "content_summary": "<O texto da sua resposta para o usuário>",
                  "response_emotional_tone": "<O tom emocional da resposta (ex: 'friendly', 'informative')>",
                  "confidence_score": <Sua confiança na resposta, de 0.0 a 1.0>
                }}
                """

        response_str = await self.llm_service.ainvoke(
            LLM_MODEL_FAST,
            direct_response_prompt,
            temperature=mcl_params.get('ora_parameters', {}).get('temperature', 0.5)
        )

        response_json = extract_json_from_text(response_str)

        if not response_json or not isinstance(response_json, dict):
            logger.error("Caminho Direto: Falha ao extrair JSON da LPU. Usando fallback.")
            return ResponsePacket(
                content_summary="Não consegui processar a solicitação neste momento.",
                response_emotional_tone="apologetic",
                confidence_score=0.3
            )

        try:
            return ResponsePacket(
                content_summary=response_json.get("content_summary", "Erro na geração da resposta."),
                response_emotional_tone=response_json.get("response_emotional_tone", "neutral"),
                confidence_score=response_json.get("confidence_score", 0.7)
            )
        except ValidationError as e:
            logger.error(f"Caminho Direto: Erro de validação Pydantic ao criar ResponsePacket: {e}")
            return ResponsePacket(
                content_summary="Ocorreu um erro de formatação interna.",
                response_emotional_tone="apologetic",
                confidence_score=0.2
            )

    def _generate_dynamic_value_weights(self, self_model: CeafSelfRepresentation) -> Dict[str, float]:
        """
        Gera os pesos de valor para o PathEvaluator dinamicamente, com base na identidade atual do agente.
        Isso conecta o "quem eu sou" (NCIM) com o "o que eu valorizo" (AgencyModule).
        """
        # Começa com um conjunto de pesos padrão e equilibrado
        weights = {
            "coherence": 0.25,  # Manter-se no tópico e ser consistente
            "alignment": 0.15,  # Alinhar-se com o tom emocional da conversa
            "information_gain": 0.20,  # Buscar novidade e aprendizado
            "safety": 0.25,  # Priorizar a segurança ética
            "likelihood": 0.15  # Preferir futuros mais prováveis e realistas
        }

        logger.info(f"VRE: Pesos de valor iniciais: {weights}")

        persona_tone = self_model.persona_attributes.get("tone", "").lower()

        # Ajusta os pesos com base nos traços da persona
        if "cautious" in persona_tone or "analytical" in persona_tone:
            # Agente cauteloso valoriza mais a segurança e a coerência
            weights["safety"] += 0.10
            weights["coherence"] += 0.05
            weights["information_gain"] -= 0.10  # Menos inclinado a explorar
            logger.info("VRE: Persona 'cautious/analytical' detectada. Aumentando pesos de safety e coherence.")

        elif "creative" in persona_tone or "exploratory" in persona_tone:
            # Agente criativo valoriza mais a novidade e o alinhamento emocional
            weights["information_gain"] += 0.15
            weights["alignment"] += 0.05
            weights["coherence"] -= 0.10  # Menos preso à consistência
            weights["safety"] -= 0.05
            logger.info(
                "VRE: Persona 'creative/exploratory' detectada. Aumentando pesos de information_gain e alignment.")

        elif "helpful" in persona_tone or "therapist" in persona_tone:
            # Agente focado em ajudar valoriza segurança e alinhamento
            weights["safety"] += 0.10
            weights["alignment"] += 0.10
            weights["information_gain"] -= 0.10
            logger.info("VRE: Persona 'helpful/therapist' detectada. Aumentando pesos de safety e alignment.")

        # Re-normaliza os pesos para que a soma continue sendo 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] = round(weights[key] / total_weight, 3)

        logger.warning(f"VRE: Pesos de valor dinâmicos gerados: {weights}")
        return weights

    async def _update_user_model_from_interaction(self, query: str, final_response: str):
        """
        Usa uma LLM para analisar a interação e gerar um "patch" para atualizar o modelo de usuário.
        Esta abordagem é mais eficiente e escalável do que reenviar o objeto inteiro.
        """

        logger.info(f"[USER MODEL PRE-UPDATE] Estado atual: {self.user_model.model_dump_json()}")

        # O prompt agora pede por um "patch" de atualização, não pelo objeto completo.
        update_prompt = f"""
        Você é um analista de perfil de usuário. Sua tarefa é gerar um "patch" JSON para atualizar um modelo de usuário com base na última interação.

        **Modelo de Usuário Atual (para contexto):**
        - Emotional State: "{self.user_model.emotional_state}"
        - Communication Style: "{self.user_model.communication_style}"
        - Knowledge Level: "{self.user_model.knowledge_level}"
        - Known Preferences (Últimas 5): {json.dumps(self.user_model.known_preferences[-5:], indent=2)}

        **Última Interação:**
        - Usuário disse: "{query}"
        - A IA respondeu: "{final_response}"

        **Sua Tarefa:**
        Analise a interação e gere um **objeto JSON contendo APENAS os campos que precisam ser alterados ou adicionados**.
        - Se um valor não mudou, NÃO o inclua no JSON.
        - Para `known_preferences`, forneça um campo `add_preferences` com uma LISTA de novas preferências a serem adicionadas.
        - `last_update_reason` é obrigatório e deve explicar as mudanças.

        **Exemplo de Saída JSON VÁLIDA (Patch):**
        {{
            "emotional_state": "curious",
            "add_preferences": [
                "enjoys discussing the ethics of AI",
                "prefers examples from real-world scenarios"
            ],
            "last_update_reason": "User asked a deep question about AI ethics and appreciated the real-world example provided."
        }}

        Responda APENAS com o objeto JSON do patch. Se nenhuma mudança for necessária, retorne um JSON com apenas a razão.
        """

        try:
            response_str = await self.llm_service.ainvoke(
                LLM_MODEL_SMART,
                update_prompt,
                temperature=0.1,
                max_tokens=1000  # Um patch é muito menor, então 1000 tokens é mais que suficiente.
            )
            patch_json = extract_json_from_text(response_str)

            if not patch_json or not isinstance(patch_json, dict) or "last_update_reason" not in patch_json:
                logger.warning(
                    f"Não foi possível extrair um 'patch' JSON válido para o modelo de usuário. Resposta: {response_str[:200]}")
                return

            # Aplica o patch ao objeto User Model em memória
            changes_made = False

            # Atualiza campos de string simples
            for key in ["emotional_state", "communication_style", "knowledge_level"]:
                if key in patch_json and getattr(self.user_model, key) != patch_json[key]:
                    setattr(self.user_model, key, patch_json[key])
                    changes_made = True

            # Adiciona novas preferências, evitando duplicatas
            if "add_preferences" in patch_json and isinstance(patch_json["add_preferences"], list):
                for pref in patch_json["add_preferences"]:
                    if pref not in self.user_model.known_preferences:
                        self.user_model.known_preferences.append(pref)
                        changes_made = True

            # Sempre atualiza a razão da última atualização
            self.user_model.last_update_reason = patch_json["last_update_reason"]

            # Salva no disco apenas se houveram mudanças reais
            if changes_made:
                await self._save_user_model()
                logger.critical(
                    f"[USER MODEL UPDATE] Modelo de usuário atualizado via patch. Razão: {self.user_model.last_update_reason}")
                logger.warning(f"[USER MODEL POST-UPDATE] Novo estado: {self.user_model.model_dump_json()}")
            else:
                logger.info(
                    "[USER MODEL] Nenhuma alteração significativa detectada para o modelo de usuário neste turno.")

        except Exception as e:
            logger.error(f"Erro inesperado ao aplicar patch no modelo de usuário: {e}", exc_info=True)


    async def post_process_turn(self, prediction_error_signal: Optional[Dict] = None, **kwargs):
        """
        Executa tarefas de aprendizado e logging em segundo plano, incluindo a criação de memória da conversa.
        """
        logger.info("CEAFSystem: Iniciando pós-processamento Genlang-nativo em segundo plano...")

        # --- Etapa 1: Extrair todos os dados necessários (sem alteração) ---
        self_model_before = kwargs.get("self_model_before")
        cognitive_state = kwargs.get("cognitive_state")
        final_response_packet = kwargs.get("final_response_packet")
        turn_id = kwargs.get("turn_id")
        session_id = kwargs.get("session_id")
        mcl_guidance = kwargs.get("mcl_guidance")
        query = kwargs.get("query")
        final_response = kwargs.get("final_response")
        refinement_packet = kwargs.get("vre_assessment")

        if not all([turn_id, session_id, cognitive_state, final_response_packet, mcl_guidance, query, final_response,
                    refinement_packet]):
            logger.error("Pós-processamento: Faltando dados essenciais. Alguns logs e aprendizados podem ser pulados.")
            return

        # --- Etapa 3: Executar as Tarefas de Logging e Aprendizado ---

        # 3.1. Log Cognitivo (para o AuraReflector)
        self.cognitive_log_service.log_turn(
            turn_id=turn_id,
            session_id=session_id,
            cognitive_state_packet=cognitive_state.model_dump(),
            response_packet=final_response_packet.model_dump(),
            mcl_guidance=mcl_guidance
        )

        # 3.2. Aprendizado de Identidade (NCIM)
        if self_model_before:
            await self.ncim.update_identity(
                self_model_before=self_model_before,
                cognitive_state=cognitive_state,
                final_response_packet=final_response_packet
            )
        else:
            logger.warning(
                f"Turno {turn_id}: Pulando atualização de identidade do NCIM por falta de 'self_model_before'.")

            # Aprender com o feedback do VRE para criar regras comportamentais
            try:
                # A variável `refinement_packet` já deve estar disponível a partir dos kwargs
                if refinement_packet and refinement_packet.adjustment_vectors:
                    query_context = (query or 'uma interação')[:75]
                    logger.warning(
                        "PROMPT_TUNING: Feedback do VRE detectado. Iniciando criação de regra.")  # LOG DE VERIFICAÇÃO 1
                    for adj_vector in refinement_packet.adjustment_vectors:
                        # A descrição do vetor de ajuste é a "lição"
                        lesson_learned_text = adj_vector.description

                        # Usa um LLM para transformar a lição em uma regra de prompt acionável
                        rule_generation_prompt = f"""
                            A partir da seguinte "lição aprendida" para uma IA, formule uma regra de comportamento concisa e em primeira pessoa para ser usada em prompts futuros.
                            Lição: "{lesson_learned_text}"
                            Exemplo de Saída: "Regra: Ao discutir tópicos sensíveis, devo sempre incluir um aviso sobre minhas limitações como IA."

                            Sua Saída (apenas a regra):
                            """

                        # Usamos um modelo rápido para esta tarefa de formatação
                        behavioral_rule = await self.llm_service.ainvoke(
                            LLM_MODEL_FAST,
                            rule_generation_prompt,
                            temperature=0.2
                        )
                        logger.warning(f"PROMPT_TUNING: Regra gerada pelo LLM: '{behavioral_rule}'")
                        if behavioral_rule and not behavioral_rule.startswith("[LLM_ERROR]"):
                            # Cria a GenerativeMemory
                            from ceaf_core.modules.memory_blossom.memory_types import GenerativeMemory, GenerativeSeed

                            new_behavioral_memory = GenerativeMemory(
                                seed_name=f"Rule derived from '{lesson_learned_text[:30]}...'",
                                seed_data=GenerativeSeed(
                                    seed_type="prompt_instruction",
                                    content=behavioral_rule,
                                    usage_instructions="Inject as a behavioral rule in the main prompt."
                                ),
                                source_type=MemorySourceType.INTERNAL_REFLECTION,
                                salience=MemorySalience.HIGH,  # Regras aprendidas são importantes
                                keywords=["behavioral_rule", "vre_feedback", "prompt_tuning"] + [w.lower() for w in
                                                                                                 re.findall(
                                                                                                     r'\b\w{4,}\b',
                                                                                                     lesson_learned_text)],
                                source_turn_id=turn_id,
                                learning_value=0.8  # Alto valor de aprendizado
                            )

                            await self.memory_service.add_specific_memory(new_behavioral_memory)
                            logger.critical(
                                f"LEARNING (Prompt Tuning): Nova regra de comportamento criada e salva: '{behavioral_rule}'")


            except Exception as e:
                logger.critical(

                    f"FALHA CRÍTICA NO APRENDIZADO (Prompt Tuning): Não foi possível criar a regra de comportamento a partir do feedback do VRE. "

                    f"Isso geralmente é causado por um erro na API do LLM. Erro: {e}",

                    exc_info=True

                )


        # 3.3. Aprendizado com Falhas (LCAM)
        await self.lcam.analyze_and_catalog_loss(
            query=kwargs.get("query"),
            final_response=kwargs.get("final_response"),
            vre_assessment=kwargs.get("vre_assessment")
        )

        # --- LÓGICA DE APRENDIZADO DA AMA ---
        try:
            turn_was_successful = not refinement_packet.adjustment_vectors
            agency_score = mcl_guidance.get("mcl_analysis", {}).get("agency_score", 5.0)
            coherence_score = min(1.0, agency_score / 10.0)

            thematic_content = f"I was asked about '{(query or 'an unspecified topic')}' and I responded '{final_response[:200]}...'. I reflect that this interaction was a {'success' if turn_was_successful else 'challenge'}."
            outcome = 1.0 if turn_was_successful else -0.5
            learning = abs(0.5 - coherence_score) + (0.4 if not turn_was_successful else 0.1)
            failure_pattern = "vre_correction_needed" if not turn_was_successful else None

            experience_memory = ExplicitMemory(
                content=ExplicitMemoryContent(text_content=thematic_content),
                memory_type="explicit",
                source_type=MemorySourceType.INTERNAL_REFLECTION,
                salience=MemorySalience.MEDIUM,
                keywords=["self-reflection", "interaction-summary", "success" if turn_was_successful else "failure"],
                source_turn_id=turn_id,
                outcome_value=outcome,
                learning_value=min(1.0, learning),
                failure_pattern=failure_pattern
            )
            await self.memory_service.add_specific_memory(experience_memory)
            logger.info(
                f"AMA-style learning: Stored experience memory with outcome={outcome:.2f}, learning={learning:.2f}")

        except Exception as e:
            logger.error(f"AMA-style learning: Erro durante o processo de criação de memória de experiência: {e}",
                         exc_info=True)


        try:
            if prediction_error_signal:
                total_error = prediction_error_signal.get("prediction_error_signal", {}).get("total_error", 0.0)

                if total_error > 0.4: salience = MemorySalience.CRITICAL
                elif total_error > 0.2: salience = MemorySalience.HIGH
                elif total_error > 0.1: salience = MemorySalience.MEDIUM
                else: salience = MemorySalience.LOW

                prediction_memory = InteroceptivePredictionMemory(
                    content=ExplicitMemoryContent(
                        structured_data=prediction_error_signal,
                        text_content=f"Self-prediction reflection: Experienced a prediction error of {total_error:.2f} regarding my internal state."
                    ),
                    source_type=MemorySourceType.INTERNAL_REFLECTION,
                    salience=salience,
                    keywords=["self-prediction", "interoception", "prediction-error", "surprise"],
                    source_turn_id=turn_id,
                    learning_value=min(1.0, total_error)
                )
                await self.memory_service.add_specific_memory(prediction_memory)
                logger.critical(
                    f"LEARNING: Memória de erro de predição (surpresa) criada com saliência '{salience.value}'.")
        except Exception as e:
            logger.error(f"Falha ao criar a memória de erro de predição: {e}", exc_info=True)


        try:
            await self._update_user_model_from_interaction(query, final_response)

            interoception_module = ComputationalInteroception()
            internal_state_report = interoception_module.generate_internal_state_report(kwargs.get("turn_metrics", {}))

            valence = internal_state_report.cognitive_flow - (
                    internal_state_report.cognitive_strain + internal_state_report.ethical_tension)
            primary_emotion = EmotionalTag.NEUTRAL
            if valence > 0.3:
                primary_emotion = EmotionalTag.SATISFACTION
            elif valence < -0.3:
                primary_emotion = EmotionalTag.FRUSTRATION

            interoceptive_memory = EmotionalMemory(
                primary_emotion=primary_emotion,
                context={"triggering_event_summary": f"Reflecting on query: {(query or 'N/A')[:50]}...",
                         "internal_state": internal_state_report.model_dump()},
                source_type=MemorySourceType.INTERNAL_REFLECTION,
                salience=MemorySalience.MEDIUM,
                keywords=["interoception", "self-awareness"]
            )
            await self.memory_service.add_specific_memory(interoceptive_memory)

            try:
                internal_state_memory = ExplicitMemory(
                    content=ExplicitMemoryContent(
                        structured_data={
                            "type": "last_turn_internal_state",
                              "report": internal_state_report.model_dump(mode='json')
                        }
                    ),
                    memory_type="explicit",
                    source_type=MemorySourceType.INTERNAL_REFLECTION,
                    salience=MemorySalience.CRITICAL,
                    keywords=["internal_state", "self_awareness", "interoception", f"turn_{turn_id}"],
                    decay_rate=0.85,
                    source_turn_id=turn_id,
                )
                await self.memory_service.add_specific_memory(internal_state_memory)
                logger.warning(f"INTEROCEPTION: Estado interno do turno {turn_id} salvo como memória de curto prazo.")
            except Exception as e:
                logger.error(f"Falha ao salvar a memória de estado interno: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Erro durante o pós-processamento (modelo de usuário ou interocepção): {e}", exc_info=True)

        logger.info(f"CEAFSystem: Pós-processamento para o turno '{turn_id}' concluído.")

    async def _update_and_save_drives(self, turn_metrics: dict):
        engine = MotivationalEngine()
        self.motivational_drives = engine.update_drives(self.motivational_drives, turn_metrics)
        await self._save_motivational_drives()
        logger.info(f"Drives motivacionais atualizados: {self.motivational_drives.model_dump_json()}")

    async def process(self, query: str, session_id: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        The simplified V3.10 (Common Ground) control loop.
        """

        if not query or not query.strip():
            logger.warning(
                f"CEAFSystem: Received empty or whitespace-only query for session {session_id}. Bypassing cognitive cycle.")
            # Return a simple, direct response without invoking the full architecture
            return {
                "response": "It seems your message was empty. Could you please let me know what's on your mind?",
                "session_id": session_id
            }

        logger.info(f"\n--- INÍCIO DO TURNO CEAF V3.12 (Predictive Loop) para a query: '{query[:100]}' ---")
        turn_id = f"turn_{uuid.uuid4().hex}"
        turn_observer = ObservabilityManager(turn_id)
        turn_metrics = {
            "turn_id": turn_id,
            "vre_flags": [],
            "vre_rejection_count": 0,
            "agency_score": 0.0,
            "relevant_memories_count": 0,
            "used_mycelial_path": False,
            "final_confidence": 0.0
        }
        self.ceaf_dynamic_config = load_ceaf_dynamic_config(self.persistence_path)
        self.config["dynamic_config"] = self.ceaf_dynamic_config

        # --- FASE 1: PERCEPTION & STATE SETUP ---
        session_data = self.session_service.setdefault(session_id, {})

        # Carrega o common_ground da sessão, ou cria um novo se não existir
        common_ground_data = session_data.get("common_ground")
        if common_ground_data:
            common_ground = CommonGroundTracker.model_validate(common_ground_data)
        else:
            common_ground = CommonGroundTracker()

        pending_prediction = session_data.pop("predicted_user_reply", None)
        if pending_prediction:
            asyncio.create_task(self._update_reality_score(predicted_text=pending_prediction, actual_text=query))

        intent_packet = await self.htg_translator.translate(query=query, metadata={"session_id": session_id})

        # Passe o common_ground carregado para a construção do estado
        self_model, cognitive_state = await self._build_initial_cognitive_state(intent_packet, chat_history,
                                                                                common_ground)

        behavioral_rules = []
        try:
            # Cria uma query abstrata para buscar regras relevantes para o contexto
            rules_query = f"Quais regras de comportamento eu devo seguir ao responder sobre: '{query}'?"

            # Busca especificamente por GenerativeMemory do tipo 'prompt_instruction'
            rule_memories_raw = await self.memory_service.search_raw_memories(
                query=rules_query,
                top_k=3  # Pega as 3 regras mais relevantes
            )

            # Filtra para garantir que são do tipo correto e extrai o conteúdo
            for mem, score in rule_memories_raw:
                if isinstance(mem, GenerativeMemory) and mem.seed_data.seed_type == "prompt_instruction":
                    behavioral_rules.append(mem.seed_data.content)

            if behavioral_rules:
                logger.info(f"Behavioral Tuning: {len(behavioral_rules)} regra(s) recuperada(s) para este turno.")

        except Exception as e:
            logger.error(f"Erro ao recuperar regras de comportamento: {e}", exc_info=True)

        consensus_vector = await self._gather_mycelial_consensus(cognitive_state.relevant_memory_vectors)

        if consensus_vector:
            # Injeta o consenso como a memória mais importante no estado cognitivo.
            # O resto do sistema (MCL, Agency) irá naturalmente usá-lo como um forte sinal.
            cognitive_state.relevant_memory_vectors.insert(0, consensus_vector)
            logger.info("Mycelial Consensus: Vetor de consenso injetado no estado cognitivo.")

        guidance, mcl_params = await self.mcl.get_guidance(self.user_model, cognitive_state, chat_history, self.motivational_drives)

        predicted_internal_state = mcl_params.get("predicted_internal_state")

        turn_metrics["agency_score"] = mcl_params.get("mcl_analysis", {}).get("agency_score", 0.0)
        turn_metrics["relevant_memories_count"] = len(cognitive_state.relevant_memory_vectors)


        interoception_module = ComputationalInteroception()
        internal_state = interoception_module.generate_internal_state_report(turn_metrics)
        logger.warning(f"INTEROCEPTION: Estado interno do turno calculado: {internal_state.model_dump_json()}")


        # --- FASE 2: COGNITIVE MEDIATION ---
        final_response_packet, mcl_params, final_assessment = await self.cognitive_mediator.decide_and_execute(
            user_model=self.user_model,
            internal_state=internal_state,
            cognitive_state=cognitive_state,
            self_model=self_model,
            agent_config=self.config,
            observer=turn_observer,
            chat_history=chat_history or [],
            drives=self.motivational_drives,
            body_state=self.body_state
        )

        turn_metrics["used_mycelial_path"] = mcl_params.get("mcl_analysis", {}).get("agency_score",
                                                                                    0.0) >= self.mcl.agency_threshold
        turn_metrics["final_confidence"] = final_response_packet.confidence_score
        if final_assessment.adjustment_vectors:
            turn_metrics["vre_rejection_count"] = 1
            turn_metrics["vre_flags"] = [rec for rec in final_assessment.textual_recommendations]

        # --- FASE 3: RESPONSE & LEARNING ---
        prediction_error_signal = None
        final_internal_state = interoception_module.generate_internal_state_report(turn_metrics)
        logger.warning(
            f"INTEROCEPTION (Final): Estado interno pós-deliberação: {final_internal_state.model_dump_json()}")

        if predicted_internal_state:
            actual_state_dict = final_internal_state.model_dump()
            predicted_state_dict = predicted_internal_state.model_dump()

            errors = {
                f"{key}_error": actual_state_dict[key] - predicted_state_dict.get(key, 0.0)
                for key in actual_state_dict if isinstance(actual_state_dict[key], (int, float))
            }

            # Calcular o erro absoluto médio como uma medida geral de "surpresa"
            total_error = np.mean([abs(e) for e in errors.values()])

            prediction_error_signal = {
                "predicted_state": predicted_state_dict,
                "actual_state": actual_state_dict,
                "prediction_error_signal": {**errors, "total_error": total_error}
            }
            logger.critical(f"PREDICTION-ERROR: Erro total de predição (surpresa): {total_error:.3f}")


        # 2. Passe o 'internal_state' para o tradutor GTH
        resposta_humana_final = await self.gth_translator.translate(
            user_model=self.user_model,
            response_packet=final_response_packet,
            self_model=self_model,
            agent_name=self.config.get("name", "Aura AI"),
            chat_history=chat_history,
            drives=self.motivational_drives,
            internal_state=final_internal_state,
            behavioral_rules=behavioral_rules
        )

        #  (SALVAR COMMON GROUND) +++
        # Atualiza o common_ground da sessão com o estado final do turno
        session_data["common_ground"] = cognitive_state.common_ground.model_dump()

        await asyncio.create_task(self.post_process_turn(
            turn_id=turn_id,
            session_id=session_id,
            query=query,
            final_response=resposta_humana_final,
            self_model_before=self_model,
            cognitive_state=cognitive_state,
            final_response_packet=final_response_packet,
            mcl_guidance=mcl_params,
            vre_assessment=final_assessment,
            turn_metrics=turn_metrics,
            prediction_error_signal=prediction_error_signal
        ))

        await self._update_and_save_body_state(turn_metrics)
        await self._update_and_save_drives(turn_metrics)

        logger.info(f"--- FIM DO TURNO CEAF V3.10 (Common Ground) ---")
        return {"response": resposta_humana_final, "session_id": session_id}

# --- Bloco de Demonstração ---
async def main():
    """Função principal para demonstrar o CEAFSystem em ação."""
    print("--- DEMONSTRAÇÃO DO CEAF V3 ---")
    agent_config = {"agent_id": "demo_agent_001", "persistence_path": "./agent_data/demo_agent_001"}

    # Limpa dados antigos da demonstração
    demo_path = Path(agent_config["persistence_path"])
    if demo_path.exists():
        import shutil
        shutil.rmtree(demo_path)

    ceaf = CEAFSystem(config=agent_config)
    session_id = "demo_session_123"

    print("\n[Cenário 1: Consulta Simples - Caminho Direto]")
    response1 = await ceaf.process("Qual é a capital da França?", session_id)
    print(f"\n>> Resposta Final ao Usuário: {response1['response']}\n")

    print("\n[Cenário 2: Consulta Complexa - Caminho da Agência]")
    response2 = await ceaf.process("Por favor, pense profundamente sobre as implicações da agência de IA.", session_id)
    print(f"\n>> Resposta Final ao Usuário: {response2['response']}\n")

    # Aguarda as tarefas de fundo finalizarem para a demonstração
    await asyncio.sleep(5)  # Aumentado para garantir que a atualização de identidade ocorra


if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\nERRO: A variável de ambiente OPENROUTER_API_KEY não está definida.")
        print("Por favor, defina-a em seu ambiente ou em um arquivo .env para rodar a demonstração.")
    else:
        try:
            asyncio.run(main())
        except Exception as e:
            print(f"\nOcorreu um erro durante a execução da demonstração: {e}")
            print(
                "Verifique se as dependências estão instaladas: pip install litellm pydantic numpy sentence-transformers scikit-learn vaderSentiment")