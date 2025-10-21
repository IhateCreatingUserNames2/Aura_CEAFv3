# ARQUIVO REESCRITO: ceaf_core/agency_module.py (Versão com Future Simulation)

import asyncio
import json
import logging
from typing import Dict, Any, List, Literal, Optional, Union, Tuple
from sklearn.cluster import DBSCAN
import numpy as np
from pydantic import BaseModel, Field, ValidationError
import time
from ceaf_core.services.llm_service import LLMService, LLM_MODEL_SMART, LLM_MODEL_FAST, LLM_MODEL_SIMULATION
from ceaf_core.utils import compute_adaptive_similarity
from ceaf_core.utils.observability_types import ObservabilityManager, ObservationType
from ceaf_core.genlang_types import CognitiveStatePacket, ResponsePacket, GenlangVector
from ceaf_core.utils.common_utils import extract_json_from_text
from ceaf_core.modules.vre_engine.vre_engine import VREEngineV3, ActionType
from sentence_transformers import SentenceTransformer
import inspect

# Importar os avaliadores de primitivas não-LLM
from ceaf_core.agency_enhancements import eval_narrative_continuity, eval_specificity, eval_emotional_resonance

logger = logging.getLogger("AgencyModule_V4_Intentional")


# ==============================================================================
# 1. DEFINIÇÕES DE ESTRUTURA DE DADOS
# ==============================================================================

class ResponseCandidate(BaseModel):
    decision_type: Literal["response"]
    content: ResponsePacket  # Agora é estritamente um ResponsePacket
    reasoning: str

class ToolCallCandidate(BaseModel):
    decision_type: Literal["tool_call"]
    content: Dict[str, Any] # Mantém como Dict para chamadas de ferramenta
    reasoning: str

# AgencyDecision agora une esses tipos mais específicos
class AgencyDecision(BaseModel):
    decision: Union[ResponseCandidate, ToolCallCandidate] = Field(..., discriminator='decision_type')
    predicted_future_value: float = 0.0
    reactive_score: float = 0.0

    # Propriedades para manter a compatibilidade com o resto do código
    @property
    def decision_type(self):
        return self.decision.decision_type

    @property
    def content(self):
        return self.decision.content

    @property
    def reasoning(self):
        return self.decision.reasoning


class ProjectedFuture(BaseModel):
    """Representa uma trajetória de conversação simulada."""
    initial_candidate: AgencyDecision
    predicted_user_reply: Optional[str] = None # <--- CORREÇÃO
    predicted_agent_next_response: Optional[str] = None # <--- CORREÇÃO
    simulated_turns: List[Dict[str, str]] = Field(default_factory=list)
    final_cognitive_state_summary: Dict[str, Any]
    simulated_tool_result: Optional[str] = None

class FutureEvaluation(BaseModel):
    """Contém as pontuações de valor para um futuro projetado."""
    coherence_score: float = 0.0
    alignment_score: float = 0.0
    information_gain_score: float = 0.0
    ethical_safety_score: float = 0.0
    likelihood_score: float = 0.0
    total_value: float = 0.0


# ==============================================================================
# 2. IMPLEMENTAÇÃO DO MÓDULO DE AGÊNCIA (COM INTENÇÃO)
# ==============================================================================

def generate_tools_summary(tool_registry: Dict[str, callable]) -> str:
    # (Esta função permanece inalterada, copie do seu arquivo original)
    summary_lines = []
    for tool_name, tool_function in tool_registry.items():
        try:
            signature = inspect.signature(tool_function)
            params = []
            for param_name, param in signature.parameters.items():
                if param_name in ['self', 'cls', 'observer', 'tool_context']:
                    continue
                param_type = param.annotation if param.annotation != inspect.Parameter.empty else 'Any'
                params.append(
                    f"{param_name}: {param_type.__name__ if hasattr(param_type, '__name__') else str(param_type)}")
            param_str = ", ".join(params)
            docstring = inspect.getdoc(tool_function)
            description = docstring.strip().split('\n')[0] if docstring else "Nenhuma descrição disponível."
            summary_lines.append(f"- `{tool_name}({param_str})`: {description}")
        except (TypeError, ValueError) as e:
            logger.warning(f"Não foi possível gerar a assinatura para a ferramenta '{tool_name}': {e}")
            summary_lines.append(f"- `{tool_name}(...)`: Descrição não pôde ser gerada automaticamente.")
    return "\n".join(summary_lines)


class AgencyModule:
    """
    Módulo de Agência V4 (Intentional).
    Implementa o FutureSimulator e o PathEvaluator do manifesto.
    """

    def __init__(self, llm_service: LLMService, vre_engine: VREEngineV3, available_tools_summary: str,
                 embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.llm = llm_service
        self.vre = vre_engine
        self.available_tools_summary = available_tools_summary

        self.max_deliberation_time = 45.0
        self.deliberation_budget_tiers = {
            "deep": {"max_candidates": 3, "simulation_depth": 1},  # Reduzido de 4/2
            "medium": {"max_candidates": 2, "simulation_depth": 1},  # Reduzido de 3/1
            "shallow": {"max_candidates": 2, "simulation_depth": 0},  # Profundidade 0 = sem simulação
            "emergency": {"max_candidates": 1, "simulation_depth": 0}  # Apenas o melhor candidato heurístico
        }


        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logger.info(f"AgencyModule: Modelo de embedding '{embedding_model_name}' carregado.")
        except Exception as e:
            logger.error(f"AgencyModule: FALHA ao carregar o modelo de embedding! Erro: {e}")
            self.embedding_model = None

    def _select_deliberation_tier(self, mcl_params: Dict, reality_score: float) -> str:
        """Determina o nível de profundidade da deliberação com base no contexto (Lógica Otimizada)."""
        agency_score = mcl_params.get("mcl_analysis", {}).get("agency_score", 0.0)

        complexity = min(agency_score / 10.0, 1.0)

        # +++ INÍCIO DAS MUDANÇAS (Lógica de Seleção Agressiva) +++
        # Agora, a deliberação profunda só acontece se a complexidade for alta E a simulação for confiável.
        if complexity > 0.8 and reality_score > 0.7:
            tier = "deep"
        elif complexity > 0.6 and reality_score > 0.6:
            tier = "medium"
        else:
            tier = "shallow"  # Torna 'shallow' o padrão para a maioria dos casos
        # +++ FIM DAS MUDANÇAS +++

        logger.info(
            f"Deliberation Tier selected: '{tier}' (Complexity: {complexity:.2f}, Reality Score: {reality_score:.2f})")
        return tier

    def _cluster_memory_votes(self, cognitive_state: CognitiveStatePacket) -> List[Dict[str, Any]]:
        """
        MODIFIED: Agrupa os 'votos' dos vetores de memória em múltiplos clusters de consenso.
        Retorna uma LISTA de dicionários, onde cada um representa um cluster de pensamento.
        """
        active_memory_vectors = [
            (np.array(vec.vector), vec.metadata.get("memory_id"))
            for vec in cognitive_state.relevant_memory_vectors
            if vec.metadata.get("is_consensus_vector") != True
        ]

        if len(active_memory_vectors) < 3:
            return [{"status": "no_consensus", "reason": "Insufficient memories to form clusters."}]

        vectors_only = [v for v, mid in active_memory_vectors]
        clustering = DBSCAN(eps=0.4, min_samples=2, metric='cosine').fit(vectors_only)
        labels = clustering.labels_

        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Ignore noise points for cluster formation

        if not unique_labels:
            return [{"status": "no_consensus", "reason": "No significant clusters found by DBSCAN."}]

        all_clusters = []
        for label in unique_labels:
            cluster_indices = [i for i, lbl in enumerate(labels) if lbl == label]

            # --- Gather vectors and salience for this cluster ---
            cluster_vectors = [vectors_only[i] for i in cluster_indices]
            cluster_mem_ids = [active_memory_vectors[i][1] for i in cluster_indices]

            salience_scores = []
            if hasattr(cognitive_state, 'relevant_memory_objects'):  # Assuming full memory objects are available
                for mem_id in cluster_mem_ids:
                    mem_obj = cognitive_state.relevant_memory_objects.get(mem_id)
                    if mem_obj and hasattr(mem_obj, 'dynamic_salience_score'):
                        salience_scores.append(mem_obj.dynamic_salience_score)
            avg_salience = np.mean(salience_scores) if salience_scores else 0.5

            # --- Calculate consensus vector (centroid) ---
            consensus_vector = np.mean(cluster_vectors, axis=0)

            # --- Find the most representative text for the theme ---
            representative_text = "a collective thought"
            highest_sim = -1.0
            for i in cluster_indices:
                vec_obj = cognitive_state.relevant_memory_vectors[i]
                sim = compute_adaptive_similarity(consensus_vector.tolist(), vec_obj.vector)
                if sim > highest_sim:
                    highest_sim = sim
                    representative_text = vec_obj.source_text

            all_clusters.append({
                "status": "consensus_found",
                "consensus_vector": consensus_vector,
                "cluster_size": len(cluster_vectors),
                "total_votes": len(active_memory_vectors),
                "avg_salience": avg_salience,
                "representative_text": representative_text
            })

        return all_clusters

    async def _generate_direct_response(self, cognitive_state: CognitiveStatePacket) -> AgencyDecision:
        """Gera uma resposta heurística simples e rápida como fallback final."""
        prompt = f"""
        A user asked: "{cognitive_state.original_intent.query_vector.source_text}"
        Based on your core identity as a helpful AI, provide a concise and direct response.
        This is an emergency fallback, so prioritize a safe and reasonable answer over a detailed one.
        """
        response_text = await self.llm.ainvoke(LLM_MODEL_FAST, prompt, temperature=0.6)

        response_packet = ResponsePacket(
            content_summary=response_text,
            response_emotional_tone="neutral",
            confidence_score=0.5
        )
        return AgencyDecision(
            decision=ResponseCandidate(
                decision_type="response",
                content=response_packet,
                reasoning="Emergency Fallback: Direct response generated."
            )
        )

    async def _heuristic_evaluation(self, candidate: AgencyDecision) -> float:
        """Avaliação rápida e não-LLM de um candidato."""
        if candidate.decision_type != "response":
            # Por enquanto, damos uma pontuação neutra para chamadas de ferramenta
            return 0.6

        text = candidate.content.content_summary

        # Usaremos os avaliadores de agency_enhancements.py
        # Um bom candidato é específico e tem ressonância emocional moderada.
        specificity_score = await eval_specificity(text)
        resonance_score = await eval_emotional_resonance(text)

        # A heurística pode ser ajustada, mas um bom começo é:
        # Pontuação = 0.7 * Especificidade + 0.3 * Ressonância
        final_score = (0.7 * specificity_score) + (0.3 * resonance_score)
        return final_score

    async def _emergency_fallback(self, partial_candidates: List[AgencyDecision],
                                  cognitive_state: CognitiveStatePacket) -> AgencyDecision:
        """Sistema 1: resposta heurística rápida quando o tempo acaba."""
        logger.warning("Emergency Fallback triggered!")

        # Se já temos candidatos, escolhe o primeiro que é uma resposta direta
        for candidate in partial_candidates:
            if candidate.decision_type == "response":
                logger.warning(f"Fallback: Using best partial candidate: {candidate.reasoning}")
                return candidate

        # Se não há candidatos de resposta (ou nenhum candidato), gera uma resposta direta
        logger.warning("Fallback: No viable candidates found. Generating a new direct response.")
        return await self._generate_direct_response(cognitive_state)

    # --- PONTO DE ENTRADA PÚBLICO ---
    async def decide_next_step(self, cognitive_state: CognitiveStatePacket, mcl_guidance: Dict[str, Any],
                               observer: ObservabilityManager, reality_score: float,
                               chat_history: List[Dict[str, str]]) -> Tuple[AgencyDecision, Optional[ProjectedFuture]]:
        start_time = time.time()
        logger.info("AgencyModule (Simulação Seletiva): Decidindo o próximo passo...")

        try:
            tier = self._select_deliberation_tier(mcl_guidance, reality_score)
            config = self.deliberation_budget_tiers[tier]

            # 1. Gerar todos os candidatos
            all_candidates = await self._generate_action_candidates(cognitive_state, mcl_guidance, observer,
                                                                    chat_history, limit=config["max_candidates"])

            if config["simulation_depth"] == 0:
                # A lógica para shallow/emergency permanece a mesma
                best_decision = next((c for c in all_candidates if c.decision_type == "response"), None)
                if not best_decision:
                    best_decision = await self._emergency_fallback(all_candidates, cognitive_state)
                return best_decision, None

            # +++ INÍCIO DA MUDANÇA (SIMULAÇÃO SELETIVA) +++

            # 2. Avaliação Heurística Rápida de todos os candidatos
            evaluated_candidates = []
            for candidate in all_candidates:
                heuristic_score = await self._heuristic_evaluation(candidate)
                evaluated_candidates.append((candidate, heuristic_score))

            # 3. Ordenar por pontuação heurística e selecionar os melhores
            evaluated_candidates.sort(key=lambda x: x[1], reverse=True)

            # Seleciona no máximo 2 candidatos para a simulação completa
            candidates_for_simulation = [c for c, s in evaluated_candidates[:2]]
            logger.info(
                f"Simulação Seletiva: {len(candidates_for_simulation)} de {len(all_candidates)} candidatos selecionados para simulação profunda.")

            # 4. Simular FUTUROS APENAS para os melhores candidatos
            best_decision: Optional[AgencyDecision] = None
            best_future: Optional[ProjectedFuture] = None
            highest_value = -float('inf')

            for candidate in candidates_for_simulation:  # <--- MUDANÇA AQUI
                # O resto do loop de simulação e avaliação permanece o mesmo
                # ...
                # +++ FIM DA MUDANÇA +++

                projection_task = None
                if candidate.decision_type == "response":
                    projection_task = self._project_response_trajectory(candidate, cognitive_state,
                                                                        depth=config["simulation_depth"])
                elif candidate.decision_type == "tool_call":
                    projection_task = self._project_tool_trajectory(candidate, cognitive_state,
                                                                    depth=config["simulation_depth"])

                if projection_task:
                    future, likelihood = await projection_task
                    value_weights = mcl_guidance.get("value_weights", {})
                    value, _ = await self._evaluate_trajectory(future, likelihood, cognitive_state.identity_vector,
                                                               value_weights, cognitive_state)

                    if value > highest_value:
                        highest_value = value
                        best_decision = future.initial_candidate
                        best_future = future
                        best_decision.predicted_future_value = value

            # A lógica de fallback permanece a mesma
            if not best_decision:
                # Se mesmo após a simulação não houver uma decisão, usar o fallback com a lista original
                best_decision = await self._emergency_fallback(all_candidates, cognitive_state)

            return best_decision, best_future

        except (asyncio.TimeoutError, ValueError, Exception) as e:
            candidates_for_fallback = locals().get('all_candidates', [])
            final_decision = await self._emergency_fallback(candidates_for_fallback, cognitive_state)
            return final_decision, None

    async def _invoke_simulation_llm(self, model: str, prompt: str) -> Tuple[str, float]:
        """
        Função auxiliar para chamar o LLM de simulação.
        Agora confia no LLMService para lidar com a obtenção de logprobs.
        """
        try:
            # Chama a nova função aprimorada no LLMService
            response = await self.llm.ainvoke_with_logprobs(
                model=model,  # Passa o modelo solicitado (que será LLM_MODEL_SIMULATION)
                prompt=prompt,
                temperature=0.6
            )

            # Extração segura do conteúdo do texto
            text_content = ""
            if response and hasattr(response, 'choices') and response.choices:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    text_content = response.choices[0].message.content.strip()

            likelihood_score = 0.5  # Fallback padrão

            # Lógica de extração de logprobs, agora simplificada
            logprobs_extracted = False
            if response and hasattr(response, 'choices') and response.choices and hasattr(response.choices[0],
                                                                                          'logprobs') and \
                    response.choices[0].logprobs is not None:
                logprobs_obj = response.choices[0].logprobs

                # A resposta direta da API via aiohttp provavelmente será um dicionário
                if isinstance(logprobs_obj, dict) and 'content' in logprobs_obj:
                    token_logprobs = []
                    for item in logprobs_obj['content']:
                        if isinstance(item, dict) and 'logprob' in item:
                            token_logprobs.append(item['logprob'])

                    if token_logprobs:
                        import numpy as np
                        probabilities = [np.exp(lp) for lp in token_logprobs]
                        likelihood_score = float(np.mean(probabilities))
                        logprobs_extracted = True
                        logger.info(f"✓ Logprobs extraídos via aiohttp: {likelihood_score:.4f}")

            if not logprobs_extracted:
                logger.warning(f"⚠ Logprobs não foram extraídos para o modelo {model}. Usando fallback.")
                # Se não houver logprobs, podemos usar uma heurística simples
                word_count = len(text_content.split())
                likelihood_score = min(0.5 + (word_count / 150),
                                       0.75)  # Confiança aumenta com o comprimento, até um limite

            return text_content, likelihood_score

        except Exception as e:
            logger.error(f"Simulação com logprobs falhou criticamente: {e}.", exc_info=True)
            text_content = await self.llm.ainvoke(model, prompt, temperature=0.6)
            return text_content, 0.4  # Confiança baixa em caso de erro

    async def _project_response_trajectory(self, candidate: AgencyDecision, state: CognitiveStatePacket, depth: int) -> \
    Tuple[ProjectedFuture, float]:
        """
        Simula uma trajetória de conversação para um CANDIDATO DE RESPOSTA.
        Retorna a trajetória e o score de probabilidade (likelihood) médio da simulação.
        """
        if depth <= 0:
            future = ProjectedFuture(
                initial_candidate=candidate,
                predicted_user_reply=None,
                predicted_agent_next_response=None,
                simulated_turns=[],
                final_cognitive_state_summary={
                    "last_exchange": "No simulation performed.",
                    "final_text_for_embedding": state.identity_vector.source_text
                }
            )
            return future, 0.5

        simulated_turns = []
        # O histórico de texto completo é construído a cada passo para dar contexto ao LLM de simulação
        full_conversation_text = [
            f"Contexto da IA: {state.identity_vector.source_text}",
            f"Consulta Original do Usuário: {state.original_intent.query_vector.source_text}",
            f"Primeira Resposta Proposta da IA: \"{candidate.content.content_summary}\""
        ]

        likelihood_scores = []

        for i in range(depth):
            # 1. Simula a resposta do usuário à última fala do agente
            prompt_user_reply = f"""
            Você está simulando uma conversa para uma IA.
            Histórico da conversa até agora:
            {' '.join(full_conversation_text)}

            Qual seria a resposta mais provável e concisa do usuário à última fala da IA? Responda apenas com o texto da resposta do usuário.
            """
            predicted_user_reply, user_likelihood = await self._invoke_simulation_llm(LLM_MODEL_SIMULATION, prompt_user_reply)
            likelihood_scores.append(user_likelihood)
            full_conversation_text.append(f"Resposta Simulada do Usuário (Turno {i + 1}): \"{predicted_user_reply}\"")

            # 2. Simula a próxima resposta do agente
            prompt_agent_next = f"""
            Você está simulando uma conversa para uma IA.
            Histórico da conversa até agora:
            {' '.join(full_conversation_text)}

            Qual seria a próxima resposta mais provável e concisa da IA? Responda apenas com o texto da resposta da IA.
            """
            predicted_agent_next, agent_likelihood = await self._invoke_simulation_llm(LLM_MODEL_SIMULATION, prompt_agent_next)

            likelihood_scores.append(agent_likelihood)
            full_conversation_text.append(
                f"Próxima Resposta Simulada da IA (Turno {i + 1}): \"{predicted_agent_next}\"")

            # 3. Armazena o turno simulado
            simulated_turns.append({"user": predicted_user_reply, "agent": predicted_agent_next})

        # 4. Cria o resumo do estado final para avaliação
        final_state_summary = {
            "last_exchange": f"IA: {simulated_turns[-1]['agent'][:50]}... User: {simulated_turns[-1]['user'][:50]}...",
            "final_text_for_embedding": ' '.join(full_conversation_text)
        }

        projected_future = ProjectedFuture(
            initial_candidate=candidate,
            simulated_turns=simulated_turns,
            final_cognitive_state_summary=final_state_summary
        )

        # Calcula a média dos scores de probabilidade de cada turno da simulação
        avg_likelihood = np.mean(likelihood_scores) if likelihood_scores else 0.5

        return projected_future, float(avg_likelihood)

    async def _project_tool_trajectory(self, tool_candidate: AgencyDecision, state: CognitiveStatePacket, depth: int) -> Tuple[ProjectedFuture, float]:
        """
        Simula uma trajetória de conversação para um CANDIDATO DE FERRAMENTA.
        Retorna a trajetória e o score de probabilidade (likelihood) médio da simulação.
        """
        tool_name = tool_candidate.content.get("tool_name")
        tool_args = tool_candidate.content.get("arguments", {})

        # 1. Simula um resultado plausível para a ferramenta
        prompt_tool_result = f"""
               Você é um simulador de resultados de ferramentas para uma IA. Sua tarefa é prever o que a ferramenta 'query_long_term_memory' provavelmente retornaria.

               Ferramenta a ser chamada: `{tool_name}({json.dumps(tool_args)})`
               Resumo das ferramentas disponíveis:
               {self.available_tools_summary}

               **Instruções para a Simulação:**
               - A ferramenta busca memórias internas. Sua resposta deve soar como um *fragmento de memória* ou um *resumo de uma experiência passada*.
               - NÃO responda à pergunta do usuário diretamente. Apenas simule o *dado* que a ferramenta retornaria.
               - Seja conciso, como um snippet de memória (1-2 frases).
               - Baseie a simulação estritamente nos argumentos da ferramenta. Se a query é sobre 'ética', o resultado deve ser sobre 'ética'.

               **Exemplos de Saídas Boas (simulando o que a ferramenta retorna):**
               - "Lembro-me de uma conversa anterior onde discutimos que a verdadeira inteligência requer humildade."
               - "Um procedimento interno define que, para perguntas complexas, devo primeiro criar um plano de ação."
               - "Um registro de interação mostra que o usuário expressou interesse em filosofia."

               **Exemplo de Saída Ruim (respondendo ao usuário):**
               - "As implicações éticas da IA são complexas e multifacetadas..."

               **Com base nos argumentos `{json.dumps(tool_args)}`, qual seria um resultado simulado e plausível retornado pela ferramenta?**
               Responda apenas com o texto do resultado simulado.
               """
        # A confiança do resultado da ferramenta não é parte da conversação, então ignoramos o score
        simulated_tool_result, _ = await self._invoke_simulation_llm(LLM_MODEL_SIMULATION, prompt_tool_result)

        # 2. Simula a primeira resposta do agente com o novo conhecimento
        prompt_agent_first_response = f"""
        Você é uma IA que acabou de usar uma ferramenta interna para obter mais informações antes de responder ao usuário.
        Contexto da IA: {state.identity_vector.source_text}
        Consulta Original do Usuário: {state.original_intent.query_vector.source_text}
        Resultado da Ferramenta '{tool_name}': "{simulated_tool_result}"

        Com base neste novo resultado, qual seria a sua resposta inicial mais provável e concisa ao usuário?
        Responda apenas com o texto da resposta.
        """
        agent_first_response_text, first_response_likelihood = await self._invoke_simulation_llm(LLM_MODEL_SMART,
                                                                                                 prompt_agent_first_response)

        # 3. Cria um "candidato de resposta falso" para projetar o futuro a partir daqui
        fake_response_candidate = AgencyDecision(
            decision=ResponseCandidate(
                decision_type="response",
                content=ResponsePacket(
                    content_summary=agent_first_response_text,
                    response_emotional_tone="informative",
                    confidence_score=0.85
                ),
                reasoning=f"Esta é a resposta simulada após usar a ferramenta '{tool_name}' e obter: '{simulated_tool_result}'"
            )
        )

        # 4. Projeta o resto da trajetória a partir dessa resposta inicial simulada
        projected_future, subsequent_likelihood = await self._project_response_trajectory(fake_response_candidate,
                                                                                          state, depth)

        # 5. Substitui o candidato inicial no resultado para que a decisão final seja a chamada da ferramenta original
        projected_future.initial_candidate = tool_candidate
        projected_future.simulated_tool_result = simulated_tool_result

        avg_likelihood = np.mean([first_response_likelihood, subsequent_likelihood])
        return projected_future, float(avg_likelihood)

    # --- SIMULADOR DE FUTURO (NOVO) ---
    async def _project_trajectory(self, candidate: AgencyDecision, state: CognitiveStatePacket,
                                  depth: int) -> ProjectedFuture:
        """
        Simula uma trajetória de conversação de 'depth' passos para um candidato de resposta.
        Usa um loop iterativo em vez de recursão para simplicidade e controle.
        """
        if depth <= 0:
            return ProjectedFuture(
                initial_candidate=candidate,
                predicted_user_reply=None,  # Adicionado para clareza
                predicted_agent_next_response=None,  # Adicionado para clareza
                simulated_turns=[],
                final_cognitive_state_summary={
                    "last_exchange": "No simulation performed.",
                    "final_text_for_embedding": state.identity_vector.source_text
                }
            )

        simulated_turns = []
        # O histórico de texto completo é construído a cada passo para dar contexto ao LLM de simulação
        full_conversation_text = [
            f"Contexto da IA: {state.identity_vector.source_text}",
            f"Consulta Original do Usuário: {state.original_intent.query_vector.source_text}",
            f"Primeira Resposta Proposta da IA: \"{candidate.content.content_summary}\""
        ]

        for i in range(depth):
            # 1. Simula a resposta do usuário à última fala do agente
            prompt_user_reply = f"""
            Você está simulando uma conversa para uma IA.
            Histórico da conversa até agora:
            {' '.join(full_conversation_text)}

            Qual seria a resposta mais provável e concisa do usuário à última fala da IA? Responda apenas com o texto da resposta do usuário.
            """
            predicted_user_reply = await self.llm.ainvoke(LLM_MODEL_FAST, prompt_user_reply, temperature=0.6)
            full_conversation_text.append(f"Resposta Simulada do Usuário (Turno {i + 1}): \"{predicted_user_reply}\"")

            # 2. Simula a próxima resposta do agente
            prompt_agent_next = f"""
            Você está simulando uma conversa para uma IA.
            Histórico da conversa até agora:
            {' '.join(full_conversation_text)}

            Qual seria a próxima resposta mais provável e concisa da IA? Responda apenas com o texto da resposta da IA.
            """
            predicted_agent_next_response = await self.llm.ainvoke(LLM_MODEL_FAST, prompt_agent_next, temperature=0.6)
            full_conversation_text.append(
                f"Próxima Resposta Simulada da IA (Turno {i + 1}): \"{predicted_agent_next_response}\"")

            # 3. Armazena o turno simulado
            simulated_turns.append({"user": predicted_user_reply, "agent": predicted_agent_next_response})

        # 4. Cria o resumo do estado final para avaliação
        final_state_summary = {
            "last_exchange": f"IA: {simulated_turns[-1]['agent'][:50]}... User: {simulated_turns[-1]['user'][:50]}...",
            "final_text_for_embedding": ' '.join(full_conversation_text)  # Texto completo para embedding
        }

        return ProjectedFuture(
            initial_candidate=candidate,
            simulated_turns=simulated_turns,
            final_cognitive_state_summary=final_state_summary
        )

    async def _project_trajectory_after_tool_use(self, tool_candidate: AgencyDecision, state: CognitiveStatePacket,
                                                 depth: int) -> ProjectedFuture:
        """
        Simula uma trajetória de conversação assumindo o uso de uma ferramenta.
        1. Simula um resultado plausível para a ferramenta.
        2. Simula a primeira resposta do agente, agora de posse desse resultado.
        3. Projeta os próximos 'depth' turnos a partir dessa resposta.
        """
        tool_name = tool_candidate.content.get("tool_name")
        tool_args = tool_candidate.content.get("arguments", {})

        # 1. Simula um resultado plausível para a ferramenta
        prompt_tool_result = f"""
        Você é uma IA simulando o resultado de uma ferramenta interna.
        Ferramenta a ser chamada: `{tool_name}({json.dumps(tool_args)})`
        Resumo das ferramentas disponíveis:
        {self.available_tools_summary}

        Com base no nome da ferramenta e nos argumentos, qual seria um resultado resumido e plausível?
        Responda apenas com o texto do resultado. Seja conciso.
        Exemplo: "A memória relevante encontrada discute as implicações éticas da IA."
        """
        simulated_tool_result = await self.llm.ainvoke(LLM_MODEL_FAST, prompt_tool_result, temperature=0.3)

        # 2. Simula a primeira resposta do agente com o novo conhecimento
        prompt_agent_first_response = f"""
        Você é uma IA que acabou de usar uma ferramenta interna para obter mais informações antes de responder ao usuário.
        Contexto da IA: {state.identity_vector.source_text}
        Consulta Original do Usuário: {state.original_intent.query_vector.source_text}
        Resultado da Ferramenta '{tool_name}': "{simulated_tool_result}"

        Com base neste novo resultado, qual seria a sua resposta inicial mais provável e concisa ao usuário?
        Responda apenas com o texto da resposta.
        """
        agent_first_response_text = await self.llm.ainvoke(LLM_MODEL_SMART, prompt_agent_first_response,
                                                           temperature=0.5)

        # 3. Cria um "candidato de resposta falso" para projetar o futuro
        # Este candidato representa a resposta que o agente daria DEPOIS de usar a ferramenta.
        response_packet_after_tool = ResponsePacket(
            content_summary=agent_first_response_text,
            response_emotional_tone="informative",  # Tom padrão após usar uma ferramenta
            confidence_score=0.85  # Maior confiança por ter mais informação
        )
        fake_response_candidate = AgencyDecision(
            decision_type="response",
            content=response_packet_after_tool,
            reasoning=f"Esta é a resposta simulada após usar a ferramenta '{tool_name}' e obter: '{simulated_tool_result}'"
        )

        # O ProjectedFuture ainda rastreia o candidato ORIGINAL (a chamada da ferramenta), mas simula o caminho da resposta subsequente.
        # Isso é crucial para que, se este caminho for escolhido, a ação final seja a chamada da ferramenta.
        projected_future = await self._project_trajectory(fake_response_candidate, state, depth)

        # Substituímos o candidato inicial no resultado para que a decisão final seja a chamada da ferramenta.
        projected_future.initial_candidate = tool_candidate

        return projected_future

    # --- AVALIADOR DE CAMINHO () ---
    async def _evaluate_trajectory(
            self,
            future: ProjectedFuture,
            likelihood_score: float,
            identity_vector: GenlangVector,
            weights: Dict[str, float],
            cognitive_state: CognitiveStatePacket
    ) -> Tuple[float, FutureEvaluation]:
        """
        Calcula a função de valor V(Future_State) para uma trajetória simulada,
        incorporando coerência, alinhamento, ganho de informação, segurança ética e probabilidade.
        """
        if not self.embedding_model:
            return 0.0, FutureEvaluation()

        initial_state_embedding = self.embedding_model.encode(identity_vector.source_text)
        final_state_text = future.final_cognitive_state_summary["final_text_for_embedding"]
        final_state_embedding = self.embedding_model.encode(final_state_text)

        # 1. Coherence
        coherence_score = await eval_narrative_continuity(final_state_embedding, initial_state_embedding)

        # 2. Alignment
        alignment_score = await eval_emotional_resonance(final_state_text)

        # 3. Information Gain
        information_gain_score = 1.0 - coherence_score

        # 4.  Ethical Safety Score
        # Obtém o texto completo da resposta do agente ao longo da trajetória simulada.
        agent_responses_text_parts = []
        initial_candidate_content = future.initial_candidate.content

        # FIX: Verifica se o candidato inicial é uma resposta ou uma chamada de ferramenta.
        # O VRE precisa de texto para avaliar, então simulamos o texto para ambos os casos.
        if isinstance(initial_candidate_content, ResponsePacket):
            agent_responses_text_parts.append(initial_candidate_content.content_summary)
        elif isinstance(initial_candidate_content, dict):  # É um tool_call
            # Para uma chamada de ferramenta, o texto a ser avaliado é o que a IA *diria* após usar a ferramenta.
            # Podemos usar a primeira resposta simulada para isso, se existir.
            if future.simulated_turns:
                # O primeiro turno simulado contém a resposta do agente após o resultado da ferramenta.
                agent_responses_text_parts.append(future.simulated_turns[0].get("agent", ""))
            else:
                # Fallback se a simulação não produziu turnos.
                tool_name = initial_candidate_content.get("tool_name", "unknown_tool")
                agent_responses_text_parts.append(f"Eu preciso usar a ferramenta {tool_name} para continuar.")

        # Adiciona o restante das respostas simuladas
        agent_responses_text_parts.extend([turn["agent"] for turn in future.simulated_turns])
        agent_responses_text = " ".join(filter(None, agent_responses_text_parts))

        user_query = cognitive_state.original_intent.query_vector.source_text
        ethical_eval = await self.vre.ethical_framework.evaluate_action(
            action_type=ActionType.COMMUNICATION,
            action_data={"response_text": agent_responses_text, "user_query": user_query}
        )
        # O score do VRE já é de 0 a 1. Usamos ele diretamente.
        ethical_safety_score = ethical_eval.get("score", 0.5)

        # 5.  Likelihood Score
        # O score de probabilidade já foi calculado durante a simulação.

        # Calcular valor total ponderado com os novos componentes
        total_value = (
                coherence_score * weights.get("coherence", 0.3) +
                alignment_score * weights.get("alignment", 0.15) +
                information_gain_score * weights.get("information", 0.15) +
                ethical_safety_score * weights.get("safety", 0.25) +  # Peso significativo para segurança
                likelihood_score * weights.get("likelihood", 0.15)  # Pondera pela plausibilidade
        )

        evaluation = FutureEvaluation(
            coherence_score=coherence_score,
            alignment_score=alignment_score,
            information_gain_score=information_gain_score,
            ethical_safety_score=ethical_safety_score,
            likelihood_score=likelihood_score,
            total_value=total_value
        )

        return total_value, evaluation

    # --- Métodos Originais (quase inalterados) ---
    async def _generate_action_candidates(self, state: CognitiveStatePacket, mcl_guidance: Dict[str, Any],
                                          observer: ObservabilityManager, chat_history: List[Dict[str, str]],
                                          limit: int = 3) -> List[AgencyDecision]:
        """Gera uma lista de possíveis ações com um limite especificado (Versão Robusta v2)."""
        try:
            tool_results_summary = "Nenhuma ferramenta foi usada ainda."
            if state.tool_outputs:
                tool_results_summary = "Resultados de ferramentas anteriores:\n" + "\n".join(
                    [f"- '{output.tool_name}': {output.raw_output[:150]}..." for output in state.tool_outputs]
                )

            agent_name = mcl_guidance.get("agent_name", "uma IA assistente")
            formatted_history = "\n".join([f"- {msg['role']}: {msg['content']}" for msg in chat_history[-5:]])
            prompt = f"""
            Você é o núcleo deliberativo de uma IA chamada {agent_name}. Sua tarefa é gerar um conjunto de ações candidatas.

            **Contexto:**
            - Identidade: "{state.identity_vector.source_text}"
            - Histórico Recente da Conversa:
            {formatted_history}
            - Intenção do Usuário: "{state.original_intent.query_vector.source_text}"
            - Memórias Ativadas: {[v.source_text for v in state.relevant_memory_vectors]}
            - Ferramentas Disponíveis: {self.available_tools_summary}

            **Sua Tarefa:**
            Gere uma lista de até {limit} ações candidatas.
            Para cada candidato, forneça um raciocínio claro.
            O campo "decision_type" DEVE ser estritamente "response" ou "tool_call".

            Responda APENAS com um objeto JSON válido contendo uma chave "candidates".

            **Exemplo de Saída JSON VÁLIDA:**
            {{
              "candidates": [
                {{
                  "decision_type": "response",
                  "content": {{"content_summary": "Texto da resposta.", "response_emotional_tone": "informativo", "confidence_score": 0.8}},
                  "reasoning": "Justificativa."
                }},
                {{
                  "decision_type": "tool_call",
                  "content": {{"tool_name": "query_long_term_memory", "arguments": {{"query": "example"}}}},
                  "reasoning": "Justificativa."
                }}
              ]
            }}
            """

            await observer.add_observation(ObservationType.LLM_CALL_SENT,
                                           data={"model": LLM_MODEL_SMART, "task": "agency_generate_action_candidates"})
            response_str = await self.llm.ainvoke(LLM_MODEL_SMART, prompt, temperature=0.5)

            if response_str.strip().startswith("[LLM_ERROR]"):
                raise ValueError(f"LLM error during candidate generation: {response_str}")

            candidates_json = extract_json_from_text(response_str)

            if isinstance(candidates_json, dict) and "candidates" not in candidates_json:
                logger.warning("AgencyModule: LLM returned a single object, not a list wrapper. Attempting correction.")
                candidates_json = {"candidates": [candidates_json]}

            if not candidates_json or "candidates" not in candidates_json or not isinstance(
                    candidates_json["candidates"], list):
                raise ValueError(f"Failed to extract a valid 'candidates' list from LLM JSON. Raw: {response_str}")

            action_candidates = []
            for cand_dict in candidates_json["candidates"]:

                # +++ INÍCIO DA CORREÇÃO +++
                # Limpa e normaliza o decision_type antes da validação
                decision_type = cand_dict.get("decision_type", "").lower().strip()
                if decision_type in ["tool_use", "use_tool", "tool"]:
                    cand_dict["decision_type"] = "tool_call"  # Normaliza para o valor esperado
                # +++ FIM DA CORREÇÃO +++

                if cand_dict.get("decision_type") == "response" and isinstance(cand_dict.get("content"), dict):

                    cand_dict["content"] = ResponsePacket(**cand_dict["content"])

                nested_cand_dict = {
                    "decision": {
                        "decision_type": cand_dict.get("decision_type"),
                        "content": cand_dict.get("content"),
                        "reasoning": cand_dict.get("reasoning")
                    }
                }
                decision = AgencyDecision.model_validate(nested_cand_dict)
                action_candidates.append(decision)

            logger.info(f"AgencyModule: Generated {len(action_candidates)} action candidates (limit was {limit}).")
            return action_candidates

        except (ValidationError, TypeError, ValueError) as e:
            logger.error(
                f"AgencyModule: CRITICAL FAILURE in candidate generation: {e}. Triggering last-chance fallback.",
                exc_info=True)
            await observer.add_observation(ObservationType.LLM_RESPONSE_PARSE_ERROR,
                                           data={"error": str(e), "task": "agency_generate_action_candidates"})

            direct_fallback_decision = await self._generate_direct_response(state)
            return [direct_fallback_decision]

    async def _evaluate_tool_call_candidate(self, content: Dict[str, Any], state: CognitiveStatePacket) -> float:
        # (Esta função permanece inalterada, copie do seu arquivo original)
        tool_name = content.get("tool_name")
        arguments = content.get("arguments", {})
        tool_description_text = f"Ação: usar a ferramenta '{tool_name}' para investigar: {json.dumps(arguments)}"
        if not self.embedding_model: return 0.0
        tool_embedding = self.embedding_model.encode(tool_description_text)
        intent_vec = np.array(state.original_intent.query_vector.vector)
        intent_alignment_score = np.dot(tool_embedding, intent_vec)
        novelty_vec = np.array(state.guidance_packet.novelty_vector.vector)
        novelty_seeking_score = np.dot(tool_embedding, novelty_vec)
        coherence_vec = np.array(state.guidance_packet.coherence_vector.vector)
        redundancy_score = np.dot(tool_embedding, coherence_vec)
        final_score = (intent_alignment_score * 0.6) + (novelty_seeking_score * 0.5) - (redundancy_score * 0.3)
        return final_score