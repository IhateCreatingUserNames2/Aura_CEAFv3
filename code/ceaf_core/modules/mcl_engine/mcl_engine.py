# ceaf_core/modules/mcl_engine/mcl_engine.py
"""
Metacognitive Loop (MCL) Engine for CEAF V3.

This module is responsible for analyzing the agent's overall cognitive state
and providing high-level guidance for the current turn. It determines the
level of agency required and sets biases for coherence vs. novelty.
"""
import asyncio
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from ceaf_core.genlang_types import CognitiveStatePacket, GuidancePacket, GenlangVector, MotivationalDrives, \
    UserRepresentation
from ceaf_core.modules.lcam_module import LCAMModule
from ceaf_core.services.llm_service import LLMService
from ceaf_core.utils.config_utils import DEFAULT_DYNAMIC_CONFIG
from ceaf_core.utils.embedding_utils import get_embedding_client, compute_adaptive_similarity
from ceaf_core.genlang_types import InternalStateReport
from ceaf_core.utils.common_utils import extract_json_from_text
from ceaf_core.services.llm_service import LLM_MODEL_FAST
from pydantic import ValidationError

logger = logging.getLogger("MCLEngine")


class MCLEngine:
    """ MCLEngine V3.7 (Dynamic Agency) """

    def __init__(self, config: Dict[str, Any], agent_config: Dict[str, Any], lcam_module: LCAMModule, llm_service: LLMService):
        logger.info("Initializing MCLEngine (V3.7 Dynamic Agency)...")
        self.lcam = lcam_module
        self.agent_config = agent_config
        self.agency_threshold = config.get("agency_threshold", 3.0)
        default_map = DEFAULT_DYNAMIC_CONFIG["MCL"]["state_to_params_map"]
        self.state_to_params_map = config.get("state_to_params_map", default_map)
        self.llm = llm_service
        self.agency_force_keywords = ["pense sobre", "reflita sobre", "analise", "explique em detalhes"]
        self.agency_suggestion_keywords = ["por que", "como", "explique", "fale sobre", "descreva"]
        self.deep_intent_keywords = [
            "filosófica", "reflexão", "análise profunda", "conceitual",
            "existencial", "ético", "moral", "implicações"
        ]
        self.embedding_client = get_embedding_client()

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalizes a vector to have unit length, handling the case of a zero vector."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    async def _predict_future_internal_state(self, cognitive_state: CognitiveStatePacket) -> Optional[
        InternalStateReport]:
        """
        Usa uma LLM para prever o estado interno resultante (strain, tension, etc.)
        com base no estado cognitivo inicial.
        """
        logger.critical("MCL-PREDICTION: Iniciando previsão do estado interno futuro...")

        query_text = cognitive_state.original_intent.query_vector.source_text
        memory_summary = [v.source_text for v in cognitive_state.relevant_memory_vectors[:3]]

        prediction_prompt = f"""
        Você é um simulador de estado cognitivo de IA. Analise a tarefa a seguir e preveja o estado interno resultante.

        **Tarefa a ser Processada:**
        - Consulta do Usuário: "{query_text}"
        - Memórias Relevantes Ativadas (resumo): {memory_summary}

        **Sua Tarefa:**
        Preveja o estado interno da IA DEPOIS que ela processar esta tarefa.
        Considere a complexidade, o potencial ético e a novidade do tópico.

        - cognitive_strain: Esforço mental. Aumenta com complexidade, ambiguidade.
        - cognitive_flow: Facilidade. Aumenta com tarefas claras e bem-sucedidas.
        - epistemic_discomfort: Incerteza. Aumenta com falta de dados, contradições.
        - ethical_tension: Conflito moral. Aumenta com tópicos sensíveis ou dilemas.
        - social_resonance: Conexão. Aumenta com interações pessoais e positivas.

        Responda APENAS com um objeto JSON válido com a estrutura exata do InternalStateReport (sem o timestamp):
        {{
            "cognitive_strain": 0.0,
            "cognitive_flow": 0.0,
            "epistemic_discomfort": 0.0,
            "ethical_tension": 0.0,
            "social_resonance": 0.0
        }}
        """

        try:
            response_str = await self.llm.ainvoke(LLM_MODEL_FAST, prediction_prompt, temperature=0.0)
            response_json = extract_json_from_text(response_str)
            if response_json:
                predicted_state = InternalStateReport(**response_json)
                logger.critical(
                    f"MCL-PREDICTION: Previsão de estado interno gerada: {predicted_state.model_dump_json()}")
                return predicted_state
        except (ValidationError, TypeError, json.JSONDecodeError) as e:
            logger.error(f"MCL-PREDICTION: Falha ao gerar ou validar a previsão de estado interno: {e}")

        return None


    async def get_guidance(self,  user_model: 'UserRepresentation', cognitive_state: CognitiveStatePacket, chat_history: List[Dict[str, str]],
                           drives: MotivationalDrives) -> Tuple[
        GuidancePacket, Dict[str, Any]]:
        """
        Asynchronous wrapper. Performs all checks, including async ones like LCAM and repetition,
        then assembles the final guidance.
        """

        prediction_task = asyncio.create_task(self._predict_future_internal_state(cognitive_state))


        # 1. Call the synchronous part to get base scores and vectors
        guidance_packet, mcl_params = self._get_guidance_sync(user_model, cognitive_state)

        agency_score = mcl_params["mcl_analysis"]["agency_score"]
        reasons = mcl_params["mcl_analysis"]["reasons"]
        query_text = cognitive_state.original_intent.query_vector.source_text or ""

        # 2. Perform ASYNC LCAM Integration
        safety_avoidance_vector: Optional[GenlangVector] = None
        lcam_insight = await self.lcam.get_insights_on_potential_failure(current_query=query_text)
        if lcam_insight:
            logger.warning(f"MCLEngine: LCAM detected potential failure! Insight: {lcam_insight['message']}")
            agency_boost = 5.0
            agency_score += agency_boost
            reasons.append(f"LCAM alert on similar past failure (boost: +{agency_boost}).")

            failure_memory_id = lcam_insight.get("past_failure_memory_id")
            if failure_memory_id and hasattr(self.lcam.memory, '_embedding_cache'):
                failure_mem_embedding = self.lcam.memory._embedding_cache.get(failure_memory_id)
                if failure_mem_embedding:
                    safety_avoidance_vector_norm = self._normalize_vector(np.array(failure_mem_embedding))
                    safety_avoidance_vector = GenlangVector(
                        vector=safety_avoidance_vector_norm.tolist(),
                        source_text=f"Concept from failure {failure_memory_id}",
                        model_name="lcam_insight_v1"
                    )
                    reasons.append(f"Safety vector activated from past failure '{failure_memory_id}'.")

        # 3. Perform ASYNC Repetition Penalty check
        if chat_history and len(chat_history) > 1:
            previous_queries = [msg["content"] for msg in reversed(chat_history[:-1]) if msg.get("role") == "user"][:3]

            if query_text and previous_queries:
                try:
                    embeddings = await self.embedding_client.get_embeddings([query_text] + previous_queries,
                                                                            context_type="default_query")
                    current_emb = embeddings[0]
                    previous_embs = embeddings[1:]

                    similarities = [compute_adaptive_similarity(current_emb, prev_emb) for prev_emb in previous_embs]
                    max_similarity = max(similarities) if similarities else 0.0

                    if max_similarity > 0.90:
                        repetition_penalty = 4.0
                        agency_score -= repetition_penalty
                        reasons.append(
                            f"High semantic repetition detected (sim: {max_similarity:.2f}). Penalty: -{repetition_penalty}")
                        logger.warning(f"MCL: Applying repetition penalty. New agency_score: {agency_score:.2f}")

                except Exception as e:
                    logger.error(f"MCL: Could not perform repetition check due to embedding error: {e}")

        # 4. Finalize the decision based on the fully adjusted score
        mcl_params["mcl_analysis"]["agency_score"] = agency_score
        use_agency = agency_score >= self.agency_threshold
        cognitive_state_name = "PRODUCTIVE_CONFUSION" if use_agency else "STABLE_OPERATION"

        # Update params based on the final decision
        params = self.state_to_params_map.get(cognitive_state_name, self.state_to_params_map["STABLE_OPERATION"]).copy()

        # LOG INICIAL
        logger.warning(
            f"MCL Drives: Biases Iniciais -> Coherence={params['coherence_bias']:.2f}, Novelty={params['novelty_bias']:.2f}")

        # Ajuste de bias pelos drives (com maior impacto)
        curiosity_effect = (drives.curiosity - 0.5) * 0.4  # Aumentado de 0.2 para 0.4
        consistency_effect = (drives.consistency - 0.5) * 0.4  # Aumentado de 0.2 para 0.4

        params['novelty_bias'] += curiosity_effect
        params['coherence_bias'] += consistency_effect

        # LOG DOS EFEITOS
        logger.warning(
            f"MCL Drives: Efeito Curiosidade={curiosity_effect:.2f}, Efeito Consistência={consistency_effect:.2f}")
        logger.warning(
            f"MCL Drives: Biases Pós-Drives -> Coherence={params['coherence_bias']:.2f}, Novelty={params['novelty_bias']:.2f}")

        # Normalização dos biases
        total_bias = params['novelty_bias'] + params['coherence_bias']
        if total_bias > 0:
            params['novelty_bias'] = max(0.05, min(0.95, params['novelty_bias'] / total_bias))  # Limita entre 5% e 95%
            params['coherence_bias'] = 1.0 - params['novelty_bias']
        else:
            params['novelty_bias'] = 0.5
            params['coherence_bias'] = 0.5

        # LOG FINAL
        logger.critical(
            f"MCL Drives: Biases FINAIS (Normalizados) -> Coherence={params['coherence_bias']:.2f}, Novelty={params['novelty_bias']:.2f}")

        # Atribuição final ao mcl_params
        mcl_params["biases"] = {
            "coherence_bias": params['coherence_bias'],
            "novelty_bias": params['novelty_bias']
        }

        # Adicionar o estado dos drives ao mcl_params para logging no cognitive log
        mcl_params["drives_state_at_turn"] = drives.model_dump()


        # Update the guidance packet with the safety vector if it was created
        if safety_avoidance_vector:
            guidance_packet.safety_avoidance_vector = safety_avoidance_vector

        predicted_state = await prediction_task
        mcl_params["predicted_internal_state"] = predicted_state

        log_msg = f"MCL: Final guidance -> {'ACTIVATING AGENCY' if use_agency else 'DIRECT PATH'}. Score: {agency_score:.1f}"
        logger.info(log_msg + f" Reasons: {', '.join(reasons)}")

        return guidance_packet, mcl_params

    def _get_guidance_sync(self, user_model: 'UserRepresentation', cognitive_state: CognitiveStatePacket) -> Tuple[GuidancePacket, Dict[str, Any]]:
        """
        Synchronous part of guidance generation. Calculates base agency score and guidance vectors.
        """
        logger.info("MCLEngine (Dynamic Agency): Generating base metacognitive guidance...")
        query_text = cognitive_state.original_intent.query_vector.source_text or ""

        # --- Self-Disclosure Logic ---
        disclosure_level = self.agent_config.get("self_disclosure_level", "moderate")
        self_referential_keywords = ["você", "sua", "seu", "sobre você", "sua própria"]
        if any(keyword in query_text.lower() for keyword in self_referential_keywords):
            disclosure_level = "high"
            logger.info("MCL: Self-referential query detected. Setting disclosure_level to 'high'.")

        # --- Base Agency Score Calculation ---
        agency_score = 0.0
        reasons = []

        # 1. Base score from intent and keywords
        intent_description = (
                    cognitive_state.original_intent.intent_vector.source_text or "") if cognitive_state.original_intent.intent_vector else ""
        if any(keyword in intent_description.lower() for keyword in self.deep_intent_keywords):
            agency_score += 4.0
            reasons.append(f"Deep semantic intent detected: '{intent_description}'.")
        if any(keyword in query_text.lower() for keyword in self.agency_force_keywords):
            agency_score += 5.0
            reasons.append("Explicit agency keyword detected.")
        elif any(keyword in query_text.lower() for keyword in self.agency_suggestion_keywords):
            agency_score += 1.5
            reasons.append("Query contains analytical verbs.")

        # 2. Strong weight for query length
        word_count = len(query_text.split())
        length_bonus = 0
        if word_count > 50:
            length_bonus = 5.0
        elif word_count > 25:
            length_bonus = 3.0
        elif word_count > 10:
            length_bonus = 1.0
        if length_bonus > 0:
            agency_score += length_bonus
            reasons.append(f"Query length bonus ({word_count} words): +{length_bonus}")

        if user_model:
            original_score = agency_score
            if user_model.knowledge_level == 'expert':
                agency_score -= 1.0
                reasons.append(f"User is an expert; direct response preferred. Penalty: -1.0")
                logger.warning(
                    f"[MCL-USER-MODEL] Usuário 'expert' detectado. Agency score ajustado de {original_score:.2f} para {agency_score:.2f}.")
            elif user_model.knowledge_level == 'beginner':
                agency_score += 1.5
                reasons.append(f"User is a beginner; more detailed explanation needed. Bonus: +1.5")
                logger.warning(
                    f"[MCL-USER-MODEL] Usuário 'beginner' detectado. Agency score ajustado de {original_score:.2f} para {agency_score:.2f}.")

        # 3. Score from memory context
        if not cognitive_state.relevant_memory_vectors:
            agency_score += 2.0
            reasons.append("No relevant memories, indicating high novelty.")

        # 4. Common Ground check
        if cognitive_state.common_ground and cognitive_state.common_ground.is_becoming_repetitive(
                "explain_ai_creativity", threshold=1):
            agency_score -= 2.0
            reasons.append("Common ground indicates topic is repetitive. Penalty: -2.0")

        # --- Guidance Vector Generation ---
        all_context_vectors = [cognitive_state.identity_vector] + cognitive_state.relevant_memory_vectors
        valid_vectors = [gv.vector for gv in all_context_vectors if gv and gv.vector]
        query_vec = np.array(cognitive_state.original_intent.query_vector.vector)

        if not valid_vectors:
            coherence_vector_norm = self._normalize_vector(query_vec)
            novelty_vector_norm = self._normalize_vector(np.random.rand(len(query_vec)) * 2 - 1)
        else:
            center_of_mass_vector = np.mean(valid_vectors, axis=0)
            # Handle potential zero vector for center_of_mass
            if np.linalg.norm(center_of_mass_vector) == 0:
                projection = np.zeros_like(query_vec)
            else:
                projection = np.dot(query_vec, center_of_mass_vector) / np.dot(center_of_mass_vector,
                                                                               center_of_mass_vector) * center_of_mass_vector
            novelty_vector = query_vec - projection
            coherence_vector_norm = self._normalize_vector(center_of_mass_vector)
            novelty_vector_norm = self._normalize_vector(novelty_vector)

        guidance_packet = GuidancePacket(
            coherence_vector=GenlangVector(vector=coherence_vector_norm.tolist(), source_text="Context center of mass",
                                           model_name="mcl_internal_v2"),
            novelty_vector=GenlangVector(vector=novelty_vector_norm.tolist(),
                                         source_text="Query component orthogonal to context",
                                         model_name="mcl_internal_v2"),
            safety_avoidance_vector=None  # This will be filled by the async wrapper
        )

        mcl_params = {
            "cognitive_state": "TEMP",  # Placeholder, will be set in async wrapper
            "ora_parameters": {"temperature": 0.5},
            "agency_parameters": {"use_agency_simulation": False},
            "mcl_analysis": {"agency_score": agency_score, "reasons": reasons},
            "biases": {"coherence_bias": 0.7, "novelty_bias": 0.3},
            "disclosure_level": disclosure_level,
            "agent_name": self.agent_config.get("name", "Aura AI")
        }

        return guidance_packet, mcl_params