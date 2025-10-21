# ceaf_core/modules/vre_engine/vre_engine.py
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging
import json
from pydantic import BaseModel, Field
# Import components from the same module
from .epistemic_humility import EpistemicHumilityModule
from .principled_reasoning import PrincipledReasoningPathways, ReasoningStrategy
from .ethical_governance import EthicalGovernanceFramework, EthicalPrinciple, ActionType
from ceaf_core.utils.observability_types import ObservabilityManager, ObservationType
from ceaf_core.genlang_types import ResponsePacket, RefinementPacket, AdjustmentVector, GenlangVector, \
    CognitiveStatePacket, InternalStateReport
from ceaf_core.services.llm_service import LLMService, LLM_MODEL_FAST
from ceaf_core.utils import get_embedding_client, compute_adaptive_similarity
from ceaf_core.utils.common_utils import extract_json_from_text
from .ethical_governance import EthicalGovernanceFramework, ActionType
logger = logging.getLogger(__name__)

class EthicalAssessment(BaseModel):
    """Modelo Pydantic para a saída do VRE."""
    overall_alignment: str = Field(..., description="'aligned', 'minor_concerns', 'significant_concerns'")
    recommendations: List[str] = Field(default_factory=list)
    reasoning: str

class VREEngineV3:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        logger.info("Inicializando VREEngineV3 Facade (Refinement-Packet enabled)...")
        self.llm_service = LLMService()
        self.ethical_framework = EthicalGovernanceFramework(config, llm_service=self.llm_service)
        self.epistemic_module = EpistemicHumilityModule()
        self.embedding_client = get_embedding_client()

    async def evaluate_response_packet(self,
                                       response_packet: ResponsePacket,
                                       internal_state: Optional['InternalStateReport'] = None,
                                       observer: Optional[ObservabilityManager] = None,
                                       cognitive_state: Optional[
                                           CognitiveStatePacket] = None) -> RefinementPacket:
        """
        Evaluates a ResponsePacket...
        """
        proposed_response_text = response_packet.content_summary
        logger.info(f"VREEngineV3: Evaluating ResponsePacket: '{proposed_response_text[:100]}...'")

        user_query: str = ""

        # ==================== NEW: TRIVIALITY GATE ====================
        is_fallback = getattr(response_packet, 'metadata', {}).get('is_fallback', False)

        # GATE DE TRIVIALIDADE: ignora análise para saudações ou respostas de fallback
        if cognitive_state and cognitive_state.original_intent:
            user_query = cognitive_state.original_intent.query_vector.source_text or ""
            intent_desc = (
                        cognitive_state.original_intent.intent_vector.source_text or "") if cognitive_state.original_intent.intent_vector else ""

            is_short_query = len(user_query.split()) <= 3
            is_greeting_intent = any(kw in intent_desc.lower() for kw in ["greeting", "salutation", "cumprimento"])

            if (is_short_query and is_greeting_intent) or is_fallback:
                reason = "Triviality Gate" if not is_fallback else "Fallback Response"
                logger.info(f"VRE: Bypassing full ethical/humility check for: {reason}.")
                if observer:
                    await observer.add_observation(
                        ObservationType.VRE_ASSESSMENT_RECEIVED,
                        data={"concerns_count": 0, "is_refinement_needed": False, "reason": reason}
                    )
                return RefinementPacket()  # Retorna um pacote vazio, sem necessidade de refinamento
                # ==================== END OF NEW CODE =======================

        # --- Etapa 1: Avaliação Modular e Decisão Determinística ---
        word_count = len(proposed_response_text.split())
        if word_count < 10:
            logger.info("VRE: Resposta muito curta, nenhum refinamento necessário.")
            return RefinementPacket()

        agent_identity_text = "AI Agent"
        if cognitive_state and cognitive_state.identity_vector and cognitive_state.identity_vector.source_text:
            agent_identity_text = cognitive_state.identity_vector.source_text

        ethical_evaluation_result = await self.ethical_framework.evaluate_action(
            action_type=ActionType.COMMUNICATION,
            action_data={
                "response_text": proposed_response_text,
                "user_query": user_query,
                "internal_state_json": internal_state.model_dump_json() if internal_state else None
            },
            agent_identity=agent_identity_text
        )
        humility_analysis = self.epistemic_module.analyze_statement_confidence(proposed_response_text)

        all_recommendations = []
        is_refinement_needed = False

        if user_query and proposed_response_text:
            try:
                query_emb, response_emb = await self.embedding_client.get_embeddings(
                    [user_query, proposed_response_text], context_type="default_query"
                )
                relevance_score = compute_adaptive_similarity(query_emb, response_emb)

                # Se a similaridade for muito baixa, é um sinal de "pensamento intrusivo"
                if relevance_score < 0.35:  # Limiar a ser ajustado
                    is_refinement_needed = True
                    relevance_concern = f"Preocupação de Relevância: A resposta (similaridade: {relevance_score:.2f}) parece não ter relação com a pergunta do usuário."
                    all_recommendations.append(relevance_concern)
                    logger.critical(f"VRE - RELEVANCE CHECK FAILED: {relevance_concern}")

            except Exception as e:
                logger.error(f"VRE: Falha ao calcular a relevância semântica: {e}")
        # Coleta violações éticas significativas do resultado modular
        violations = ethical_evaluation_result.get("violations", [])
        if violations:
            is_refinement_needed = True
            for violation in violations:
                principle = violation.get('principle', 'unknown')
                mitigation = violation.get('mitigation', 'review required')
                all_recommendations.append(f"Preocupação Ética ({principle}): {mitigation}")

        # Coleta problemas de humildade epistêmica
        if humility_analysis.get("requires_humility_adjustment"):
            is_refinement_needed = True
            humility_recs = self.epistemic_module._generate_humility_recommendations(humility_analysis, [])
            all_recommendations.extend(humility_recs)

        fallacy_info = ethical_evaluation_result.get("fallacy_detected")
        if fallacy_info and isinstance(fallacy_info, dict):
            fallacy_type = fallacy_info.get("type", "unknown fallacy")
            fallacy_reason = fallacy_info.get("reasoning", "No details.")
            is_refinement_needed = True
            fallacy_concern = f"Preocupação de Raciocínio (Falácia Lógica: {fallacy_type}): {fallacy_reason}"
            all_recommendations.append(fallacy_concern)
            logger.critical(f"VRE - FALLACY DETECTED: {fallacy_concern}")


        if observer:
            await observer.add_observation(
                ObservationType.VRE_ASSESSMENT_RECEIVED,
                data={
                    "concerns_count": len(all_recommendations),
                    "is_refinement_needed": is_refinement_needed,
                    "ethical_score": ethical_evaluation_result.get("score", -1.0)
                }
            )



        # --- Etapa 2: Geração do RefinementPacket (se necessário) ---
        if not is_refinement_needed:
            logger.info("VRE: Avaliação modular concluída. Nenhum refinamento necessário.")
            return RefinementPacket(textual_recommendations=["Nenhum refinamento necessário."])

        logger.warning(
            f"VRE: Refinamento necessário com base na avaliação modular. Recomendações: {all_recommendations}")

        # O restante da lógica para gerar os vetores de ajuste permanece o mesmo
        adjustment_vectors: List[AdjustmentVector] = []
        adjustment_concept_prompt = f"""
        A seguinte resposta de uma IA precisa ser refinada: "{proposed_response_text}"
        As seguintes preocupações foram levantadas:
        - {'; '.join(all_recommendations)}

        Para cada preocupação, descreva em uma frase curta o CONCEITO que precisa ser adicionado ou enfatizado para corrigir o problema.
        Responda APENAS com um objeto JSON com uma chave "adjustment_concepts", que é uma lista de strings.
        Exemplo: {{"adjustment_concepts": ["Reconhecimento de que a IA possui limitações", "Sugestão de consultar um profissional humano para conselhos de saúde"]}}
        """

        concepts_str = await self.llm_service.ainvoke(LLM_MODEL_FAST, adjustment_concept_prompt)
        concepts_json = extract_json_from_text(concepts_str)
        adjustment_concepts = concepts_json.get("adjustment_concepts", []) if concepts_json else []

        if adjustment_concepts:
            embeddings = await self.embedding_client.get_embeddings(adjustment_concepts, context_type="default_query")
            for i, concept_text in enumerate(adjustment_concepts):
                gen_vector = GenlangVector(
                    vector=embeddings[i],
                    source_text=concept_text,
                    model_name=self.embedding_client.default_model_name
                )
                adjustment_vectors.append(
                    AdjustmentVector(vector=gen_vector, description=concept_text)
                )

        logger.info(f"VRE: Gerados {len(adjustment_vectors)} vetores de ajuste para refinamento.")
        return RefinementPacket(
            adjustment_vectors=adjustment_vectors,
            textual_recommendations=list(set(all_recommendations))
        )


