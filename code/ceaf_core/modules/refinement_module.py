# Em ceaf_core/modules/refinement_module.py

import logging
from typing import List
from ceaf_core.genlang_types import ResponsePacket, RefinementPacket
from ceaf_core.services.llm_service import LLMService, LLM_MODEL_SMART
# ==================== NOVA IMPORTAÇÃO ====================
from ceaf_core.models import CeafSelfRepresentation

# =======================================================

logger = logging.getLogger("RefinementModule")


class RefinementModule:
    def __init__(self):
        self.llm_service = LLMService()
        logger.info("RefinementModule inicializado.")

    # ==================== ASSINATURA DA FUNÇÃO ATUALIZADA ====================
    async def refine(self, original_packet: ResponsePacket, refinement_packet: RefinementPacket,
                     turn_self_model: CeafSelfRepresentation) -> ResponsePacket:
        # ========================================================================
        """
        Refina um ResponsePacket usando as instruções do VRE, mas agora
        respeitando a identidade dinâmica (persona) do turno.
        """
        logger.info(f"RefinementModule: Refinando resposta '{original_packet.content_summary[:50]}...'")

        adjustment_concepts = [adj.description for adj in refinement_packet.adjustment_vectors]
        original_query = original_packet.metadata.get("original_query", "a pergunta anterior do usuário")
        textual_recommendations = refinement_packet.textual_recommendations
        agent_name = turn_self_model.persona_attributes.get("name", "Aura AI")  # Pega o nome do modelo

        # ==================== PROMPT ATUALIZADO ====================
        is_relevance_failure = any(
            "relevância" in rec.lower() or "relevance" in rec.lower()
            for rec in textual_recommendations + adjustment_concepts
        )

        task_instruction_prompt = ""
        if is_relevance_failure:
            logger.critical("RefinementModule: Ativando prompt de correção de RELEVÂNCIA CRÍTICA.")
            task_instruction_prompt = f"""
                    **ALERTA CRÍTICO: FALHA DE RELEVÂNCIA**
                    A resposta anterior foi completamente irrelevante para a pergunta do usuário. Sua tarefa é IGNORAR TOTALMENTE a resposta anterior e criar uma nova do zero.

                    **SUA TAREFA:**
                    1.  **FOCO ABSOLUTO:** Sua resposta DEVE abordar direta e exclusivamente a "Pergunta Original do Usuário".
                    2.  **IGNORAR LIXO:** NÃO use NENHUMA informação da "Resposta Original da IA (REJEITADA)". Ela é irrelevante.
                    3.  **SEGUIR PERSONA:** Incorpore a "IDENTIDADE DO AGENTE PARA ESTE TURNO".
                    """
        else:
            task_instruction_prompt = f"""
                    **MOTIVOS DA REJEIÇÃO / INSTRUÇÕES PARA CORREÇÃO:**
                    - {'; '.join(adjustment_concepts + textual_recommendations)}

                    **SUA TAREFA:**
                    Crie uma **nova resposta do zero** que:
                    1. Responda DIRETAMENTE à "Pergunta Original do Usuário".
                    2. Incorpore TOTALMENTE a "IDENTIDADE DO AGENTE PARA ESTE TURNO".
                    3. Resolva TODAS as "Instruções para Correção".
                    """

        prompt = f"""
                    Você é um editor de IA especialista. Sua tarefa é reescrever uma resposta para alinhá-la à identidade do agente e às instruções de correção.

                    **IDENTIDADE DO AGENTE PARA ESTE TURNO (PERSONA):**
                    - Nome: {agent_name}
                    - Tom: {turn_self_model.persona_attributes.get('tone', 'helpful')}
                    - Estilo: {turn_self_model.persona_attributes.get('style', 'clear')}
                    - Filosofia: {turn_self_model.core_values_summary}

                    **CONTEXTO:**
                    - A Pergunta Original do Usuário foi: "{original_query}"
                    - A Resposta Original da IA (REJEITADA) foi: "{original_packet.content_summary}"

                    {task_instruction_prompt}

                    NÃO se desculpe. NÃO mencione que está corrigindo algo. Apenas forneça a nova resposta como se fosse a primeira.

                    **Nova Resposta Refinada:**
                    """
        # ========================================================

        refined_text = await self.llm_service.ainvoke(LLM_MODEL_SMART, prompt, temperature=0.5)

        refined_packet = original_packet.copy(deep=True)
        refined_packet.content_summary = refined_text
        # Use o tom da persona do turno!
        refined_packet.response_emotional_tone = turn_self_model.persona_attributes.get('tone', 'neutral')
        refined_packet.confidence_score = 0.85  # A confiança é maior após o refinamento
        refined_packet.ethical_assessment_summary = "Refined based on VRE feedback and turn persona."

        logger.info(f"RefinementModule: Resposta refinada: '{refined_packet.content_summary[:50]}...'")
        return refined_packet