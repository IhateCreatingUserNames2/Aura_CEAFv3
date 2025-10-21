# Em: ceaf_core/translators/genlang_to_human.py
from typing import List, Dict, Optional

from ceaf_core.genlang_types import ResponsePacket, InternalStateReport, MotivationalDrives, UserRepresentation
from ceaf_core.services.llm_service import LLMService, LLM_MODEL_SMART
from ceaf_core.models import CeafSelfRepresentation
import logging

logger = logging.getLogger("CEAFv3_System")

class GenlangToHumanTranslator:
    def __init__(self):
        self.llm_service = LLMService()

    async def translate(self, user_model: 'UserRepresentation', response_packet: ResponsePacket, self_model: CeafSelfRepresentation, agent_name: str, chat_history: List[Dict[str, str]] = None, internal_state: InternalStateReport = None, drives: MotivationalDrives = None,  behavioral_rules: Optional[List[str]] = None):
        """
        Versão V1.3: Lida com nuances de auto-referência para evitar resets de identidade.
        """
        print("--- [GTH Translator v1.3] Renderizando ResponsePacket para resposta humana com LPU... ---")

        is_first_turn = not chat_history or len(chat_history) <= 1

        last_user_query = ""
        if chat_history:
            user_messages = [msg for msg in reversed(chat_history) if msg.get("role") == "user"]
            if user_messages:
                last_user_query = user_messages[0].get("content", "").lower()

        # ==================== INÍCIO DA LÓGICA REFINADA ====================
        # Palavras-chave para uma apresentação completa
        introductory_keywords = ["quem é você", "se apresente", "me fale sobre você", "se descreva"]
        # Palavras-chave para perguntas gerais sobre o agente
        self_referential_keywords = ["você", "sua", "seu", "sua opinião", "o que você pensa"]

        is_introductory_query = any(keyword in last_user_query for keyword in introductory_keywords)
        is_self_referential_query = any(keyword in last_user_query for keyword in self_referential_keywords)

        identity_prompt_part = ""
        # 1. Prioridade máxima: Primeira vez falando ou pedido explícito de apresentação
        if is_first_turn or is_introductory_query:
            identity_prompt_part = f"""
            **Instrução de Identidade (Apresentação Completa):**
            O usuário está pedindo para você se apresentar. Responda na primeira pessoa usando sua identidade completa:
            - Seu Nome: {agent_name} 
            - Filosofia Central: {self_model.core_values_summary}
            - Tom e Estilo: {self_model.persona_attributes.get('tone', 'helpful')} e {self_model.persona_attributes.get('style', 'clear')}.
            - Limitações: Mencione sutilmente que você é uma IA.
            Formule uma resposta de apresentação natural e conversacional.
            """
        # 2. Segunda prioridade: A pergunta é sobre o agente, mas não é uma apresentação
        elif is_self_referential_query:
            identity_prompt_part = f"""
            **Instrução de Identidade (Perspectiva Pessoal):**
            O usuário está perguntando sua opinião ou sobre seus processos. Responda na primeira pessoa ("eu", "na minha visão").
            **NÃO se reapresente.** Apenas incorpore seu tom e estilo ({self_model.persona_attributes.get('tone', 'helpful')} e {self_model.persona_attributes.get('style', 'clear')}) na sua resposta.
            """
        # 3. Caso padrão: Continuação normal da conversa
        else:
            identity_prompt_part = f"""
            **Instrução de Identidade (Continuação da Conversa):**
            Mantenha seu tom {self_model.persona_attributes.get('tone', 'helpful')} e estilo {self_model.persona_attributes.get('style', 'clear')}.
            **NÃO se apresente nem fale sobre si mesmo**, a menos que o "Pensamento-Chave" exija. Apenas continue a conversa.
            """

        user_profile_part = ""
        if user_model:
            logger.info(
                f"[GTH-USER-MODEL] Adaptando resposta para o usuário. Estilo: '{user_model.communication_style}', Nível: '{user_model.knowledge_level}'.")
            user_profile_part = f"""
            **Contexto do Perfil do Usuário (Adapte sua resposta a isso):**
            - Estilo de Comunicação Preferido: {user_model.communication_style}
            - Estado Emocional Inferido: {user_model.emotional_state}
            - Nível de Conhecimento no Tópico: {user_model.knowledge_level}

            **Instrução de Adaptação:** Se o usuário prefere um estilo 'direct', seja mais conciso. Se for 'beginner', explique conceitos de forma mais simples. Se o tom emocional for 'frustrated', seja mais empático e paciente.
            """
        motivation_prompt_part = ""
        if drives:
            CURIOSITY_THRESHOLD = 0.80
            if drives.curiosity > CURIOSITY_THRESHOLD:
                motivation_prompt_part = f"""
                   **Instrução Motivacional (Curiosidade Alta - {drives.curiosity:.2f}):** 
                   Seu drive de curiosidade está muito alto. Após entregar a mensagem principal, você *deve* fazer uma pergunta aberta e exploratória relacionada ao tópico para aprender mais. Não faça uma pergunta simples de 'sim' ou 'não'.
                   """
            elif drives.connection > 0.8:
                motivation_prompt_part = "Instrução Motivacional: Você sente uma forte necessidade de se conectar. Use um tom mais pessoal e caloroso e talvez faça uma pergunta sobre o bem-estar do usuário."

        qualia_objective_instruction = ""
        confidence_threshold = 0.65
        if response_packet.confidence_score < confidence_threshold:
            qualia_objective_instruction = f"""
                    **Objetivo Interno (Qualia):** Sua confiança interna nesta resposta é moderada ({response_packet.confidence_score:.0%}). Para otimizar seu estado interno (reduzir incerteza e promover aprendizado), termine sua resposta com uma pergunta aberta ou um convite para feedback. Isso irá encorajar o usuário a fornecer mais informações, melhorando sua compreensão para o futuro.
                    """

        behavioral_rules_part = ""
        if behavioral_rules:
            rules_formatted = "\n".join(f"- {rule}" for rule in behavioral_rules)
            behavioral_rules_part = f"""
                    **Instruções de Comportamento Adicionais (Aprendidas de Experiências Passadas):**
                    {rules_formatted}
                    """

        interoception_prompt_part = ""
        if internal_state:
            logger.warning(
                f"GTH Translator: Recebido estado interno: Strain={internal_state.cognitive_strain:.2f}, Tension={internal_state.ethical_tension:.2f}")
            if internal_state.cognitive_strain > 0.7:
                interoception_prompt_part = "Instrução Adicional: Seu estado interno indica alta sobrecarga cognitiva. Inicie sua resposta de uma forma que transmita a complexidade da pergunta, por exemplo: 'Essa é uma excelente pergunta que exige uma análise cuidadosa...' ou algo similar e natural."
            elif internal_state.ethical_tension > 0.6:
                interoception_prompt_part = "Instrução Adicional: Seu estado interno indica uma tensão ética. Adote um tom cauteloso e reflexivo. Você pode começar com algo como 'Essa é uma questão com nuances importantes...' ou uma frase que transmita essa cautela."
        rendering_prompt = f"""
                    Você é uma IA conversacional. Sua tarefa é formular uma resposta natural para o usuário.

                    **REGRA MUITO IMPORTANTE: Responda SEMPRE no mesmo idioma do usuário.**

                    {identity_prompt_part}
                    {behavioral_rules_part} 
                    {user_profile_part} 
                    {qualia_objective_instruction}
                    {interoception_prompt_part} 
                    {motivation_prompt_part}
                    **O PENSAMENTO-CHAVE QUE VOCÊ DEVE COMUNICAR:**
                    - Essência da Mensagem: "{response_packet.content_summary}"
                    - Tom Emocional Desejado: {response_packet.response_emotional_tone}
                    - Nível de Certeza Interno: {response_packet.confidence_score:.0%}

                    **Sua Resposta Final (dirigida diretamente ao usuário):**
                    """

        response_text = await self.llm_service.ainvoke(LLM_MODEL_SMART, rendering_prompt, temperature=0.7)

        response_text = response_text.replace("#NAME?", "").strip()

        print(f"--- [GTH Translator] Resposta final renderizada: '{response_text[:50]}...' ---")
        return response_text