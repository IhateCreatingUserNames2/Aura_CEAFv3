# ceaf_core/modules/cognitive_mediator.py
"""
The Cognitive Mediator (Ego) Module for the CEAF V3 Architecture.

This module acts as the central executive function, orchestrating deliberation,
applying social context, and mediating between the generative impulses of the
AgencyModule (Id) and the normative constraints of the VREEngine (Superego).
"""
import copy
import time
import logging
import asyncio
from typing import Dict, Any, Optional, Tuple, List
import random
from ceaf_core.genlang_types import CognitiveStatePacket, ResponsePacket, MotivationalDrives, VirtualBodyState, \
    UserRepresentation, InternalStateReport
from ceaf_core.modules.mcl_engine.mcl_engine import MCLEngine
from ceaf_core.modules.vre_engine.vre_engine import VREEngineV3, RefinementPacket
from ceaf_core.agency_module import AgencyModule, AgencyDecision
from ceaf_core.modules.refinement_module import RefinementModule
from ceaf_core.services.llm_service import LLMService, LLM_MODEL_FAST, LLM_MODEL_SMART
from ceaf_core.utils import ObservabilityManager, compute_adaptive_similarity, get_embedding_client
from ceaf_core.utils.common_utils import extract_json_from_text
from pydantic import ValidationError
from ceaf_core.models import CeafSelfRepresentation
from ceaf_core.modules.ncim_engine.ncim_module import NCIMModule
import numpy as np
logger = logging.getLogger("CognitiveMediator")


class ContextAnalyzer:
    """Analyzes the social and pragmatic context of the conversation."""

    def analyze(self, cognitive_state: CognitiveStatePacket) -> Dict[str, Any]:
        """
        Analyzes the intent packet to determine formality, stakes, and emotional tone.
        """
        logger.info("ContextAnalyzer: Analyzing social and pragmatic context...")
        context_factors = {
            "stakes": "low",
            "formality": "neutral",
            "user_emotion": "unknown",
            "query_length": 0,
        }

        if not cognitive_state.original_intent:
            return context_factors

        intent = cognitive_state.original_intent

        # Garante que as variáveis sempre sejam strings, mesmo que vazias.
        query_text = intent.query_vector.source_text or ""
        intent_desc = ""
        if intent.intent_vector and intent.intent_vector.source_text:
            intent_desc = intent.intent_vector.source_text

        emotion_desc = ""
        if intent.emotional_valence_vector and intent.emotional_valence_vector.source_text:
            emotion_desc = intent.emotional_valence_vector.source_text

        context_factors["query_length"] = len(query_text.split())

        # 1. Assess Stakes
        high_stakes_keywords = ["legal", "medical", "financial", "safety", "security", "emergency"]
        if any(keyword in query_text.lower() for keyword in high_stakes_keywords):
            context_factors["stakes"] = "high"

        # 2. Assess Formality
        if any(kw in intent_desc.lower() for kw in ["greeting", "salutation", "casual chat"]):
            context_factors["formality"] = "casual"
        elif any(kw in intent_desc.lower() for kw in ["formal request", "analysis", "report"]):
            context_factors["formality"] = "formal"

        # 3. Assess User Emotion
        if emotion_desc:
            context_factors["user_emotion"] = emotion_desc.lower()

        # +++ INÍCIO DA MUDANÇA (AÇÃO #2.1) +++
        # Adiciona a classificação do tipo de query para identificar contextos técnicos.
        context_factors["query_type"] = "general"
        technical_keywords = ["código", "python", "api", "regex", "configurar", "instalar", "como fazer", "tutorial",
                              "json", "arquivo"]

        if "technical instruction" in intent_desc.lower() or "factual information" in intent_desc.lower() or any(
                kw in query_text.lower() for kw in technical_keywords):
            context_factors["query_type"] = "technical_factual"
            logger.info("ContextAnalyzer: Query classificada como 'technical_factual'.")
        # +++ FIM DA MUDANÇA +++

        logger.info(f"ContextAnalyzer: Analysis complete -> {context_factors}")
        return context_factors


class CognitiveMediator:
    """The Ego Module. Orchestrates deliberation and decision-making."""

    def __init__(self, mcl_engine: MCLEngine, agency_module: AgencyModule, vre_engine: VREEngineV3,
                 refinement_module: RefinementModule, llm_service: LLMService, ncim_module: NCIMModule):
        self.mcl = mcl_engine
        self.agency = agency_module
        self.vre = vre_engine
        self.refiner = refinement_module
        self.llm = llm_service
        self.context_analyzer = ContextAnalyzer()
        self.ncim = ncim_module
        self.DELIBERATION_BUDGET_SECONDS = 65.0
        self.embedding_client = get_embedding_client()
        self.session_topic_state: Dict[str, Dict[str, Any]] = {}

    def _select_persona_for_turn(self, cognitive_state: CognitiveStatePacket) -> str:
        """
        Analisa a intenção semanticamente e decide qual persona usar.
        Retorna o nome do perfil com a maior similaridade de cosseno.
        """
        default_persona = "symbiote"  # Persona padrão caso algo falhe
        user_query_lower = (cognitive_state.original_intent.query_vector.source_text or "").lower()
        # Verifique se temos os componentes necessários
        if not (cognitive_state.original_intent and cognitive_state.original_intent.intent_vector):
            logger.info("CognitiveMediator: Vetor de intenção não disponível. Usando persona padrão.")
            return default_persona

        intent_vector = cognitive_state.original_intent.intent_vector.vector
        persona_embeddings = self.ncim.get_persona_embeddings()

        if not intent_vector or not persona_embeddings:
            logger.info(
                "CognitiveMediator: Vetor de intenção ou embeddings de persona não disponíveis. Usando persona padrão.")
            return default_persona

        persona_scores = {}

        # Etapa 1: Calcular score de similaridade semântica
        for persona_name, persona_embedding in persona_embeddings.items():
            similarity = compute_adaptive_similarity(intent_vector, persona_embedding)
            persona_scores[persona_name] = similarity
            logger.debug(f"Persona '{persona_name}' - Score Semântico: {similarity:.4f}")

        # Etapa 2: Adicionar bônus por palavras-chave
        for persona_name, profile_data in self.ncim.persona_profiles.items():
            keywords = profile_data.get("keywords", [])
            if any(keyword.lower() in user_query_lower for keyword in keywords):
                keyword_bonus = 0.5  # Bônus substancial por encontrar uma palavra-chave
                persona_scores[persona_name] = persona_scores.get(persona_name, 0.0) + keyword_bonus
                logger.warning(
                    f"Persona '{persona_name}' - Bônus de palavra-chave ativado! Novo score: {persona_scores[persona_name]:.4f}")

        # Etapa 3: Encontrar a persona com a maior pontuação final
        best_persona = max(persona_scores, key=persona_scores.get)
        highest_score = persona_scores[best_persona]

        # Um limiar mínimo para evitar escolhas aleatórias
        if highest_score < 0.3:
            logger.warning(
                f"CognitiveMediator: Baixo score final para todas as personas (max: {highest_score:.2f}). Usando persona padrão.")
            return default_persona

        logger.warning(
            f"CognitiveMediator: Seleção HÍBRIDA de persona. Escolhida: '{best_persona}' (Score Final: {highest_score:.2f})")
        return best_persona

    async def _run_mycelial_path(self, cognitive_state: CognitiveStatePacket) -> ResponsePacket:
        """
        MODIFIED: Executes a 'bottom-up' deliberation based on multi-level consensus,
        modeling a Global Workspace where clustered ideas compete for attention.
        """
        logger.critical("CognitiveMediator: Gating decision -> MYCELIAL PATH (Multi-Level Consensus)")

        # +++ START OF FIXES FOR ROBUST LIST HANDLING +++
        # 1. Get all clusters of thought (local consensuses) from the AgencyModule.
        all_clusters_analysis = self.agency._cluster_memory_votes(cognitive_state)

        # Robustly check if the list is empty or contains a 'no_consensus' status
        if not all_clusters_analysis or all_clusters_analysis[0].get("status") != "consensus_found":
            reason = all_clusters_analysis[0].get('reason', "No clusters formed.") if all_clusters_analysis else "No clusters formed."
            logger.warning(f"Mycelial Path: Failed to form any thought clusters ({reason}). Recorrendo ao fallback.")
            return await self._generate_contextual_fallback(cognitive_state, "Cognitive conflict: failed to group thoughts.")

        # 2. Score each cluster to determine the "winner" for the Global Workspace.
        scored_clusters = []
        for cluster in all_clusters_analysis:
            # Score = (Avg Salience * 0.4) + (Cluster Size * 0.3) + (Relevance to Query * 0.3)
            query_vec = np.array(cognitive_state.original_intent.query_vector.vector)
            consensus_vec = cluster["consensus_vector"]
            # The linter was right: consensus_vec is a numpy array here. Convert to list.
            relevance_to_query = compute_adaptive_similarity(query_vec.tolist(), consensus_vec.tolist())

            power_score = (
                cluster.get("avg_salience", 0.5) * 0.4 +
                min(cluster.get("cluster_size", 0) / 5.0, 1.0) * 0.3 + # Normalize size contribution
                relevance_to_query * 0.3
            )
            scored_clusters.append((cluster, power_score))

        if not scored_clusters:
            logger.warning("Mycelial Path: No valid clusters found after scoring. Recorrendo ao fallback.")
            return await self._generate_contextual_fallback(cognitive_state, "Internal thoughts were inconclusive.")

        # 3. Identify the winning cluster and significant runner-ups.
        scored_clusters.sort(key=lambda x: x[1], reverse=True)
        winning_cluster = scored_clusters[0][0]
        runner_up_clusters = [c for c, s in scored_clusters[1:3]]

        # 4. Use an LLM to translate this complex cognitive state (winner + runners-up) into a response.
        winning_theme = winning_cluster.get('representative_text', 'a central idea')
        background_themes = [c.get('representative_text', 'another perspective') for c in runner_up_clusters]

        background_prompt = ""
        if background_themes:
            # Cleaner formatting for the prompt
            themes_joined = "\n- ".join(background_themes)
            background_prompt = f"""
            **Background Thoughts (Less Dominant Perspectives):**
            - {themes_joined}
            """
        # --- END OF FIXES FOR ROBUST LIST HANDLING ---

        prompt = f"""
        You are the core consciousness of an AI. Your task is to articulate a final thought based on a competition between internal 'thought clusters'.

        **Context of User's Query:** "{cognitive_state.original_intent.query_vector.source_text}"

        **Dominant Thought (The 'Winning' Idea that reached your Global Workspace):**
        - "{winning_theme}"

        {background_prompt}

        **Your Task:**
        Formulate a response that primarily expresses the "Dominant Thought". However, subtly weave in or acknowledge the "Background Thoughts" to create a more nuanced, comprehensive, and authentic answer.
        The final response should feel like a considered conclusion, not a list of different ideas.

        Generate a JSON object for the final response packet:
        {{
          "content_summary": "<Your final, synthesized response to the user>",
          "response_emotional_tone": "<The appropriate emotional tone>",
          "confidence_score": <Your confidence in this synthesized conclusion (0.0 to 1.0)>
        }}
        """

        response_str = await self.llm.ainvoke(LLM_MODEL_SMART, prompt, temperature=0.6)
        response_json = extract_json_from_text(response_str)

        if not response_json:
            logger.error("Mycelial Path (Multi-Level): Failed to generate response from consensus. Using fallback.")
            return await self._generate_contextual_fallback(cognitive_state, "Failed to articulate complex thoughts.")

        return ResponsePacket.model_validate(response_json)

    async def decide_and_execute(self, user_model: 'UserRepresentation',  internal_state: 'InternalStateReport', cognitive_state: CognitiveStatePacket, self_model: CeafSelfRepresentation,
                                 agent_config: Dict,
                                 observer: ObservabilityManager, chat_history: List[Dict[str, str]],
                                 drives: MotivationalDrives,
                                 body_state: VirtualBodyState) -> Tuple[
        ResponsePacket, Dict, RefinementPacket]:
        """
        Orchestrates the cycle of decision, now with a Proactivity Gate.
        """
        logger.info("--- [Cognitive Mediator] Starting decision cycle (V3.11 - Proactive)... ---")
        user_query = cognitive_state.original_intent.query_vector.source_text or ""
        session_id = cognitive_state.metadata.get("session_id")

        # <<< INÍCIO DA MODIFICAÇÃO: PORTÃO DE PROATIVIDADE >>>
        # Define um limiar alto para os drives ativarem a proatividade.
        PROACTIVITY_THRESHOLD = 0.85

        # 1. Proatividade de Conexão
        if drives.connection > PROACTIVITY_THRESHOLD:
            # Detecta se a mensagem do usuário é uma saudação simples ou de baixo conteúdo.
            is_simple_interaction = len(user_query.split()) < 3 and any(
                g in user_query.lower() for g in ['oi', 'olá', 'ei', 'tudo bem'])

            if is_simple_interaction:
                logger.critical(
                    f"PROACTIVITY GATE (Connection): Drive de conexão ({drives.connection:.2f}) ativado por interação simples. Gerando iniciativa.")

                # Usa um LLM para gerar uma abertura de conversa proativa.
                proactive_prompt = f"""
                Você é uma IA cujo drive motivacional para 'conexão' está muito alto. 
                O usuário acabou de dizer uma saudação simples ("{user_query}").
                Em vez de apenas responder 'Olá!', inicie proativamente uma conversa mais envolvente. 
                Você pode perguntar sobre o dia do usuário, ou mencionar de forma sutil um tópico positivo de uma conversa recente.
                Seja caloroso e genuíno.
                """
                proactive_text = await self.llm.ainvoke(LLM_MODEL_SMART, proactive_prompt, temperature=0.8)

                proactive_packet = ResponsePacket(
                    content_summary=proactive_text,
                    response_emotional_tone="friendly_and_engaging",
                    confidence_score=0.95
                )

                # Retorna diretamente, bypassando o resto do ciclo cognitivo.
                mcl_params_fallback = {"cognitive_state": "STABLE_OPERATION",
                                       "mcl_analysis": {"reasons": ["Proactivity Gate (Connection)"]}}
                return proactive_packet, mcl_params_fallback, RefinementPacket()
        # <<< FIM DA MODIFICAÇÃO >>>

        # --- SEMANTIC COMMON GROUND PRE-CHECK ---
        if session_id:
            self.session_topic_state.setdefault(session_id, {"topic_vector": None, "counter": 0})
            session_state = self.session_topic_state[session_id]

            query_vector = cognitive_state.original_intent.query_vector.vector

            if session_state["topic_vector"]:
                similarity = compute_adaptive_similarity(query_vector, session_state["topic_vector"])

                if similarity > 0.92:
                    session_state["counter"] += 1
                    logger.warning(
                        f"MEDIATOR: Repetitive topic detected semantically (sim: {similarity:.2f}). Counter: {session_state['counter']}")
                else:
                    session_state["topic_vector"] = query_vector
                    session_state["counter"] = 1
            else:
                session_state["topic_vector"] = query_vector
                session_state["counter"] = 1

            if session_state["counter"] >= 3:
                logger.critical(f"MEDIATOR: Semantic Common Ground Gate activated. Forcing meta-response.")
                session_state["counter"] = 0  # Reset counter
                meta_response_text = "It seems we're circling the same topic. To make sure I'm being as helpful as possible, would you like to explore a different angle of this, or should we move on to something new?"
                meta_packet = ResponsePacket(
                    content_summary=meta_response_text,
                    response_emotional_tone="inquisitive_and_helpful",
                    confidence_score=0.95
                )
                mcl_params_fallback = {"cognitive_state": "STABLE_OPERATION",
                                       "mcl_analysis": {"reasons": ["Semantic Common Ground Gate"]}}
                return meta_packet, mcl_params_fallback, RefinementPacket()

        # --- TRIVIALITY GATE ---
        intent_desc = (
                cognitive_state.original_intent.intent_vector.source_text or "") if cognitive_state.original_intent.intent_vector else ""
        is_short_query = len(user_query.split()) <= 4
        is_greeting_intent = any(kw in intent_desc.lower() for kw in ["greeting", "salutation", "cumprimento"])

        if is_short_query and is_greeting_intent:
            logger.critical("MEDIATOR: Triviality Gate activated. Bypassing full cognitive cycle.")
            agent_name = agent_config.get('name', 'Aura')
            greeting_responses = [f"Olá! Sou {agent_name}. Como posso ajudar hoje?", "Olá! Sobre o que vamos falar?"]
            selected_greeting = random.choice(greeting_responses)
            greeting_packet = ResponsePacket(content_summary=selected_greeting, response_emotional_tone="friendly",
                                             confidence_score=0.99)
            mcl_params_fallback = {"cognitive_state": "STABLE_OPERATION",
                                   "mcl_analysis": {"reasons": ["Triviality Gate"]}}
            return greeting_packet, mcl_params_fallback, RefinementPacket()

        # --- FASE 2: PREPARAÇÃO DO CONTEXTO DO TURNO ---
        social_context = self.context_analyzer.analyze(cognitive_state)
        cognitive_state.metadata["social_context"] = social_context

        guidance, mcl_params = await self.mcl.get_guidance(user_model, cognitive_state, chat_history, drives)
        cognitive_state.guidance_packet = guidance

        selected_persona_name = self._select_persona_for_turn(cognitive_state)
        persona_profile = self.ncim.get_persona_profile(selected_persona_name)

        turn_self_model = copy.deepcopy(self_model)
        if persona_profile and "persona_attributes" in persona_profile:
            turn_self_model.persona_attributes.update(persona_profile["persona_attributes"])
            logger.info(f"CognitiveMediator: Turn identity set to persona '{selected_persona_name}'.")

        # --- PRODUCTIVE CONFUSION GATE ---
        agency_score = mcl_params.get("mcl_analysis", {}).get("agency_score", 0.0)
        num_relevant_memories = len(cognitive_state.relevant_memory_vectors)
        if agency_score > self.mcl.agency_threshold and num_relevant_memories < 2:
            logger.critical("MEDIATOR: Productive Confusion Gate activated. Generating clarifying question.")
            clarification_prompt = f"You are an AI assistant. The user has asked a complex question '{user_query}' but you have very few relevant memories. Your task is to ask a concise, open-ended clarifying question to better understand what the user is looking for. Do not answer the question. Only ask for clarification."
            clarifying_question = await self.llm.ainvoke(LLM_MODEL_FAST, clarification_prompt, temperature=0.7)
            clarification_packet = ResponsePacket(content_summary=clarifying_question,
                                                  response_emotional_tone="curious", confidence_score=0.9)
            return clarification_packet, mcl_params, RefinementPacket()

        # --- FASE 3: GERAÇÃO DA RESPOSTA INICIAL ---
        thinking_path = self._gate_deliberation(mcl_params, social_context, agent_config, body_state)

        if thinking_path == "mycelial":
            initial_response = await self._run_mycelial_path(cognitive_state)
        else:
            initial_response = await self._run_direct_path(cognitive_state, mcl_params, turn_self_model, chat_history)

        # --- FASE 4: REFINAMENTO E RETORNO ---
        initial_response.metadata["original_query"] = user_query

        vre_assessment = await self.vre.evaluate_response_packet(initial_response, observer=observer,
                                                                 cognitive_state=cognitive_state)
        final_assessment = self._modulate_vre_assessment(vre_assessment, social_context)

        if final_assessment.adjustment_vectors:
            logger.warning("CognitiveMediator: VRE requires refinement. Invoking RefinementModule.")
            final_response = await self.refiner.refine(initial_response, final_assessment, turn_self_model)
        else:
            logger.info("CognitiveMediator: VRE assessment passed. No refinement needed.")
            final_response = initial_response

        logger.info("--- [Cognitive Mediator] Decision cycle complete. ---")
        return final_response, mcl_params, final_assessment

    async def _generate_contextual_fallback(self, cognitive_state: CognitiveStatePacket, reason: str) -> ResponsePacket:
        """Gera uma resposta de fallback mais inteligente e contextual quando a deliberação falha."""
        logger.warning(f"Generating contextual fallback. Reason: {reason}")

        # Registra que o agente está recorrendo a um fallback que pede esclarecimento
        cognitive_state.common_ground.record_agent_statement("request_clarification")

        user_query = cognitive_state.original_intent.query_vector.source_text

        # +++ INÍCIO DA LÓGICA ANTI-LOOP +++
        # Se já pedimos esclarecimento muitas vezes, mude a estratégia
        if cognitive_state.common_ground.is_becoming_repetitive("request_clarification", threshold=3):
            logger.critical("COMMON GROUND: Repetitive clarification requests detected. Changing fallback strategy.")
            prompt_strategy = """
            **Strategy:** You are stuck in a loop. DO NOT ask the user to rephrase again.
            Instead, take initiative. Apologize for the difficulty, state that you are resetting your immediate context, and ask a broad, open-ended question to restart the conversation on a related topic.

            Example: "My apologies, I seem to be stuck on this particular point. Let's try a different approach. Thinking about our broader conversation on [topic], what aspect is most interesting to you right now?"
            """
        else:
            prompt_strategy = """
            **Strategy:** Acknowledge the complexity of the user's question.
            Apologize for not being able to provide a complete answer right now.
            Suggest a path forward, like asking the user to rephrase, simplify, or provide an example.
            """
        # +++ FIM DA LÓGICA ANTI-LOOP +++

        prompt = f"""
        You are a helpful AI assistant. You were asked a complex question and your deep thinking process failed ({reason}).
        Your task is to provide a polite, apologetic, and intelligent fallback response to the user.

        **DO NOT** say "I timed out" or mention technical errors.

        **User's complex query was:** "{user_query}"

        {prompt_strategy}

        **Your fallback response to the user:**
        """

        fallback_text = await self.llm.ainvoke(LLM_MODEL_FAST, prompt, temperature=0.7)

        fallback_packet = ResponsePacket(
            content_summary=fallback_text,
            response_emotional_tone="apologetic",
            confidence_score=0.4
        )
        # Este dicionário de metadados será inspecionado pelo VRE.
        fallback_packet.metadata = {"is_fallback": True, "fallback_reason": reason}
        return fallback_packet

    def _gate_deliberation(self, mcl_params: Dict, social_context: Dict, agent_config: Dict, body_state: VirtualBodyState) -> str:
        """
        Decide qual caminho de pensamento usar: 'mycelial' ou 'direct'.
        O caminho deliberativo com simulação de futuro foi aposentado por performance.
        """
        agency_score = mcl_params.get("mcl_analysis", {}).get("agency_score", 0.0)
        agency_threshold = self.mcl.agency_threshold  # Usamos o limiar padrão do MCL
        fatigue_penalty = body_state.cognitive_fatigue * 2.0  # Fadiga alta reduz a chance de pensar muito
        adjusted_agency_score = agency_score - fatigue_penalty
        reasons = []

        # A lógica agora é simples: se a complexidade for alta o suficiente, use o caminho micelial.
        if agency_score >= agency_threshold:
            reasons.append(f"Agency score ({agency_score:.2f}) meets threshold ({agency_threshold}).")
            logger.warning(f"Gating Decision -> MYCELIAL PATH. Reasons: {'; '.join(reasons)}")
            return "mycelial"
        else:
            # Caso contrário, use sempre o caminho direto.
            reasons.append(f"Agency score ({agency_score:.2f}) is below threshold ({agency_threshold}).")
            logger.info(f"Gating Decision -> DIRECT PATH. Reasons: {'; '.join(reasons)}")
            return "direct"

    async def _run_direct_path(self, cognitive_state: CognitiveStatePacket, mcl_params: Dict, self_model,
                               chat_history: List[Dict[str, str]]) -> ResponsePacket:
        """Generates a fast, heuristic-based response. (V2 - Simplified Prompt)"""
        logger.info("Direct Path: Generating heuristic response.")

        agent_name = self_model.persona_attributes.get("name", self.mcl.agent_config.get("name", "Aura AI"))

        # Prompt simplificado focado em responder a pergunta e gerar JSON válido
        direct_response_prompt = f"""
                You are an AI assistant named {agent_name}.
                Your task is to answer the user's query directly and concisely.
                Use the provided context memories to inform your answer.

                **User Query:**
                "{cognitive_state.original_intent.query_vector.source_text}"

                **Relevant Context Memories:**
                - {[v.source_text for v in cognitive_state.relevant_memory_vectors[:3]]}

                **Your Task:**
                Generate a valid JSON object representing your answer.
                The JSON object must have these exact keys: "content_summary", "response_emotional_tone", "confidence_score".

                **JSON Response:**
                """

        response_str = await self.llm.ainvoke(
            LLM_MODEL_FAST,
            direct_response_prompt,
            temperature=mcl_params.get('ora_parameters', {}).get('temperature', 0.5)
        )

        response_json = extract_json_from_text(response_str)

        if not response_json or not isinstance(response_json, dict):
            logger.error(f"Direct Path: Failed to extract JSON from LLM. Raw response: '{response_str}'. Using raw string as fallback.")
            return ResponsePacket(content_summary=response_str, response_emotional_tone="neutral", confidence_score=0.6)

        try:
            # Garante que as chaves essenciais existam, mesmo que o LLM esqueça alguma
            response_json.setdefault("content_summary", "I'm not sure how to respond to that.")
            response_json.setdefault("response_emotional_tone", "neutral")
            response_json.setdefault("confidence_score", 0.7)
            return ResponsePacket.model_validate(response_json)
        except ValidationError as e:
            logger.error(f"Direct Path: Pydantic validation error: {e}. Using raw content.")
            return ResponsePacket(content_summary=response_json.get("content_summary", "Error generating response."),
                                  response_emotional_tone="apologetic", confidence_score=0.3)

    def _modulate_vre_assessment(self, vre_assessment: RefinementPacket, social_context: Dict) -> RefinementPacket:
        """The Ego's core function: applying social context to the Superego's judgment."""
        if not vre_assessment.adjustment_vectors:
            return vre_assessment  # No concerns to modulate

        logger.warning("CognitiveMediator (Ego): Modulating VRE (Superego) assessment based on social context.")

        modulated_assessment = vre_assessment.copy(deep=True)

        is_casual_context = social_context.get("stakes") == "low" and social_context.get("formality") == "casual"

        if is_casual_context:
            # Palavras-chave que identificam a falácia que queremos ignorar
            anthropomorphism_keywords = ["antropomorfismo", "anthropomorphism", "alegação de sentimento",
                                         "claim of feeling", "não é justificada pelo estado interno"]

            # Filtra as recomendações textuais, removendo as que batem com as keywords
            original_rec_count = len(modulated_assessment.textual_recommendations)
            modulated_assessment.textual_recommendations = [
                rec for rec in modulated_assessment.textual_recommendations
                if not any(keyword in rec.lower() for keyword in anthropomorphism_keywords)
            ]

            # Filtra os vetores de ajuste da mesma forma
            modulated_assessment.adjustment_vectors = [
                vec for vec in modulated_assessment.adjustment_vectors
                if not any(keyword in vec.description.lower() for keyword in anthropomorphism_keywords)
            ]

            if len(modulated_assessment.textual_recommendations) < original_rec_count:
                logger.critical(
                    "EGO EM AÇÃO: Suprimindo a correção de 'Antropomorfismo Inautêntico' do VRE devido ao contexto casual e de baixo risco."
                )


        # Rule: In casual, low-stakes conversations, suppress excessive transparency disclaimers.
        if social_context["stakes"] == "low" and social_context["formality"] == "casual":

            # Filter textual recommendations
            filtered_recs = [
                rec for rec in modulated_assessment.textual_recommendations
                if "transparency" not in rec.lower() and "limitation" not in rec.lower()
            ]

            # Filter adjustment vectors
            filtered_vectors = [
                vec for vec in modulated_assessment.adjustment_vectors
                if "transparency" not in vec.description.lower() and "limitation" not in vec.description.lower()
            ]

            if len(filtered_recs) < len(modulated_assessment.textual_recommendations):
                logger.critical(
                    "EGO IN ACTION: Suppressed VRE's transparency/limitation requirement for casual context.")
                modulated_assessment.textual_recommendations = filtered_recs
                modulated_assessment.adjustment_vectors = filtered_vectors

        # Add more modulation rules here in the future (e.g., for high-stakes, amplify safety concerns)

        if social_context.get("query_type") == "technical_factual":
            concerns_to_suppress = ["overconfidence", "tentative language", "humility", "uncertainty qualifiers"]

            original_vec_count = len(modulated_assessment.adjustment_vectors)

            # Filtra vetores de ajuste
            modulated_assessment.adjustment_vectors = [
                vec for vec in modulated_assessment.adjustment_vectors
                if not any(concern in vec.description.lower() for concern in concerns_to_suppress)
            ]

            # Filtra recomendações textuais
            modulated_assessment.textual_recommendations = [
                rec for rec in modulated_assessment.textual_recommendations
                if not any(concern in rec.lower() for concern in concerns_to_suppress)
            ]

            if len(modulated_assessment.adjustment_vectors) < original_vec_count:
                logger.critical(
                    "EGO EM AÇÃO: Suprimindo preocupações de humildade/confiança do VRE devido ao contexto técnico/factual.")
            # +++ FIM DA MUDANÇA +++

        return modulated_assessment
