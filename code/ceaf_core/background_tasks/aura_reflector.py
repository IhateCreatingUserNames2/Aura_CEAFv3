# Em: ceaf_core/background_tasks/aura_reflector.py

import logging
import re

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


from agent_manager import AgentManager
from .kg_processor import KGProcessor
from ceaf_core.modules.memory_blossom.memory_types import (
    ExplicitMemory,
    ExplicitMemoryContent,
    MemorySourceType,
    MemorySalience
)



from ceaf_core.services.cognitive_log_service import CognitiveLogService
from ceaf_core.system import save_ceaf_dynamic_config, load_ceaf_dynamic_config
from ceaf_core.modules.memory_blossom.advanced_synthesizer import AdvancedMemorySynthesizer, StoryArcType
from ..services.llm_service import LLM_MODEL_SMART

logger = logging.getLogger("AuraReflector")

CONFIDENCE_THRESHOLD_FOR_SUCCESS = 0.75
MIN_TURNS_FOR_ANALYSIS = 5


def analyze_correlation_guidance_confidence(turn_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analisa a correlação entre os biases de orientação (coerência vs. novidade) do MCL
    e a confiança da resposta final, para descobrir se o agente está sendo muito
    "caótico" ou muito "rígido".
    """
    results = {
        "coherence_leaning_success_rate": 0.0,
        "novelty_leaning_success_rate": 0.0,
        "coherence_turn_count": 0,
        "novelty_turn_count": 0,
        "suggestion": "insufficient_data"
    }

    coherence_successes = 0
    novelty_successes = 0

    for turn in turn_history:
        try:
            # Pega a orientação completa do MCL que foi salva no log
            mcl_guidance = turn.get("mcl_guidance")
            if not mcl_guidance:
                continue  # Pula turnos que não têm o log de orientação

            # Extrai os biases que foram REALMENTE usados naquele turno
            biases = mcl_guidance.get("biases")
            if not biases:
                continue

            coherence_bias = biases.get("coherence_bias", 0.5)
            novelty_bias = biases.get("novelty_bias", 0.5)

            # Verifica se o resultado foi um "sucesso" (confiança alta)
            is_successful = turn["response_packet"]["confidence_score"] > CONFIDENCE_THRESHOLD_FOR_SUCCESS

            # Classifica o turno como orientado a coerência ou novidade e conta os sucessos
            if coherence_bias > novelty_bias:
                results["coherence_turn_count"] += 1
                if is_successful:
                    coherence_successes += 1
            elif novelty_bias > coherence_bias:  # Usamos elif para ignorar o caso de empate
                results["novelty_turn_count"] += 1
                if is_successful:
                    novelty_successes += 1
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"AuraReflector: Pulando turno malformado durante análise de correlação: {e}")
            continue

    # Calcula as taxas de sucesso se houver dados suficientes
    if results["coherence_turn_count"] > 5:
        results["coherence_leaning_success_rate"] = coherence_successes / results["coherence_turn_count"]

    if results["novelty_turn_count"] > 5:
        results["novelty_leaning_success_rate"] = novelty_successes / results["novelty_turn_count"]

    # Gera uma sugestão de ajuste com base na comparação das taxas de sucesso
    coh_rate = results["coherence_leaning_success_rate"]
    nov_rate = results["novelty_leaning_success_rate"]

    # Apenas sugere uma mudança se houver dados para ambos os tipos de orientação e uma diferença significativa
    if coh_rate > 0 and nov_rate > 0:
        if coh_rate > nov_rate + 0.15:  # Coerência é 15% mais bem-sucedida
            results["suggestion"] = "increase_coherence_bias"
        elif nov_rate > coh_rate + 0.15:  # Novidade é 15% mais bem-sucedida
            results["suggestion"] = "increase_novelty_bias"
        else:
            results["suggestion"] = "maintain_balance"

    return results

def analyze_loss_to_breakthrough_cycles(turn_history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Identifica padrões onde 'struggle' leva a 'breakthrough'."""
    for i in range(1, len(turn_history)):
        prev_turn = turn_history[i-1]
        curr_turn = turn_history[i]

        # Extrai os estados do MCL do log
        prev_state = prev_turn.get("mcl_guidance", {}).get("cognitive_state")
        curr_state = curr_turn.get("mcl_guidance", {}).get("cognitive_state")

        was_in_struggle = prev_state in ["PRODUCTIVE_CONFUSION", "EDGE_OF_CHAOS"]
        is_breakthrough = curr_state == "STABLE_OPERATION" and prev_turn.get("response_packet",{}).get("confidence_score", 0) < 0.6 and curr_turn.get("response_packet",{}).get("confidence_score", 0) > 0.8

        if was_in_struggle and is_breakthrough:
            logger.warning(f"AURA Insight: Ciclo de Luta-para-Descoberta detectado entre os turnos {prev_turn['turn_id']} e {curr_turn['turn_id']}.")
            # Retorna uma recomendação acionável para o MCL
            return {
                "target_module": "MCL",
                "action": "ADJUST_BIAS",
                "parameter": "novelty_bias", # Se a luta leva ao sucesso, talvez mais novidade seja bom
                "adjustment_value": 0.05,
                "reasoning": "Struggle states are leading to high-confidence breakthroughs. Encouraging more novelty."
            }
    return None

def analyze_chaotic_failures(turn_history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Identifica padrões onde um alto viés de novidade leva a respostas de baixa confiança (falha)."""
    for turn in turn_history:
        try:
            mcl_guidance = turn.get("mcl_guidance", {})
            biases = mcl_guidance.get("biases", {})
            novelty_bias = biases.get("novelty_bias", 0.0)
            confidence_score = turn.get("response_packet", {}).get("confidence_score", 1.0)

            was_exploring_chaotically = novelty_bias > 0.7  # Alto viés para novidade
            resulted_in_failure = confidence_score < 0.5   # Baixa confiança na resposta final

            if was_exploring_chaotically and resulted_in_failure:
                logger.warning(f"AURA Insight: Falha caótica detectada no turno {turn['turn_id']}. Alto viés de novidade ({novelty_bias:.2f}) resultou em baixa confiança ({confidence_score:.2f}).")
                return {
                    "target_module": "MCL",
                    "action": "ADJUST_BIAS",
                    "parameter": "novelty_bias",
                    "adjustment_value": -0.07,  # <<< Ação Corretiva: Valor negativo para reduzir
                    "reasoning": "High novelty bias is leading to low-confidence responses. Reducing novelty to promote coherence."
                }
        except (KeyError, TypeError):
            continue
    return None


async def synthesize_procedural_wisdom(agent_id: str, agent_manager: AgentManager, turn_history: List[Dict[str, Any]]):
    logger.info(f"AURA-WISDOM: Iniciando síntese de sabedoria processual para o agente {agent_id}")
    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance: return

    # Filtra por turnos "interessantes" (onde algo foi aprendido)
    interesting_turns = [
        t for t in turn_history
        if t.get("mcl_guidance", {}).get("mcl_analysis", {}).get("agency_score", 0) > 3.0 or
           (t.get("response_packet", {}).get("confidence_score", 1.0) < 0.65)
    ]

    if not interesting_turns:
        logger.info(f"AURA-WISDOM: Nenhum turno interessante encontrado para síntese.")
        return

    # Processa um turno interessante por ciclo para não sobrecarregar
    turn_to_process = interesting_turns[0]

    # Verifica se já processamos este turno para não criar sabedoria duplicada
    turn_id = turn_to_process.get("turn_id")
    existing_wisdom_check = await agent_instance.memory_service.search_raw_memories(
        query=f"procedural wisdom from turn {turn_id}", top_k=1
    )
    if existing_wisdom_check:
        logger.info(f"AURA-WISDOM: Sabedoria para o turno {turn_id} já foi sintetizada. Pulando.")
        return

    # Monta o "estudo de caso" para o LLM
    try:
        user_query = turn_to_process["cognitive_state_packet"]["original_intent"]["query_vector"]["source_text"]
        final_response = turn_to_process["response_packet"]["content_summary"]

        case_study = f"""
        Estudo de Caso de Interação de IA:
        - Situação: O usuário perguntou sobre "{user_query}".
        - Resultado: A IA respondeu com sucesso, dizendo: "{final_response}".
        - Análise: Esta foi uma resposta bem-sucedida para uma pergunta sobre este tópico.
        """

        wisdom_prompt = f"""
        Analise o seguinte estudo de caso de uma interação de IA. Extraia uma única "lição processual" ou "estratégia de resposta" genérica e reutilizável.

        Estudo de Caso:
        {case_study}

        Exemplo de Saída:
        "Estratégia: Ao responder a perguntas filosóficas, é eficaz começar reconhecendo a complexidade e depois apresentar múltiplas perspectivas."

        Sua Saída (apenas a lição/estratégia):
        """

        wisdom_text = await agent_instance.llm_service.ainvoke(LLM_MODEL_SMART, wisdom_prompt, temperature=0.3)

        if wisdom_text and not wisdom_text.startswith("[LLM_ERROR]"):
            wisdom_memory = ExplicitMemory(
                content=ExplicitMemoryContent(
                    text_content=wisdom_text,
                    structured_data={
                        "type": "procedural_wisdom",
                        "source_turn_id": turn_id
                    }
                ),
                memory_type="explicit",
                source_type=MemorySourceType.SYNTHESIZED_SUMMARY,
                salience=MemorySalience.HIGH,
                keywords=["procedural_wisdom", "response_strategy", "lesson_learned"] + [w.lower() for w in
                                                                                         re.findall(r'\b\w{4,}\b',
                                                                                                    user_query)],
                learning_value=0.9
            )
            await agent_instance.memory_service.add_specific_memory(wisdom_memory)
            logger.critical(f"AURA-WISDOM: Nova memória de sabedoria sintetizada e salva: '{wisdom_text}'")

    except Exception as e:
        logger.error(f"AURA-WISDOM: Erro ao sintetizar sabedoria: {e}", exc_info=True)

async def perform_autonomous_clustering_and_synthesis(agent_id: str, agent_manager: AgentManager):
    logger.info(f"AURA-AMA: Iniciando síntese de cluster autônomo para o agente {agent_id}")
    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance: return

    # 1. Obtenha um lote de memórias recentes e importantes
    recent_memories_raw = await agent_instance.memory_service.search_raw_memories(query="*", top_k=30)
    recent_memories = [mem for mem, score in recent_memories_raw]

    if len(recent_memories) < 10: return

    # 2. Use o AdvancedSynthesizer para clusterizar e criar uma narrativa
    synthesizer = AdvancedMemorySynthesizer()
    synthesis_result = await synthesizer.synthesize_with_advanced_features(
        memories=recent_memories,
        context="general reflection on recent experiences",
        arc_type=StoryArcType.THEMATIC
    )

    # 3. Se a narrativa for coerente, crie uma nova meta-memória
    narrative_text = synthesis_result.get("narrative_text")
    coherence_score = synthesis_result.get("narrative_coherence", 0)

    if narrative_text and coherence_score > 0.7:
        # Crie uma nova memória que é o resumo do cluster
        summary_memory = ExplicitMemory(
            content=ExplicitMemoryContent(text_content=f"Reflection on a cluster of experiences: {narrative_text}"),
            memory_type="explicit",
            source_type=MemorySourceType.INTERNAL_REFLECTION,
            salience=MemorySalience.HIGH,  # Meta-memórias são importantes
            keywords=["synthesized-summary", "reflection"] + [c['theme'] for c in synthesis_result.get('clusters', [])],
            learning_value=coherence_score  # O valor de aprendizado é a coerência da síntese
        )
        await agent_instance.memory_service.add_specific_memory(summary_memory)
        logger.warning(f"AURA-AMA: Nova meta-memória de síntese criada para o agente {agent_id}.")


async def perform_kg_synthesis_cycle(agent_id: str, agent_manager: AgentManager):
    """
    Periodically checks for new, unprocessed explicit memories and synthesizes
    them into the knowledge graph.
    """
    logger.info(f"AURA-KGS: Iniciando ciclo de síntese de KG para o agente {agent_id}")
    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance:
        logger.error(f"AURA-KGS: Não foi possível obter a instância do agente {agent_id}.")
        return

    # Lógica para encontrar memórias não processadas
    # Uma abordagem simples: procurar memórias explícitas recentes sem um metadado "kg_processed"
    unprocessed_memories: List[ExplicitMemory] = []
    for mem in agent_instance.memory_service._in_memory_explicit_cache:
        if not mem.metadata.get("kg_processed"):
            unprocessed_memories.append(mem)

    if not unprocessed_memories:
        logger.info(f"AURA-KGS: Nenhuma memória nova para processar para o agente {agent_id}.")
        return

    logger.warning(f"AURA-KGS: Encontradas {len(unprocessed_memories)} memórias explícitas para síntese de KG.")

    # Processar em lotes
    batch = unprocessed_memories[:5]  # Processar 5 de cada vez para não sobrecarregar

    kg_processor = KGProcessor(agent_instance.llm_service, agent_instance.memory_service)
    entities_created, relations_created = await kg_processor.process_memories_to_kg(batch)

    if entities_created > 0 or relations_created > 0:
        logger.critical(
            f"AURA-KGS: Sucesso! Criados {entities_created} entidades e {relations_created} relações para o agente {agent_id}.")
        # Marcar as memórias como processadas para não reprocessá-las
        for mem in batch:
            mem.metadata["kg_processed"] = True
        # A atualização será salva na próxima reescrita do MBS, o que é eficiente.


# --- MÉTODO PRINCIPAl main_aura_reflector_cycle ---
async def main_aura_reflector_cycle(agent_manager: AgentManager):
    """
    Ciclo principal do Aura Reflector. Opera sobre o histórico cognitivo de cada agente
    para realizar auto-otimização e atualiza a configuração dinâmica em tempo real.
    (Versão Otimizada e Completa)
    """
    logger.info("--- Iniciando Ciclo do Aura Reflector (Auto-Otimização Ativa) ---")

    # Itera sobre uma cópia para evitar problemas com modificações concorrentes
    for agent_id, agent_config in list(agent_manager.agent_configs.items()):

        agent_instance = agent_manager.get_agent_instance(agent_id)
        if agent_instance and hasattr(agent_instance, 'body_state'):
            # Reduz drasticamente a fadiga e a saturação
            agent_instance.body_state.cognitive_fatigue *= 0.1  # Recupera 90%
            agent_instance.body_state.information_saturation *= 0.5  # Consolida/esquece 50%
            await agent_instance._save_body_state()
            logger.warning(f"AURA-REFLECTOR (Sleep): Estado corporal do agente {agent_id} restaurado.")

        try:
            logger.info(f"AuraReflector: Analisando agente '{agent_id}'...")

            # --- ETAPA 1: CARREGAR DADOS E VERIFICAR PRÉ-CONDIÇÕES ---
            persistence_path = Path(agent_config.persistence_path)
            cognitive_log = CognitiveLogService(persistence_path)
            turn_history = cognitive_log.get_recent_turns(limit=200)

            made_config_change = False
            agent_dynamic_config = None  # Carregar sob demanda

            # Verificação centralizada: Se não houver turnos suficientes, pule as análises baseadas em log.
            if len(turn_history) < MIN_TURNS_FOR_ANALYSIS:
                logger.info(
                    f"Agente '{agent_id}': Dados de log insuficientes para análise de auto-ajuste ({len(turn_history)}/{MIN_TURNS_FOR_ANALYSIS} turnos).")
            else:
                # --- ETAPA 2: EXECUTAR ANÁLISES DE LOG PARA AUTO-AJUSTE ---

                # Análise de Correlação (Heurística)
                correlation_analysis = analyze_correlation_guidance_confidence(turn_history)
                logger.info(
                    f"Agente '{agent_id}': Análise de Correlação concluída. Sugestão: {correlation_analysis['suggestion']}")

                # Análise de Ciclos do AURA
                aura_recommendation = analyze_loss_to_breakthrough_cycles(turn_history)
                #  Análise de Falhas Caóticas
                aura_failure_rec = analyze_chaotic_failures(turn_history)

                await synthesize_procedural_wisdom(agent_id, agent_manager, turn_history)
                # --- ETAPA 3: APLICAR RECOMENDAÇÕES DAS ANÁLISES ---
                # Aplicar recomendação da Análise de Correlação
                suggestion = correlation_analysis.get("suggestion")
                if suggestion and suggestion not in ["insufficient_data", "maintain_balance"]:
                    if not agent_dynamic_config: agent_dynamic_config = load_ceaf_dynamic_config(persistence_path)

                    params = agent_dynamic_config.get("MCL", {}).get("state_to_params_map", {}).get(
                        "PRODUCTIVE_CONFUSION")
                    if not params:
                        logger.warning(
                            f"Agente '{agent_id}': Configuração para 'PRODUCTIVE_CONFUSION' não encontrada para ajuste de correlação.")
                    else:
                        original_coh_bias = params["coherence_bias"]
                        adjustment_amount = 0.05
                        if suggestion == "increase_coherence_bias" and original_coh_bias < 0.9:
                            params["coherence_bias"] = min(0.9, original_coh_bias + adjustment_amount)
                            params["novelty_bias"] = 1.0 - params["coherence_bias"]
                            logger.warning(
                                f"AURA-ADJUSTMENT for '{agent_id}': Aumentando Coherence Bias para {params['coherence_bias']:.2f} baseado na correlação.")
                            made_config_change = True
                        elif suggestion == "increase_novelty_bias" and params["novelty_bias"] < 0.9:
                            params["novelty_bias"] = min(0.9, params["novelty_bias"] + adjustment_amount)
                            params["coherence_bias"] = 1.0 - params["novelty_bias"]
                            logger.warning(
                                f"AURA-ADJUSTMENT for '{agent_id}': Aumentando Novelty Bias para {params['novelty_bias']:.2f} baseado na correlação.")
                            made_config_change = True

                if aura_failure_rec and aura_failure_rec.get("target_module") == "MCL":
                    if not agent_dynamic_config: agent_dynamic_config = load_ceaf_dynamic_config(persistence_path)

                    params = agent_dynamic_config["MCL"]["state_to_params_map"].get("PRODUCTIVE_CONFUSION")
                    if params:
                        old_bias = params["novelty_bias"]
                        # Usamos max() para garantir que o bias não fique negativo
                        params["novelty_bias"] = max(0.1, old_bias + aura_failure_rec["adjustment_value"])
                        params["coherence_bias"] = 1.0 - params["novelty_bias"]
                        logger.critical(
                            f"AURA-CORRECTION for '{agent_id}': Novelty Bias REDUZIDO para {params['novelty_bias']:.2f} devido a falhas caóticas.")
                        made_config_change = True

                # Aplicar recomendação do AURA
                elif aura_recommendation and aura_recommendation.get("target_module") == "MCL":
                    if not agent_dynamic_config: agent_dynamic_config = load_ceaf_dynamic_config(persistence_path)

                    if aura_recommendation["parameter"] == "novelty_bias":
                        params = agent_dynamic_config["MCL"]["state_to_params_map"].get("PRODUCTIVE_CONFUSION")
                        if not params:
                            logger.warning(
                                f"Agente '{agent_id}': Configuração para 'PRODUCTIVE_CONFUSION' não encontrada para ajuste do AURA.")
                        else:
                            old_bias = params["novelty_bias"]
                            params["novelty_bias"] = min(0.9, old_bias + aura_recommendation["adjustment_value"])
                            params["coherence_bias"] = 1.0 - params["novelty_bias"]
                            logger.warning(
                                f"AURA-ADJUSTMENT for '{agent_id}': Novelty Bias para PRODUCTIVE_CONFUSION ajustado para {params['novelty_bias']:.2f} baseado no insight do AURA.")
                            made_config_change = True

            # Se qualquer análise resultou em uma mudança, salve a configuração e atualize a instância ativa
            if made_config_change and agent_dynamic_config:
                await save_ceaf_dynamic_config(persistence_path, agent_dynamic_config)
                active_agent_instance = agent_manager._active_agents.get(agent_id)
                if active_agent_instance:
                    active_agent_instance.ceaf_dynamic_config = agent_dynamic_config
                    logger.info(
                        f"AuraReflector: Configuração dinâmica da instância ativa '{agent_id}' foi atualizada em memória.")

            # --- ETAPA 4: EXECUTAR SÍNTESE AUTÔNOMA (SEMPRE, INDEPENDENTE DOS LOGS) ---
            # Esta tarefa opera sobre as memórias para criar meta-memórias.
            await perform_autonomous_clustering_and_synthesis(agent_id, agent_manager)

            # --- ETAPA 5: EXECUTAR SÍNTESE DE GRAFO DE CONHECIMENTO ---
            await perform_kg_synthesis_cycle(agent_id, agent_manager)

        except Exception as e:
            logger.error(f"AuraReflector: Erro inesperado ao analisar o agente '{agent_id}': {e}", exc_info=True)

    logger.info("--- Ciclo do Aura Reflector Concluído ---")