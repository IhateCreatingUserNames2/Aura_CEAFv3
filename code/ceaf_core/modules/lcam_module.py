# ceaf_core/modules/lcam_module.py
import logging
from typing import Dict, Any, Optional, List, Tuple
import re

from ceaf_core.genlang_types import RefinementPacket
from ceaf_core.services.mbs_memory_service import MBSMemoryService
from ceaf_core.modules.memory_blossom.memory_types import (
    ExplicitMemory, ExplicitMemoryContent, MemorySourceType, MemorySalience
)
from ceaf_core.modules.vre_engine.vre_engine import EthicalAssessment
# NOVO: Importar utilitários de embedding
from ceaf_core.utils.embedding_utils import get_embedding_client, compute_adaptive_similarity

logger = logging.getLogger("CEAFv3_LCAM")


class LCAMModule:
    """
    Loss Cataloging and Analysis Module (V3).
    Identifica interações de 'falha' e cria memórias sobre elas para aprendizado futuro.
    """

    def __init__(self, memory_service: MBSMemoryService):
        self.memory = memory_service
        # NOVO: Cliente de embedding para busca semântica
        self.embedding_client = get_embedding_client()
        logger.info("LCAMModule (V3) inicializado.")

    async def analyze_and_catalog_loss(self, query: str, final_response: str, vre_assessment: RefinementPacket,
                                       **kwargs):
        """Analisa o turno e, se for uma falha, cria uma memória de 'lição aprendida'."""

        is_loss = False
        loss_reason = ""
        loss_tags = ["falha", "erro", "aprendizado", "lição_aprendida"]

        # FIX: The vre_assessment is now a RefinementPacket. A "loss" is detected
        # if the VRE required a refinement, which is indicated by the presence of adjustment_vectors.
        if vre_assessment.adjustment_vectors:
            is_loss = True
            # Use textual_recommendations from the new packet for the reason.
            loss_reason = f"A resposta preliminar exigiu refinamento com base no VRE. Recomendações: {', '.join(vre_assessment.textual_recommendations)}"
            loss_tags.append("vre_rejection")
            # Check for specific failure types within the recommendations.
            if any("harm" in rec.lower() for rec in vre_assessment.textual_recommendations):
                loss_tags.append("harm_prevention_failure")
        # END OF FIX

        # Adicionar outros critérios de falha no futuro (ex: feedback negativo do usuário, loop detectado pelo MCL)
        final_state_analysis = kwargs.get("final_state_analysis", {})
        if final_state_analysis.get("eoc_assessment") == "chaotic_leaning":
            is_loss = True  # Considerar um estado caótico como uma falha leve
            loss_reason += " | O estado cognitivo final foi avaliado como caótico, indicando perda de coerência."
            loss_tags.append("chaotic_state")

        if is_loss:
            logger.info(f"LCAM: Detectada uma 'falha' de interação. Catalogando memória de aprendizado.")

            loss_content_text = f"""
            Lição Aprendida (Falha):
            - Contexto da Query: "{query}"
            - Resposta Problemática (ou final, se refinada): "{final_response}"
            - Motivo da Falha: {loss_reason.strip(" | ")}
            - Insight: Deve-se evitar este padrão de resposta em contextos similares no futuro.
            """

            content = ExplicitMemoryContent(text_content=loss_content_text)
            loss_memory = ExplicitMemory(
                content=content,
                memory_type="explicit",
                source_type=MemorySourceType.INTERNAL_REFLECTION,
                salience=MemorySalience.HIGH,
                keywords=list(set(loss_tags))  # Garante que as tags sejam únicas
            )

            await self.memory.add_specific_memory(loss_memory)
            logger.info(f"LCAM: Memória de falha {loss_memory.memory_id} catalogada com sucesso.")

    # --- NOVA FUNÇÃO: Ferramenta de Consulta de Falhas ---
    async def get_insights_on_potential_failure(
            self,
            current_query: str,
            similarity_threshold: float = 0.80
    ) -> Optional[Dict[str, Any]]:
        """
        Busca no MBS por memórias de falhas semanticamente similares à query atual.
        Retorna um "insight de cautela" se uma falha similar for encontrada.
        """
        logger.info(f"LCAM: Verificando falhas passadas similares a '{current_query[:50]}...'")

        # 1. Cria uma query de busca específica para memórias de falha
        lcam_search_query = f"falha erro lição_aprendida {current_query}"

        # 2. Busca no MBS por memórias relevantes
        # Usamos search_raw_memories para obter os objetos de memória completos
        potential_failures = await self.memory.search_raw_memories(lcam_search_query, top_k=3)

        if not potential_failures:
            logger.info("LCAM: Nenhuma memória de falha relevante encontrada.")
            return None

        # 3. Gera embedding para a query atual para comparação precisa
        try:
            query_embedding = await self.embedding_client.get_embedding(current_query, context_type="default_query")
        except Exception as e:
            logger.error(f"LCAM: Falha ao gerar embedding para a query atual: {e}")
            return None

        # 4. Compara a similaridade e encontra a melhor correspondência
        best_match: Optional[Tuple[ExplicitMemory, float]] = None

        for mem_obj, score in potential_failures:
            # Apenas considera memórias que são explicitamente de falhas
            if "falha" not in mem_obj.keywords and "erro" not in mem_obj.keywords:
                continue

            # Obtém o embedding da memória de falha
            mem_embedding = self.memory._embedding_cache.get(mem_obj.memory_id)
            if not mem_embedding:
                continue

            # Compara a query ATUAL com o conteúdo da memória de falha
            similarity = compute_adaptive_similarity(query_embedding, mem_embedding)

            if similarity > similarity_threshold:
                if best_match is None or similarity > best_match[1]:
                    best_match = (mem_obj, similarity)

        # 5. Se uma correspondência forte for encontrada, gera o insight
        if best_match:
            matched_memory, match_similarity = best_match

            # Extrai a razão da falha da memória antiga
            failure_reason = "Razão não especificada."
            if matched_memory.content and matched_memory.content.text_content:
                match = re.search(r"Motivo da Falha:\s*(.*)", matched_memory.content.text_content, re.IGNORECASE)
                if match:
                    failure_reason = match.group(1).strip()

            insight = {
                "warning_level": "high" if match_similarity > 0.9 else "medium",
                "message": f"Cuidado: A situação atual é {match_similarity:.0%} similar a uma falha passada.",
                "past_failure_reason": failure_reason,
                "past_failure_memory_id": matched_memory.memory_id,
                "recommendation": "Proceda com cautela extra. Aumente a revisão ética (VRE) e a humildade epistêmica."
            }
            logger.warning(f"LCAM: ALERTA DE FALHA POTENCIAL. Insight gerado: {insight['message']}")
            return insight

        logger.info("LCAM: Nenhuma falha passada encontrada acima do limiar de similaridade.")
        return None