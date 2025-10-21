# Em: ceaf_core/translators/human_to_genlang.py

import asyncio
import json
import re

from pydantic import ValidationError
from ceaf_core.genlang_types import IntentPacket, GenlangVector
from ceaf_core.utils.embedding_utils import get_embedding_client
# NOVAS IMPORTAÇÕES
from ceaf_core.services.llm_service import LLMService, LLM_MODEL_FAST  # Usaremos um modelo rápido para análise
from ceaf_core.utils.common_utils import extract_json_from_text
import logging

logger = logging.getLogger("CEAFv3_System")

class HumanToGenlangTranslator:
    def __init__(self):
        self.embedding_client = get_embedding_client()
        # NOVO: Instância do LLMService para o tradutor
        self.llm_service = LLMService()

    async def translate(self, query: str, metadata: dict) -> IntentPacket:
        """
        Versão V1.1: Usa uma LPU com prompt robusto para analisar a query humana.
        """
        print("--- [HTG Translator v1.1] Analisando query humana com LPU... ---")

        # +++ INÍCIO DO PROMPT REFINADO +++
        analysis_prompt = f"""
                Analyze the user's query: "{query}"
                Respond ONLY with a single, valid JSON object with the exact structure:
                {{
                  "core_query": "<Reformulated core question>",
                  "intent_description": "<User's primary intent>",
                  "emotional_tone_description": "<User's emotional tone>",
                  "key_entities": ["<List of key nouns/concepts>"]
                }}
                """

        analysis_json = None
        # ==================== LÓGICA DE RETRY ====================
        for attempt in range(2):  # Tenta até 2 vezes
            analysis_str = await self.llm_service.ainvoke(LLM_MODEL_FAST, analysis_prompt, temperature=0.0)

            try:
                extracted_json = extract_json_from_text(analysis_str)
                if isinstance(extracted_json, dict):
                    # Validação mínima
                    required_keys = ["core_query", "intent_description", "emotional_tone_description", "key_entities"]
                    if all(key in extracted_json for key in required_keys):
                        analysis_json = extracted_json
                        break  # Sucesso, sai do loop

                logger.warning(
                    f"HTG Translator (Attempt {attempt + 1}): Invalid JSON structure. Raw: '{analysis_str[:100]}'")

            except Exception as e:
                logger.warning(
                    f"HTG Translator (Attempt {attempt + 1}): Exception during parsing. Error: {e}. Raw: '{analysis_str[:100]}'")

        # Se após as tentativas ainda falhar, use o fallback
        if not analysis_json:
            logger.error("HTG Translator: Falha na análise da LPU após retries. Usando fallback.")
            analysis_json = {
                "core_query": query,
                "intent_description": "unknown_intent",
                "emotional_tone_description": "unknown_emotion",
                "key_entities": []
            }

        if not analysis_json.get("key_entities") and analysis_json.get("core_query"):
            logger.warning(
                "HTG Translator: 'key_entities' está vazia. Aplicando fallback de extração de palavras-chave.")
            core_query = analysis_json["core_query"]
            # Extrai palavras com 4 a 15 caracteres alfanuméricos
            fallback_keywords = list(set(re.findall(r'\b\w{4,15}\b', core_query.lower())))
            analysis_json["key_entities"] = fallback_keywords
            logger.info(f"HTG Translator: Palavras-chave de fallback extraídas: {fallback_keywords}")

        # Gera todos os embeddings necessários em paralelo para eficiência
        texts_to_embed = [
                             analysis_json.get("core_query", query),
                             analysis_json.get("intent_description", "unknown"),
                             analysis_json.get("emotional_tone_description", "unknown")
                         ] + analysis_json.get("key_entities", [])

        embeddings = await self.embedding_client.get_embeddings(texts_to_embed, context_type="default_query")

        # Monta os GenlangVectors
        query_vector = GenlangVector(
            vector=embeddings[0],
            source_text=analysis_json.get("core_query", query),
            model_name=self.embedding_client._resolve_model_name("default_query")
        )
        intent_vector = GenlangVector(
            vector=embeddings[1],
            source_text=analysis_json.get("intent_description"),
            model_name=self.embedding_client._resolve_model_name("default_query")
        )
        emotional_vector = GenlangVector(
            vector=embeddings[2],
            source_text=analysis_json.get("emotional_tone_description"),
            model_name=self.embedding_client._resolve_model_name("default_query")
        )
        entity_vectors = [
            GenlangVector(
                vector=emb,
                source_text=text,
                model_name=self.embedding_client._resolve_model_name("default_query")
            ) for text, emb in zip(analysis_json.get("key_entities", []), embeddings[3:])
        ]

        intent_packet = IntentPacket(
            query_vector=query_vector,
            intent_vector=intent_vector,
            emotional_valence_vector=emotional_vector,
            entity_vectors=entity_vectors,
            metadata=metadata
        )

        print(
            f"--- [HTG Translator] Análise completa. Intenção: '{intent_vector.source_text}', Emoção: '{emotional_vector.source_text}' ---")
        return intent_packet