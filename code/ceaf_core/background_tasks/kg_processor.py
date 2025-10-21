# ceaf_core/background_tasks/kg_processor.py
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_core.core_schema import ValidationInfo

from ceaf_core.services.llm_service import LLMService, LLM_MODEL_SMART, LLM_MODEL_FAST
from ceaf_core.services.mbs_memory_service import MBSMemoryService
from ceaf_core.modules.memory_blossom.memory_types import (
    ExplicitMemory, KGEntityRecord, KGRelationRecord, MemorySourceType,
    MemorySalience, KGEntityType
)
from ceaf_core.utils.common_utils import extract_json_from_text

logger = logging.getLogger("KGProcessor")


# +++ START OF FIX: Make KGEntity model more robust to LLM output +++
class KGEntity(BaseModel):
    id_str: str
    label: str
    type: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)

    @field_validator('id_str', mode='before')
    @classmethod
    def normalize_id_field(cls, v, info: 'ValidationInfo'):
        """
        Accepts common incorrect key names for the entity ID ('_str', 'id', 'entity_id')
        and normalizes them to 'id_str' before validation.
        """
        # If 'id_str' is already provided correctly, use it.
        if v:
            return v

        # The 'info.data' object holds the full raw input dictionary.
        raw_data = info.data

        # Check for common variations and return the first one found.
        if '_str' in raw_data:
            return raw_data['_str']
        if 'id' in raw_data:
            return raw_data['id']
        if 'entity_id' in raw_data:
            return raw_data['entity_id']

        # If none are found, return the original value (None) and let the
        # standard Pydantic validation raise the "Field required" error.
        return v

    @field_validator('label', mode='before')
    @classmethod
    def accept_name_for_label(cls, v, info: 'ValidationInfo'):
        """
        Accepts 'name' as an alternative for the 'label' field.
        """
        if v:
            return v
        if 'name' in info.data:
            return info.data['name']
        return v



class KGRelation(BaseModel):
    source_id_str: str
    target_id_str: str
    label: str
    context: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class KGSynthesisOutput(BaseModel):
    extracted_entities: List[KGEntity] = Field(default_factory=list)
    extracted_relations: List[KGRelation] = Field(default_factory=list)


class KGProcessor:
    # ... (__init__ and _repair_json_with_llm methods are unchanged) ...
    def __init__(self, llm_service: LLMService, memory_service: MBSMemoryService):
        self.llm = llm_service
        self.mbs = memory_service
        allowed_entity_types = [e.value for e in KGEntityType]

        self.synthesis_prompt_template = f"""
                You are a Knowledge Graph Synthesizer. Your function is to process text from an AI's memories
                and extract structured knowledge as entities and relationships.

                **CRITICAL RULES:**
                1.  **Entity `type` field:** The `type` field for each entity MUST be one of the following exact values: {json.dumps(allowed_entity_types)}.
                2.  **Relation `id_str` fields:** Every object in the `extracted_relations` list MUST have BOTH a `source_id_str` and a `target_id_str`.
                3.  **Output Format:** Your response MUST BE a single, valid JSON object with top-level keys "extracted_entities" and "extracted_relations".

                **Memory Text to Process:**
                ---
                {{memory_text}}
                ---

                **Correct JSON Output Schema (using 'label' for entities):**
                {{{{
                  "extracted_entities": [
                    {{{{
                      "id_str": "...", "label": "Entity Label", "type": "...", ...
                    }}}}
                  ],
                  "extracted_relations": [
                    {{{{ "source_id_str": "...", "target_id_str": "...", "label": "...", ... }}}}
                  ]
                }}}}

                **Your JSON Output:**
                """

    async def _repair_json_with_llm(self, broken_json_str: str, error_message: str) -> Optional[Dict[str, Any]]:
        """
        Attempts to repair a broken JSON string using an LLM call.
        """
        logger.warning(f"KGProcessor: Attempting to repair broken JSON with LLM. Error: {error_message}")
        prompt = f"""
        The following text was intended to be a valid JSON object, but it failed validation with this error:
        Error: "{error_message}"

        Broken JSON Text:
        ```json
        {broken_json_str}
        ```

        Your task is to fix the JSON text so it becomes valid according to the schema.
        - Correct any syntax errors (missing commas, brackets, quotes).
        - Ensure all required fields are present. For entities, use the field name "label" instead of "name". For relations, ensure 'source_id_str' and 'target_id_str' are present.
        - Do NOT change the content of the fields, only the structure and field names.

        Respond ONLY with the corrected, valid JSON object.
        """
        try:
            repaired_str = await self.llm.ainvoke(LLM_MODEL_FAST, prompt, temperature=0.0)
            repaired_json = extract_json_from_text(repaired_str)
            if isinstance(repaired_json, dict):
                logger.info("KGProcessor: Successfully repaired JSON with LLM.")
                return repaired_json
            else:
                logger.error(
                    f"KGProcessor: LLM-based JSON repair did not return a valid dictionary. Response: {repaired_str}")
        except Exception as e:
            logger.error(f"KGProcessor: An exception occurred during the LLM-based JSON repair process: {e}")

        return None

    # ... (process_memories_to_kg method is unchanged as the fix is in the Pydantic model) ...
    async def process_memories_to_kg(self, memories: List[ExplicitMemory]) -> Tuple[int, int]:
        # This function does not need to be changed. The Pydantic model fix handles the logic.
        if not memories:
            return 0, 0

        total_entities = 0
        total_relations = 0

        for memory in memories:
            text_content, _ = await self.mbs._get_searchable_text_and_keywords(memory)
            if not text_content or len(text_content.split()) < 5:
                continue

            prompt = self.synthesis_prompt_template.format(memory_text=text_content)

            try:
                response_str = await self.llm.ainvoke(LLM_MODEL_SMART, prompt, temperature=0.0)
                json_output = extract_json_from_text(response_str)

                if not json_output:
                    logger.warning(f"KGProcessor: No valid JSON extracted for memory {memory.memory_id}. Skipping.")
                    continue

                if isinstance(json_output, list):
                    logger.warning(
                        f"KGProcessor: LLM returned a list instead of a dict for memory {memory.memory_id}. Attempting to fix.")
                    if json_output and isinstance(json_output[0], dict) and ('id_str' in json_output[0]):
                        json_output = {"extracted_entities": json_output, "extracted_relations": []}
                    else:
                        logger.error(
                            f"KGProcessor: Could not safely repair list-based JSON output for memory {memory.memory_id}. Skipping.")
                        continue

                synthesis_result = None
                try:
                    synthesis_result = KGSynthesisOutput.model_validate(json_output)
                except ValidationError as e:
                    logger.error(
                        f"KGProcessor: Pydantic validation failed for memory {memory.memory_id}. Attempting LLM repair. Details: {e}")

                    repaired_json = await self._repair_json_with_llm(json.dumps(json_output), str(e))
                    if repaired_json:
                        try:
                            synthesis_result = KGSynthesisOutput.model_validate(repaired_json)
                            logger.info(f"KGProcessor: LLM repair successful for memory {memory.memory_id}.")
                        except ValidationError as e2:
                            logger.error(
                                f"KGProcessor: Repaired JSON still failed validation for memory {memory.memory_id}: {e2}")
                            continue
                    else:
                        logger.error(
                            f"KGProcessor: LLM repair did not return valid JSON for memory {memory.memory_id}. Skipping.")
                        continue

                if not synthesis_result:
                    logger.error(
                        f"KGProcessor: Could not obtain a valid synthesis result for memory {memory.memory_id} after all steps. Skipping.")
                    continue

                # Commit Entities
                for entity_data in synthesis_result.extracted_entities:
                    entity_type_enum = KGEntityType.OTHER
                    try:
                        entity_type_str = getattr(entity_data, 'type', 'OTHER') or 'OTHER'
                        entity_type_enum = KGEntityType[entity_type_str.upper()]
                    except KeyError:
                        logger.warning(f"Unknown KG entity type '{entity_data.type}' from LLM. Defaulting to OTHER.")

                    entity_record = KGEntityRecord(
                        entity_id_str=entity_data.id_str,
                        label=entity_data.label,
                        entity_type=entity_type_enum,
                        description=entity_data.description,
                        attributes=entity_data.attributes,
                        aliases=entity_data.aliases,
                        source_type=MemorySourceType.INTERNAL_REFLECTION,
                        salience=MemorySalience.MEDIUM,
                        metadata={"source_memory_id": memory.memory_id}
                    )
                    await self.mbs.add_specific_memory(entity_record)
                    total_entities += 1

                # Commit Relations
                for relation_data in synthesis_result.extracted_relations:
                    relation_record = KGRelationRecord(
                        source_entity_id_str=relation_data.source_id_str,
                        target_entity_id_str=relation_data.target_id_str,
                        relation_label=relation_data.label,
                        description=relation_data.context,
                        attributes=relation_data.attributes,
                        source_type=MemorySourceType.INTERNAL_REFLECTION,
                        salience=MemorySalience.MEDIUM,
                        metadata={"source_memory_id": memory.memory_id}
                    )
                    await self.mbs.add_specific_memory(relation_record)
                    total_relations += 1

            except Exception as e:
                logger.error(f"KGProcessor: Unhandled exception while processing memory {memory.memory_id}: {e}",
                             exc_info=True)
                continue

        return total_entities, total_relations