# ARQUIVO REATORADO: ceaf_core/utils/common_utils.py

import json
import logging
import re
from typing import Dict, Any, Optional, List, Union, Type
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


# --- Text Processing Utilities ---

def sanitize_text_for_logging(text: Optional[str], max_length: int = 150) -> str:
    """Sanitiza e trunca texto para logs, escapando novas linhas."""
    if not text:
        return "<empty>"
    sanitized = text.replace("\n", "\\n").replace("\r", "\\r")
    if len(sanitized) > max_length:
        return sanitized[:max_length] + "..."
    return sanitized


def extract_json_from_text(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Extrai o primeiro objeto ou array JSON válido de uma string, lidando com blocos de código markdown.
    """
    if not text:
        return None

    # Prioriza blocos de código JSON explícitos
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse JSON from markdown block: {e}. Content: {sanitize_text_for_logging(json_str)}")
            # Continua para tentar extrair da string completa

    # Tenta encontrar o primeiro JSON aninhado
    json_starts = [i for i, char in enumerate(text) if char in ('{', '[')]
    for start_index in json_starts:
        balance = 0
        open_char, close_char = ('{', '}') if text[start_index] == '{' else ('[', ']')

        for end_index in range(start_index, len(text)):
            if text[end_index] == open_char:
                balance += 1
            elif text[end_index] == close_char:
                balance -= 1

            if balance == 0:
                potential_json_str = text[start_index: end_index + 1]
                try:
                    parsed_json = json.loads(potential_json_str)
                    logger.debug(f"Successfully extracted JSON: {sanitize_text_for_logging(potential_json_str)}")
                    return parsed_json
                except json.JSONDecodeError:
                    continue  # Continua procurando por um JSON válido maior

    logger.warning(f"Could not extract any valid JSON from text: {sanitize_text_for_logging(text)}")
    return None


# --- Pydantic Model Utilities ---

def pydantic_to_json_str(model_instance: BaseModel, indent: int = 2, exclude_none: bool = True) -> str:
    """Converte uma instância de modelo Pydantic para uma string JSON formatada."""
    return model_instance.model_dump_json(indent=indent, exclude_none=exclude_none)


def parse_llm_json_output(
        json_str: Optional[str],
        pydantic_model: Type[BaseModel],
        strict: bool = False
) -> Optional[BaseModel]:
    """
    Analisa a saída JSON de um LLM de forma robusta e a valida contra um modelo Pydantic.
    Alinhado com o princípio V3: "Módulos como Geradores de Sinal".
    """
    if not json_str:
        logger.warning(f"Received empty JSON string for model {pydantic_model.__name__}")
        return None

    parsed_dict = None
    try:
        parsed_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        if strict:
            logger.error(
                f"STRICT MODE: Failed to parse JSON for {pydantic_model.__name__}: {e}. Raw: {sanitize_text_for_logging(json_str)}")
            return None

        logger.warning(f"Direct JSON parsing failed for {pydantic_model.__name__}, attempting extraction. Error: {e}")
        extracted = extract_json_from_text(json_str)
        if isinstance(extracted, dict):
            parsed_dict = extracted
        else:
            logger.error(f"Could not extract valid JSON dict for {pydantic_model.__name__} from text.")
            return None

    if parsed_dict is None:
        return None

    try:
        model_instance = pydantic_model(**parsed_dict)
        return model_instance
    except ValidationError as e_val:
        logger.error(f"Pydantic validation error for {pydantic_model.__name__}: {e_val}. Parsed Dict: {parsed_dict}")
        return None
    except Exception as e_inst:
        logger.error(f"Unexpected error instantiating {pydantic_model.__name__}: {e_inst}. Parsed Dict: {parsed_dict}")
        return None


# --- Tool Output Formatting ---

def create_successful_tool_response(data: Optional[Dict[str, Any]] = None, message: Optional[str] = None) -> Dict[
    str, Any]:
    """Cria uma resposta de sucesso padronizada para ferramentas, alinhada com a V3."""
    response = {"status": "success"}
    if message:
        response["message"] = message
    # Aninha os dados para uma estrutura mais clara
    response["data"] = data if data is not None else {}
    return response


def create_error_tool_response(error_message: str, details: Optional[Any] = None, error_code: Optional[str] = None) -> \
Dict[str, Any]:
    """Cria uma resposta de erro padronizada para ferramentas."""
    response = {"status": "error", "error_message": error_message}
    if details:
        response["details"] = str(details)  # Garante que os detalhes sejam serializáveis
    if error_code:
        response["error_code"] = error_code
    return response