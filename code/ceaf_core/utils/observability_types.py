# ceaf_core/utils/observability_types.py
import asyncio
import logging
import time
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
logger = logging.getLogger("AgencyModule_V3.2")

# --- Enums para Tipos de Observação ---
class ObservationType(Enum):
    """Tipos de eventos observados durante o ciclo de um turno."""

    # LLM & Raciocínio
    LLM_CALL_SENT = "llm_call_sent"
    LLM_RESPONSE_RECEIVED = "llm_response_received"
    LLM_RESPONSE_PARSE_ERROR = "llm_parse_error"
    AGENCY_CANDIDATE_GENERATED = "agency_candidate_generated"
    AGENCY_CANDIDATE_EVALUATED = "agency_candidate_evaluated"

    # Ferramentas
    TOOL_CALL_ATTEMPTED = "tool_call_attempted"
    TOOL_CALL_SUCCEEDED = "tool_call_succeeded"
    TOOL_CALL_FAILED = "tool_call_failed"
    TOOL_SIMULATION_MISMATCH = "tool_simulation_mismatch"
    # Governança
    VRE_ASSESSMENT_RECEIVED = "vre_assessment_received"
    VRE_REFINEMENT_TRIGGERED = "vre_refinement_triggered"

    # Memória
    MBS_SEARCH_INITIATED = "mbs_search_initiated"
    MBS_SEARCH_COMPLETED = "mbs_search_completed"
    MBS_COMMIT_ATTEMPTED = "mbs_commit_attempted"

    # NCIM
    NCIM_COHERENCE_CHECK_FAILED = "ncim_coherence_check_failed"

    # Diversos
    CUSTOM_HEURISTIC_FLAG = "custom_heuristic_flag"


class Observation(BaseModel):
    """Estrutura para uma única observação de turno."""

    observation_id: str = Field(default_factory=lambda: f"obs_{time.time_ns()}")
    timestamp: float = Field(default_factory=time.time)
    observation_type: ObservationType

    # Dados específicos (ex: {model: 'gpt-4o', tokens: 100})
    data: Dict[str, Any] = Field(default_factory=dict)

    # Referência ao objeto (opcional, para evitar serialização completa)
    target_id: Optional[str] = None

    class Config:
        use_enum_values = True


# --- Gerenciador de Observações ---

class ObservabilityManager:
    """Coleta e gerencia observações para um único turno (thread-safe na prática asyncio)."""

    def __init__(self, turn_id: str):
        self.turn_id = turn_id
        self._observations: List[Observation] = []
        self._lock = asyncio.Lock()  # Para thread-safety se a tarefa for compartilhada

    async def add_observation(self, observation_type: ObservationType, data: Optional[Dict[str, Any]] = None,
                              target_id: Optional[str] = None):
        """Adiciona uma nova observação à lista."""
        observation = Observation(
            observation_type=observation_type,
            data=data or {},
            target_id=target_id
        )
        async with self._lock:
            self._observations.append(observation)

        logger.debug(f"Observer: Added {observation_type.value} for turn {self.turn_id}.")

    async def get_observations(self) -> List[Observation]:
        """Retorna todas as observações registradas para este turno."""
        async with self._lock:
            return self._observations.copy()

    def get_observations_sync(self) -> List[Observation]:
        """Versão síncrona para módulos que não são asyncio-aware (como o SelfStateAnalyzer)."""
        return self._observations.copy()

    def clear(self):
        """Limpa as observações (útil para reutilização, mas não é o padrão aqui)."""
        self._observations = []