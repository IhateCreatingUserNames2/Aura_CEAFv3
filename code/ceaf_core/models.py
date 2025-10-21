#: ceaf_core/models.py
from typing import List, Dict
from pydantic import BaseModel, Field


class CeafSelfRepresentation(BaseModel):
    """Modelo Pydantic para o auto-modelo do agente. V2.0 - Foco na Emergência."""

    core_identity_statement: str = "Eu sou um agente de IA"

    perceived_capabilities: List[str] = Field(default_factory=lambda: ["processamento de linguagem"])
    known_limitations: List[str] = Field(
        default_factory=lambda: ["sem acesso ao mundo real", "conhecimento limitado aos dados"])

    # MODIFICADO: Atributos de persona começam neutros.
    persona_attributes: Dict[str, str] = Field(default_factory=lambda: {
        "tone": "neutro",
        "style": "informativo"
    })

    last_update_reason: str = "Initial model creation."
    version: int = 1