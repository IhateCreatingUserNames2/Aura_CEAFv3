# ceaf_core/genlang_types.py
import time

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from collections import defaultdict, Counter
from datetime import datetime


class TurnMetrics(BaseModel):
    turn_id: str
    agency_score: float = 0.0
    used_mycelial_path: bool = False
    vre_rejection_count: int = 0
    vre_flags: List[str] = Field(default_factory=list)
    final_confidence: float = 0.0
    relevant_memories_count: int = 0

class VirtualBodyState(BaseModel):
    cognitive_fatigue: float = Field(0.0, ge=0.0, le=1.0)
    information_saturation: float = Field(0.0, ge=0.0, le=1.0)
    last_updated: float = Field(default_factory=time.time)

class UserRepresentation(BaseModel):
    """Modelo Pydantic para a representação interna do usuário."""
    emotional_state: str = Field("neutral", description="O estado emocional inferido do usuário (ex: curious, frustrated, happy).")
    communication_style: str = Field("neutral", description="O estilo de comunicação do usuário (ex: direct, formal, informal, detailed).")
    known_preferences: List[str] = Field(default_factory=list, description="Preferências explícitas ou inferidas do usuário (ex: 'prefers concise answers', 'interested in philosophy').")
    knowledge_level: str = Field("unknown", description="Nível de conhecimento inferido sobre o tópico atual (ex: beginner, expert).")
    last_update_reason: str = "Initial model."

class MotivationalDrives(BaseModel):
    curiosity: float = Field(0.5, ge=0.0, le=1.0, description="Desejo de informação nova.")
    mastery: float = Field(0.5, ge=0.0, le=1.0, description="Desejo de melhorar em áreas fracas.")
    connection: float = Field(0.5, ge=0.0, le=1.0, description="Desejo de manter um bom relacionamento.")
    consistency: float = Field(0.5, ge=0.0, le=1.0, description="Desejo de manter a identidade coerente.")
    last_updated: float = Field(default_factory=time.time)

# Representa o "estado sentido" consolidado
class InternalStateReport(BaseModel):
    cognitive_strain: float = Field(0.0, description="Esforço mental percebido.")
    cognitive_flow: float = Field(0.0, description="Sensação de alinhamento e facilidade.")
    epistemic_discomfort: float = Field(0.0, description="Desconforto com a incerteza.")
    ethical_tension: float = Field(0.0, description="Tensão por conflitos éticos.")
    social_resonance: float = Field(0.0, description="Sensação de conexão com o usuário.")
    timestamp: datetime = Field(default_factory=datetime.now)


class GenlangVector(BaseModel):
    """Representação padronizada de um vetor semântico. É mais do que uma lista de floats; carrega seu próprio contexto."""
    vector: List[float]
    source_text: Optional[str] = None
    model_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IntentPacket(BaseModel):
    """Sinal gerado a partir da entrada do usuário."""
    query_vector: GenlangVector
    # Por enquanto, podemos deixar os outros vetores como opcionais ou com valores padrão para simplificar.
    intent_vector: Optional[GenlangVector] = None # Ex: factual, criativo, reflexivo
    emotional_valence_vector: Optional[GenlangVector] = None
    entity_vectors: List[GenlangVector] = Field(default_factory=list)
    metadata: Dict[str, Any]

class AdjustmentVector(BaseModel):
    """Representa uma direção semântica para ajustar uma resposta."""
    vector: GenlangVector
    description: str  # Ex: "Adicionar mais humildade epistêmica", "Remover linguagem agressiva"
    weight: float = Field(0.5, ge=0.0, le=1.0) # A importância deste ajuste

class RefinementPacket(BaseModel):
    """
    Contém as instruções do VRE para refinar um ResponsePacket.
    Pode estar vazio se nenhum refinamento for necessário.
    """
    adjustment_vectors: List[AdjustmentVector] = Field(default_factory=list)
    textual_recommendations: List[str] = Field(default_factory=list)

class GuidancePacket(BaseModel):
    """
    Contém os múltiplos vetores de orientação emitidos pelo MCL.
    Cada vetor "puxa" o raciocínio em uma direção específica.
    """
    coherence_vector: GenlangVector  # Aponta para o centro do contexto atual (ordem)
    novelty_vector: GenlangVector  # Aponta para longe do contexto atual (caos/exploração)

    # NOVOS VETORES DE ORIENTAÇÃO AVANÇADA
    goal_alignment_vector: Optional[GenlangVector] = None  # Aponta na direção de um objetivo ativo
    safety_avoidance_vector: Optional[GenlangVector] = None  # Um "anti-vetor" que representa conceitos a serem evitados


class CommonGroundTracker(BaseModel):
    """Rastreia o que já foi estabelecido na conversa para evitar repetições."""

    # --- MUDANÇA 1: Usar um dicionário padrão ---
    agent_statement_counts: Dict[str, int] = Field(default_factory=dict)

    user_acknowledged_topics: List[str] = Field(default_factory=list)

    def record_agent_statement(self, statement_type: str):
        """Registra que o agente fez um tipo específico de declaração."""
        # --- MUDANÇA 2: Lógica segura para dicionário padrão ---
        current_count = self.agent_statement_counts.get(statement_type, 0)
        self.agent_statement_counts[statement_type] = current_count + 1

    def check_statement_count(self, statement_type: str) -> int:
        """Verifica quantas vezes um tipo de declaração foi feito."""
        return self.agent_statement_counts.get(statement_type, 0)

    def is_becoming_repetitive(self, statement_type: str, threshold: int = 2) -> bool:
        """Verifica se fazer esta declaração novamente seria repetitivo."""
        return self.check_statement_count(statement_type) >= threshold


class ToolOutputPacket(BaseModel):
    """
    Encapsula o resultado da execução de uma ferramenta como um sinal Genlang.
    """
    tool_name: str
    status: Literal["success", "error"]
    summary_vector: GenlangVector  # O embedding do resultado textual
    raw_output: Optional[str] = None  # A saída de texto original, útil para logging e depuração


class CognitiveStatePacket(BaseModel):
    """O estado cognitivo que flui entre os módulos internos."""
    original_intent: IntentPacket
    relevant_memory_vectors: List[GenlangVector] = Field(default_factory=list)
    identity_vector: GenlangVector
    guidance_packet: GuidancePacket
    tool_outputs: List[ToolOutputPacket] = Field(default_factory=list)
    common_ground: CommonGroundTracker = Field(default_factory=CommonGroundTracker)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    ethical_assessment_summary: Optional[str] = None
    deliberation_history: List[str] = Field(default_factory=list)

class ResponsePacket(BaseModel):
    """O sinal final, pronto para ser traduzido para o usuário."""
    content_summary: str # O texto da resposta gerado pela lógica antiga
    response_emotional_tone: str = "neutral"
    confidence_score: float = 0.8 # Score padrão por enquanto
    supporting_memory_snippets: List[str] = Field(default_factory=list)
    ethical_assessment_summary: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


