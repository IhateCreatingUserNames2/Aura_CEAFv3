# NOVO ARQUIVO: ceaf_core/utils/config_utils.py
import json
import logging
from pathlib import Path
from typing import Dict
import asyncio

logger = logging.getLogger("ConfigUtils")

# --- Lógica de Configuração Dinâmica ---
DEFAULT_DYNAMIC_CONFIG = {
    "MCL": {"agency_threshold": 2.0, "state_to_params_map": {
        "STABLE_OPERATION": {"coherence_bias": 0.8, "novelty_bias": 0.2, "use_agency_simulation": False,
                             "temperature": 0.5, "reason": "Operação estável."},
        "PRODUCTIVE_CONFUSION": {"coherence_bias": 0.4, "novelty_bias": 0.6, "use_agency_simulation": True,
                                 "temperature": 0.9, "reason": "Confusão produtiva."},
        "EDGE_OF_CHAOS": {"coherence_bias": 0.9, "novelty_bias": 0.1, "use_agency_simulation": True, "temperature": 0.3,
                          "reason": "Beira do caos."},
        "EXPLORING": {"coherence_bias": 0.5, "novelty_bias": 0.5, "use_agency_simulation": True, "temperature": 0.8,
                      "reason": "Exploração."}}},
    "MBS": {"default_coherence_bias": 0.7, "default_novelty_bias": 0.3},
    "VRE": {"evaluation_threshold": 0.6},
    "MYCELIAL_MODE": {
            "enabled": False,
            "agency_threshold": 7.0
        }
}


def load_ceaf_dynamic_config(persistence_path: Path) -> Dict:
    config_file = persistence_path / "ceaf_dynamic_config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"Carregada config dinâmica específica para o agente em {config_file}")
                return config
        except (json.JSONDecodeError, IOError) as e:
            logger.error(
                f"Erro ao carregar config dinâmica do agente em {config_file}: {e}. Usando padrões e recriando.")

    new_config = DEFAULT_DYNAMIC_CONFIG.copy()
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(new_config, f, indent=2)
            logger.info(f"Criada config dinâmica padrão para o agente em {config_file}")
    except IOError as e:
        logger.error(f"FATAL: Não foi possível salvar o arquivo de configuração em {config_file}: {e}")
    return new_config


async def save_ceaf_dynamic_config(persistence_path: Path, config: Dict):
    config_file = persistence_path / "ceaf_dynamic_config.json"

    def write_op():
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    await asyncio.to_thread(write_op)
    logger.info(f"Configuração dinâmica salva em {config_file} para o agente.")