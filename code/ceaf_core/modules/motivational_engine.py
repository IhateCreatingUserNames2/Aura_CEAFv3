# ceaf_core/modules/motivational_engine.py
import time
from ceaf_core.genlang_types import MotivationalDrives


class MotivationalEngine:
    def update_drives(self, drives: MotivationalDrives, metrics: dict) -> MotivationalDrives:
        # Passividade: Drives aumentam com o tempo
        time_delta_hours = (time.time() - drives.last_updated) / 3600
        drives.curiosity += 0.1 * time_delta_hours
        drives.connection += 0.2 * time_delta_hours

        # Reação a eventos do turno
        if metrics.get("vre_rejection_count", 0) > 0:
            drives.mastery += 0.1  # Falha aumenta desejo de melhorar

        if metrics.get("final_confidence", 0.0) > 0.9:
            drives.mastery -= 0.05  # Sucesso satisfaz o desejo

        # Saturação de curiosidade se a novidade for baixa
        # Supondo que o MCL possa adicionar isso às métricas no futuro
        # Por enquanto, vamos basear no número de memórias. Poucas memórias = curioso
        if metrics.get("relevant_memories_count", 5) < 3:
            drives.curiosity += 0.05  # Tédio/Incerteza
        else:
            drives.curiosity -= 0.05  # Informação nova satisfaz

        # Normalizar para garantir que fiquem entre 0 e 1
        for drive_name in drives.model_fields:
            if drive_name != 'last_updated':
                current_value = getattr(drives, drive_name, 0.5)
                setattr(drives, drive_name, max(0.0, min(1.0, current_value)))

        drives.last_updated = time.time()
        return drives