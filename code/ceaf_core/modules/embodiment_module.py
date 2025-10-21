# ceaf_core/modules/embodiment_module.py
import time
from ceaf_core.genlang_types import VirtualBodyState


class EmbodimentModule:
    def update_body_state(self, body_state: VirtualBodyState, metrics: dict) -> VirtualBodyState:
        # Fadiga aumenta com esforço (usando o output da interocepção!)
        strain = metrics.get("cognitive_strain", 0.0)
        body_state.cognitive_fatigue += strain * 0.1

        # Saturação aumenta com novas memórias
        new_memories_count = metrics.get("new_memories_created", 0)
        body_state.information_saturation += new_memories_count * 0.05

        # Recuperação passiva com o tempo (sono)
        time_delta_hours = (time.time() - body_state.last_updated) / 3600
        body_state.cognitive_fatigue -= 0.05 * time_delta_hours
        body_state.information_saturation -= 0.02 * time_delta_hours

        # Normalizar
        body_state.cognitive_fatigue = max(0.0, min(1.0, body_state.cognitive_fatigue))
        body_state.information_saturation = max(0.0, min(1.0, body_state.information_saturation))

        body_state.last_updated = time.time()
        return body_state
