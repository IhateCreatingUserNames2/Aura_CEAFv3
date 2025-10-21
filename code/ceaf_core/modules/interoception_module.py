# ceaf_core/modules/interoception_module.py
from ceaf_core.genlang_types import TurnMetrics, InternalStateReport


class ComputationalInteroception:
    def generate_internal_state_report(self, metrics: dict) -> InternalStateReport:
        strain = 0.0
        if metrics.get("agency_score", 0.0) > 5.0: strain += 0.4
        if metrics.get("used_mycelial_path"): strain += 0.3
        if metrics.get("vre_rejection_count", 0) > 0: strain += 0.5

        flow = 0.0
        if metrics.get("final_confidence", 0.0) > 0.85 and strain < 0.3: flow += 0.7

        discomfort = 0.0
        discomfort += (1.0 - metrics.get("final_confidence", 1.0)) * 0.8

        tension = 0.0
        if any("ética" in flag.lower() for flag in metrics.get("vre_flags", [])): tension += 0.8

        return InternalStateReport(
            cognitive_strain=min(1.0, strain),
            cognitive_flow=min(1.0, flow),
            epistemic_discomfort=min(1.0, discomfort),
            ethical_tension=min(1.0, tension)
        )