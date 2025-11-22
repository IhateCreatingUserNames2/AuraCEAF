# ceaf_core/modules/embodiment_module.py

import time
import logging
from ceaf_core.genlang_types import VirtualBodyState
from ceaf_core.models import BodyConfig  # <--- IMPORTAR BodyConfig

logger = logging.getLogger("CEAFv3_Embodiment")


class EmbodimentModule:
    """
    Gerencia o "corpo virtual" do agente, rastreando fadiga e saturação.
    Versão V2.1 (Configurable): Parâmetros biológicos customizáveis.
    """

    # --- ALTERAÇÃO NO INIT ---
    def __init__(self, config: BodyConfig = None):
        self.config = config or BodyConfig()

    def update_body_state(
            self,
            body_state: VirtualBodyState,
            metrics: dict
    ) -> VirtualBodyState:
        """
        Atualiza o estado corporal virtual com base nas métricas do turno.
        """
        # Cria uma cópia para não modificar o original diretamente
        updated_state = body_state.copy(deep=True)

        # --- 1. ATUALIZAÇÃO DA FADIGA COGNITIVA ---
        strain = metrics.get("cognitive_strain", 0.0)

        # USAR CONFIG
        fatigue_increase = strain * self.config.fatigue_accumulation_multiplier
        updated_state.cognitive_fatigue += fatigue_increase

        logger.debug(
            f"Embodiment: Strain {strain:.2f} -> Fadiga +{fatigue_increase:.2f}"
        )

        # --- 2. ATUALIZAÇÃO DA SATURAÇÃO DE INFORMAÇÃO ---
        new_memories = metrics.get("new_memories_created", 0)

        # USAR CONFIG
        saturation_increase = new_memories * self.config.saturation_accumulation_per_memory
        updated_state.information_saturation += saturation_increase

        if new_memories > 0:
            logger.debug(
                f"Embodiment: {new_memories} memórias -> Saturação +{saturation_increase:.2f}"
            )

        # --- 3. RECUPERAÇÃO PASSIVA (DECAY) ---
        time_delta_seconds = time.time() - updated_state.last_updated
        time_delta_hours = time_delta_seconds / 3600

        # USAR CONFIG
        fatigue_recovery = self.config.fatigue_recovery_rate * time_delta_hours
        updated_state.cognitive_fatigue -= fatigue_recovery

        saturation_recovery = self.config.saturation_recovery_rate * time_delta_hours
        updated_state.information_saturation -= saturation_recovery

        # ... (logging e normalização mantidos) ...

        # Normalização
        updated_state.cognitive_fatigue = max(0.0, min(1.0, updated_state.cognitive_fatigue))
        updated_state.information_saturation = max(0.0, min(1.0, updated_state.information_saturation))
        updated_state.last_updated = time.time()

        # Alertas usando Config
        if updated_state.cognitive_fatigue > self.config.fatigue_warning_threshold:
            logger.warning(f"⚠️ EMBODIMENT ALERT: Fadiga cognitiva CRÍTICA ({updated_state.cognitive_fatigue:.2f})!")

        return updated_state