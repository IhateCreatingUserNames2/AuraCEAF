# ceaf_core/modules/embodiment_module.py
import time
from ceaf_core.genlang_types import VirtualBodyState
import logging

logger = logging.getLogger("CEAFv3_Embodiment")


class EmbodimentModule:
    """
    Gerencia o "corpo virtual" do agente, rastreando fadiga e saturação.
    Versão V2: Cálculos mais responsivos e realistas.
    """

    def update_body_state(
            self,
            body_state: VirtualBodyState,
            metrics: dict
    ) -> VirtualBodyState:
        """
        Atualiza o estado corporal virtual com base nas métricas do turno.

        Args:
            body_state: Estado atual do corpo virtual
            metrics: Métricas do turno (incluindo cognitive_strain)

        Returns:
            VirtualBodyState atualizado
        """
        # Cria uma cópia para não modificar o original diretamente
        updated_state = body_state.copy(deep=True)

        # --- 1. ATUALIZAÇÃO DA FADIGA COGNITIVA ---
        strain = metrics.get("cognitive_strain", 0.0)

        # MELHORIA: Multiplicador aumentado de 0.2 para 0.3
        # Isso faz a fadiga acumular mais rapidamente
        fatigue_increase = strain * 0.3
        updated_state.cognitive_fatigue += fatigue_increase

        logger.debug(
            f"Embodiment: Strain {strain:.2f} → "
            f"Fadiga aumentou de {body_state.cognitive_fatigue:.2f} "
            f"para {updated_state.cognitive_fatigue:.2f} (+{fatigue_increase:.2f})"
        )

        # --- 2. ATUALIZAÇÃO DA SATURAÇÃO DE INFORMAÇÃO ---
        new_memories = metrics.get("new_memories_created", 0)

        # MELHORIA: Aumentado de 0.05 para 0.08 por memória
        saturation_increase = new_memories * 0.08
        updated_state.information_saturation += saturation_increase

        if new_memories > 0:
            logger.debug(
                f"Embodiment: {new_memories} novas memórias → "
                f"Saturação aumentou de {body_state.information_saturation:.2f} "
                f"para {updated_state.information_saturation:.2f} (+{saturation_increase:.2f})"
            )

        # --- 3. RECUPERAÇÃO PASSIVA (DECAY) ---
        # Calcula o tempo desde a última atualização
        time_delta_seconds = time.time() - updated_state.last_updated
        time_delta_hours = time_delta_seconds / 3600

        # MELHORIA: Recuperação agora é mais lenta e realista
        # Fadiga recupera 3% por hora (era 5%)
        fatigue_recovery = 0.03 * time_delta_hours
        updated_state.cognitive_fatigue -= fatigue_recovery

        # Saturação recupera 1.5% por hora (era 2%)
        saturation_recovery = 0.015 * time_delta_hours
        updated_state.information_saturation -= saturation_recovery

        if time_delta_hours > 0.01:  # Log apenas se passou tempo significativo
            logger.debug(
                f"Embodiment: Recuperação passiva após {time_delta_hours:.2f}h → "
                f"Fadiga -{fatigue_recovery:.3f}, Saturação -{saturation_recovery:.3f}"
            )

        # --- 4. NORMALIZAÇÃO E VALIDAÇÃO ---
        # Garante que os valores fiquem entre 0.0 e 1.0
        updated_state.cognitive_fatigue = max(0.0, min(1.0, updated_state.cognitive_fatigue))
        updated_state.information_saturation = max(0.0, min(1.0, updated_state.information_saturation))

        # Atualiza o timestamp
        updated_state.last_updated = time.time()

        # --- 5. LOGGING DE ALERTAS ---
        # Alerta se a fadiga estiver crítica
        if updated_state.cognitive_fatigue > 0.8:
            logger.warning(
                f"⚠️ EMBODIMENT ALERT: Fadiga cognitiva CRÍTICA ({updated_state.cognitive_fatigue:.2f})! "
                f"O agente precisa de 'descanso' (redução de carga cognitiva)."
            )
        elif updated_state.cognitive_fatigue > 0.6:
            logger.info(
                f"Embodiment: Fadiga cognitiva ALTA ({updated_state.cognitive_fatigue:.2f}). "
                f"Performance pode começar a degradar."
            )

        # Alerta se a saturação estiver crítica
        if updated_state.information_saturation > 0.8:
            logger.warning(
                f"⚠️ EMBODIMENT ALERT: Saturação de informação CRÍTICA ({updated_state.information_saturation:.2f})! "
                f"O agente precisa consolidar memórias."
            )

        logger.info(
            f"Embodiment: Estado atualizado → "
            f"Fadiga: {updated_state.cognitive_fatigue:.3f} | "
            f"Saturação: {updated_state.information_saturation:.3f}"
        )

        return updated_state


