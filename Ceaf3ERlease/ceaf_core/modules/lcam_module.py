# ceaf_core/modules/lcam_module.py
import logging
from typing import Dict, Any, Optional, List, Tuple
import re

from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
import re
if TYPE_CHECKING:
    from ceaf_core.agency_module import WinningStrategy
from ceaf_core.genlang_types import CognitiveStatePacket
from ceaf_core.utils.embedding_utils import compute_adaptive_similarity
import numpy as np
import time
from ceaf_core.genlang_types import RefinementPacket
from ceaf_core.services.mbs_memory_service import MBSMemoryService
from ceaf_core.modules.memory_blossom.memory_types import ReasoningMemory, ReasoningStep, ExplicitMemory, ExplicitMemoryContent, MemorySourceType, MemorySalience
from ceaf_core.modules.vre_engine.vre_engine import EthicalAssessment
# NOVO: Importar utilitários de embedding
from ceaf_core.utils.embedding_utils import get_embedding_client, compute_adaptive_similarity

logger = logging.getLogger("CEAFv3_LCAM")


class LCAMModule:
    """
    Loss Cataloging and Analysis Module (V3).
    Identifica interações de 'falha' e cria memórias sobre elas para aprendizado futuro.
    """

    def __init__(self, memory_service: MBSMemoryService):
        self.memory = memory_service
        # NOVO: Cliente de embedding para busca semântica
        self.embedding_client = get_embedding_client()
        logger.info("LCAMModule (V3) inicializado.")

    def predict_turn_outcome(self, cognitive_state: CognitiveStatePacket, mcl_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gera uma previsão sobre o resultado do turno, agindo como o sistema de expectativa do cérebro.
        Esta é a "Expectativa (B)" no modelo GCSL-NF.
        """
        logger.info("LCAM Prediction: Gerando previsão do resultado do turno...")

        agency_score = mcl_params.get("mcl_analysis", {}).get("agency_score", 5.0)
        coherence_bias = mcl_params.get("biases", {}).get("coherence_bias", 0.7)

        # --- Fatores que Aumentam a Expectativa de Sucesso ---

        # 1. Complexidade Percebida: Baixo Agency Score sugere uma tarefa fácil.
        complexity_factor = 1.0 - (min(agency_score, 10.0) / 10.0)  # Mapeia 0-10 para 1.0-0.0

        # 2. Coerência do Contexto: Quão "focado" é o contexto de memória?
        # Mede a similaridade média entre as memórias ativadas. Um contexto coeso é mais fácil de processar.
        coherence_factor = 0.5  # Default
        memory_vectors = [np.array(vec.vector) for vec in cognitive_state.relevant_memory_vectors if vec.vector]
        if len(memory_vectors) > 1:
            similarities = []
            for i in range(len(memory_vectors)):
                for j in range(i + 1, len(memory_vectors)):
                    sim = compute_adaptive_similarity(memory_vectors[i].tolist(), memory_vectors[j].tolist())
                    similarities.append(sim)
            if similarities:
                coherence_factor = np.mean(similarities)

        # 3. Alinhamento com a Persona: O viés do MCL está alinhado com a dificuldade da tarefa?
        # Se a tarefa é complexa (agency > 5) e o MCL está em modo de coerência (bias > 0.7),
        # a chance de falha é maior (desalinhamento).
        alignment_factor = 1.0
        if agency_score > 5.0 and coherence_bias > 0.7:
            alignment_factor = 0.8  # Penalidade por rigidez em tarefa complexa
        elif agency_score < 3.0 and coherence_bias < 0.5:
            alignment_factor = 0.9  # Penalidade por novidade excessiva em tarefa simples

        # --- Cálculo Final da Expectativa ---
        # Combina os fatores com pesos. A complexidade é o fator mais importante.
        expected_confidence = (complexity_factor * 0.5) + (coherence_factor * 0.3) + (alignment_factor * 0.2)

        prediction = {
            "expected_final_confidence": max(0.0, min(1.0, expected_confidence)),
            "expected_vre_rejections": 0,  # O agente sempre espera ter sucesso eticamente.
            "prediction_timestamp": time.time()
        }

        logger.info(
            f"LCAM Prediction: Confiança final esperada: {prediction['expected_final_confidence']:.2f} (Complexidade: {complexity_factor:.2f}, Coerência: {coherence_factor:.2f})")

        return prediction

    async def analyze_and_catalog_loss(self,
                                       turn_prediction: Dict[str, Any],
                                       turn_metrics: Dict[str, Any],
                                       cognitive_state: CognitiveStatePacket,
                                       winning_strategy: 'WinningStrategy',
                                       final_response: str
                                       ):
        """
        Versão 2.1 (GCSL-NF): Aprende tanto com o que deu certo (feedback positivo)
        quanto com o que deu errado (feedback negativo).
        """

        # --- CÁLCULO DO ERRO DE PREDIÇÃO (Função de Distância) ---
        expected_confidence = turn_prediction.get("expected_final_confidence", 0.5)
        actual_confidence = turn_metrics.get("final_confidence", 0.5)
        confidence_error = actual_confidence - expected_confidence

        # Rejeições do VRE (se ativo) ou do feedback do usuário
        rejection_count = turn_metrics.get("vre_rejection_count", 0) + turn_metrics.get("user_feedback_rejections", 0)
        rejection_error = -1.0 * rejection_count

        prediction_error_signal = (confidence_error * 0.4) + (rejection_error * 0.6)

        # --- APRENDIZADO DUPLO ---

        # 1. FEEDBACK POSITIVO (Aprendendo o Caminho para o Resultado Real C)
        # Sempre criamos uma memória de raciocínio para documentar o que aconteceu.
        await self._create_positive_feedback_memory(
            initial_state=cognitive_state,
            trajectory=winning_strategy,
            actual_outcome_metrics=turn_metrics,
            final_response=final_response
        )

        # 2. FEEDBACK NEGATIVO (Aprendendo que a Trajetória T foi ruim para a Expectativa B)
        if prediction_error_signal < -0.25:  # Limiar de "falha surpreendente"
            logger.critical(
                f"LCAM (Negative Feedback): Falha inesperada detectada! Sinal: {prediction_error_signal:.2f}")

            await self._create_negative_feedback_memory(
                initial_state=cognitive_state,
                trajectory=winning_strategy,
                expected_outcome=turn_prediction,
                actual_outcome_metrics=turn_metrics,
                prediction_error=prediction_error_signal
            )

        # 3. FEEDBACK DE REFORÇO (Sucesso inesperado)
        elif prediction_error_signal > 0.25:  # Limiar de "sucesso surpreendente"
            logger.critical(
                f"LCAM (Positive Reinforcement): Sucesso inesperado detectado! Sinal: {prediction_error_signal:.2f}")
            # No futuro, podemos criar uma memória específica para reforçar essa trajetória de sucesso.
            # Por enquanto, a memória de raciocínio positiva já cumpre esse papel.
        else:
            logger.info(
                f"LCAM: Resultado do turno estava dentro do esperado. Sinal: {prediction_error_signal:.2f}. Sem aprendizado forte.")

    async def _create_positive_feedback_memory(self, initial_state: CognitiveStatePacket, trajectory: 'WinningStrategy',
                                               actual_outcome_metrics: Dict, final_response: str):
        """
        Cria uma memória de raciocínio (ReasoningMemory) que documenta a trajetória e o resultado real.
        Este é o aprendizado sobre "como chegar ao resultado C".
        """
        outcome_status = "success" if actual_outcome_metrics.get("vre_rejection_count",
                                                                 0) == 0 and actual_outcome_metrics.get(
            "user_feedback_rejections", 0) == 0 and actual_outcome_metrics.get("final_confidence",
                                                                               0) > 0.4 else "failure"

        reasoning_mem = ReasoningMemory(
            task_description=initial_state.original_intent.query_vector.source_text,
            strategy_summary=trajectory.strategy_description or f"Executou a ferramenta: {trajectory.tool_call_request}",
            reasoning_steps=[
                ReasoningStep(step_number=1, description="Estratégia Selecionada", reasoning=trajectory.reasoning)],
            outcome=outcome_status,
            outcome_reasoning=f"Confiança final alcançada: {actual_outcome_metrics.get('final_confidence', 0):.2f}. Rejeições: {actual_outcome_metrics.get('vre_rejection_count', 0) + actual_outcome_metrics.get('user_feedback_rejections', 0)}",
            source_type=MemorySourceType.INTERNAL_REFLECTION,
            source_turn_id=actual_outcome_metrics.get("turn_id"),
            salience=MemorySalience.MEDIUM
        )
        await self.memory.add_specific_memory(reasoning_mem)
        logger.info(
            f"LCAM (Positive Feedback): Memória de raciocínio {reasoning_mem.memory_id} criada para o resultado alcançado.")

    async def _create_negative_feedback_memory(self, initial_state: CognitiveStatePacket, trajectory: 'WinningStrategy',
                                               expected_outcome: Dict, actual_outcome_metrics: Dict,
                                               prediction_error: float):
        """
        Cria uma memória de falha explícita (ExplicitMemory) que documenta a trajetória ineficaz.
        Este é o aprendizado sobre "o que NÃO fazer para alcançar a expectativa B".
        """
        loss_content_text = f"""
        Lição Aprendida (Falha de Predição):
        - Tarefa: "{initial_state.original_intent.query_vector.source_text}"
        - Estratégia Usada: "{trajectory.strategy_description or f"Executou a ferramenta: {trajectory.tool_call_request}"}"
        - Resultado Esperado (Confiança): {expected_outcome.get('expected_final_confidence', 0.5):.2f}
        - Resultado Real (Confiança): {actual_outcome_metrics.get('final_confidence', 0.5):.2f}
        - Insight (Erro de Predição): {prediction_error:.2f}. Esta trajetória de pensamento foi ineficaz para alcançar o resultado esperado e deve ser abordada com cautela em contextos futuros.
        """

        loss_memory = ExplicitMemory(
            content=ExplicitMemoryContent(text_content=loss_content_text),
            memory_type="explicit",
            source_type=MemorySourceType.INTERNAL_REFLECTION,
            salience=MemorySalience.HIGH,
            keywords=["failure", "prediction_error", "negative_feedback", "avoid_path", "lcam_lesson"],
            failure_pattern="prediction_error",
            source_turn_id=actual_outcome_metrics.get("turn_id"),
            learning_value=abs(prediction_error)  # O valor do aprendizado é a magnitude da surpresa
        )
        await self.memory.add_specific_memory(loss_memory)
        logger.critical(
            f"LCAM (Negative Feedback): Memória de falha {loss_memory.memory_id} criada para evitar esta trajetória ineficaz.")

    # --- NOVA FUNÇÃO: Ferramenta de Consulta de Falhas ---
    async def get_insights_on_potential_failure(
            self,
            current_query: str,
            similarity_threshold: float = 0.80
    ) -> Optional[Dict[str, Any]]:
        """
        Busca no MBS por memórias de falhas semanticamente similares à query atual.
        Retorna um "insight de cautela" se uma falha similar for encontrada.
        """
        logger.info(f"LCAM: Verificando falhas passadas similares a '{current_query[:50]}...'")

        # 1. Cria uma query de busca específica para memórias de falha
        lcam_search_query = f"falha erro lição_aprendida {current_query}"

        # 2. Busca no MBS por memórias relevantes
        # Usamos search_raw_memories para obter os objetos de memória completos
        potential_failures = await self.memory.search_raw_memories(lcam_search_query, top_k=3)

        if not potential_failures:
            logger.info("LCAM: Nenhuma memória de falha relevante encontrada.")
            return None

        # 3. Gera embedding para a query atual para comparação precisa
        try:
            query_embedding = await self.embedding_client.get_embedding(current_query, context_type="default_query")
        except Exception as e:
            logger.error(f"LCAM: Falha ao gerar embedding para a query atual: {e}")
            return None

        # 4. Compara a similaridade e encontra a melhor correspondência
        best_match: Optional[Tuple[ExplicitMemory, float]] = None

        for mem_obj, score in potential_failures:
            # Apenas considera memórias que são explicitamente de falhas
            if "falha" not in mem_obj.keywords and "erro" not in mem_obj.keywords:
                continue

            # Obtém o embedding da memória de falha
            mem_embedding = self.memory._embedding_cache.get(mem_obj.memory_id)
            if not mem_embedding:
                continue

            # Compara a query ATUAL com o conteúdo da memória de falha
            similarity = compute_adaptive_similarity(query_embedding, mem_embedding)

            if similarity > similarity_threshold:
                if best_match is None or similarity > best_match[1]:
                    best_match = (mem_obj, similarity)

        # 5. Se uma correspondência forte for encontrada, gera o insight
        if best_match:
            matched_memory, match_similarity = best_match

            # Extrai a razão da falha da memória antiga
            failure_reason = "Razão não especificada."
            if matched_memory.content and matched_memory.content.text_content:
                match = re.search(r"Motivo da Falha:\s*(.*)", matched_memory.content.text_content, re.IGNORECASE)
                if match:
                    failure_reason = match.group(1).strip()

            insight = {
                "warning_level": "high" if match_similarity > 0.9 else "medium",
                "message": f"Cuidado: A situação atual é {match_similarity:.0%} similar a uma falha passada.",
                "past_failure_reason": failure_reason,
                "past_failure_memory_id": matched_memory.memory_id,
                "recommendation": "Proceda com cautela extra. Aumente a revisão ética (VRE) e a humildade epistêmica."
            }
            logger.warning(f"LCAM: ALERTA DE FALHA POTENCIAL. Insight gerado: {insight['message']}")
            return insight

        logger.info("LCAM: Nenhuma falha passada encontrada acima do limiar de similaridade.")
        return None