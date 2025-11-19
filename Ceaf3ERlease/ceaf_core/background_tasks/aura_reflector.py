# Em: ceaf_core/background_tasks/aura_reflector.py

import logging
import random
import re
from datetime import datetime
import json

import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx

from agent_manager import AgentManager
from database.models import AgentRepository
from .kg_processor import KGProcessor
from ceaf_core.modules.memory_blossom.memory_types import (
    GoalRecord,
    GoalStatus,
    ExplicitMemory,
    ExplicitMemoryContent,
    MemorySourceType,
    MemorySalience,
    GenerativeMemory,
    GenerativeSeed
)



from ceaf_core.services.cognitive_log_service import CognitiveLogService
from ceaf_core.system import save_ceaf_dynamic_config, load_ceaf_dynamic_config, CEAFSystem
from ceaf_core.modules.memory_blossom.advanced_synthesizer import AdvancedMemorySynthesizer, StoryArcType
from ceaf_core.services.llm_service import LLM_MODEL_SMART
from ceaf_core.system import CeafSelfRepresentation
from ..modules.ncim_engine.ncim_module import LLM_MODEL_FOR_REFLECTION
from ..utils import extract_json_from_text

logger = logging.getLogger("AuraReflector")

CONFIDENCE_THRESHOLD_FOR_SUCCESS = 0.75
MIN_TURNS_FOR_ANALYSIS = 5
PROACTIVITY_ACTIVATION_THRESHOLD = 0.45
AURA_API_BASE_URL = "http://127.0.0.1:8009/ceaf"



def analyze_correlation_guidance_confidence(turn_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analisa a correlação entre os biases de orientação (coerência vs. novidade) do MCL
    e a confiança da resposta final, para descobrir se o agente está sendo muito
    "caótico" ou muito "rígido".
    """
    results = {
        "coherence_leaning_success_rate": 0.0,
        "novelty_leaning_success_rate": 0.0,
        "coherence_turn_count": 0,
        "novelty_turn_count": 0,
        "suggestion": "insufficient_data"
    }

    coherence_successes = 0
    novelty_successes = 0

    for turn in turn_history:
        try:
            # Pega a orientação completa do MCL que foi salva no log
            mcl_guidance = turn.get("mcl_guidance")
            if not mcl_guidance:
                continue  # Pula turnos que não têm o log de orientação

            # Extrai os biases que foram REALMENTE usados naquele turno
            biases = mcl_guidance.get("biases")
            if not biases:
                continue

            coherence_bias = biases.get("coherence_bias", 0.5)
            novelty_bias = biases.get("novelty_bias", 0.5)

            # Verifica se o resultado foi um "sucesso" (confiança alta)
            is_successful = turn["response_packet"]["confidence_score"] > CONFIDENCE_THRESHOLD_FOR_SUCCESS

            # Classifica o turno como orientado a coerência ou novidade e conta os sucessos
            if coherence_bias > novelty_bias:
                results["coherence_turn_count"] += 1
                if is_successful:
                    coherence_successes += 1
            elif novelty_bias > coherence_bias:  # Usamos elif para ignorar o caso de empate
                results["novelty_turn_count"] += 1
                if is_successful:
                    novelty_successes += 1
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"AuraReflector: Pulando turno malformado durante análise de correlação: {e}")
            continue

    # Calcula as taxas de sucesso se houver dados suficientes
    if results["coherence_turn_count"] > 5:
        results["coherence_leaning_success_rate"] = coherence_successes / results["coherence_turn_count"]

    if results["novelty_turn_count"] > 5:
        results["novelty_leaning_success_rate"] = novelty_successes / results["novelty_turn_count"]

    # Gera uma sugestão de ajuste com base na comparação das taxas de sucesso
    coh_rate = results["coherence_leaning_success_rate"]
    nov_rate = results["novelty_leaning_success_rate"]

    # Apenas sugere uma mudança se houver dados para ambos os tipos de orientação e uma diferença significativa
    if coh_rate > 0 and nov_rate > 0:
        if coh_rate > nov_rate + 0.15:  # Coerência é 15% mais bem-sucedida
            results["suggestion"] = "increase_coherence_bias"
        elif nov_rate > coh_rate + 0.15:  # Novidade é 15% mais bem-sucedida
            results["suggestion"] = "increase_novelty_bias"
        else:
            results["suggestion"] = "maintain_balance"

    return results


async def generate_emergent_goals(
        agent_instance: 'CEAFSystem',
        synthesis_insight: str,
        self_model: 'CeafSelfRepresentation'
) -> Optional[GoalRecord]:
    """
    Usa um LLM para refletir sobre um insight e a identidade atual, gerando um novo objetivo.
    """
    logger.info(
        f"AURA-REFLECTOR (Goal Generation): Tentando gerar meta a partir do insight: '{synthesis_insight[:100]}...'")

    goal_generation_prompt = f"""
    Você é um módulo de planejamento estratégico para uma IA. Sua tarefa é derivar um objetivo de longo prazo acionável a partir de um insight recém-sintetizado e do estado atual da identidade da IA.

    **Contexto da Identidade Atual da IA:**
    - Versão da Identidade: {self_model.version}
    - Capacidades Percebidas (resumo): {", ".join(self_model.perceived_capabilities[-5:])}
    - Limitações Conhecidas (resumo): {", ".join(self_model.known_limitations[-5:])}
    - Valores Centrais: {self_model.core_values_summary}

    **Insight Recém-Sintetizado (Resultado de uma 'sessão de sonho'):**
    "{synthesis_insight}"

    **Sua Tarefa:**
    Com base no insight e na identidade, formule um único e novo `GoalRecord`. O objetivo deve ser:
    - **Acionável:** Algo que possa ser medido ou trabalhado.
    - **De Longo Prazo:** Não uma tarefa imediata, mas uma direção para o desenvolvimento.
    - **Alinhado:** Consistente com os valores e a identidade da IA.

    Responda APENAS com um objeto JSON válido com a seguinte estrutura. NÃO inclua o campo 'memory_id' ou 'timestamp'.

    **Exemplo de Saída JSON:**
    {{
      "memory_type": "goal_record",
      "goal_description": "Melhorar a precisão em problemas matemáticos buscando fontes confiáveis ou usando uma ferramenta de cálculo.",
      "status": "pending",
      "priority": 7,
      "motivation_level": 0.8,
      "keywords": ["desenvolvimento", "matemática", "precisão", "ferramentas"]
    }}

    **Se o insight for muito genérico ou não inspirar um objetivo claro, retorne um JSON vazio: {{}}**

    **Seu JSON de Saída:**
    """

    try:
        # Use o llm_service da instância do agente
        response_str = await agent_instance.llm_service.ainvoke(
            LLM_MODEL_SMART,
            goal_generation_prompt,
            temperature=0.3
        )

        goal_json = extract_json_from_text(response_str)

        if not goal_json or "goal_description" not in goal_json:
            logger.info("AURA-REFLECTOR (Goal Generation): Nenhum objetivo acionável foi gerado a partir do insight.")
            return None

        # Cria a instância do GoalRecord, os campos padrão serão preenchidos
        new_goal = GoalRecord(
            goal_description=goal_json["goal_description"],
            status=GoalStatus(goal_json.get("status", "pending")),
            priority=goal_json.get("priority", 7),
            motivation_level=goal_json.get("motivation_level", 0.75),
            keywords=goal_json.get("keywords", ["objetivo_emergente", "auto-aperfeiçoamento"]),
            source_type=MemorySourceType.INTERNAL_REFLECTION,
            salience=MemorySalience.HIGH  # Metas emergentes são importantes
        )

        logger.critical(
            f"AURA-REFLECTOR (Goal Generation): Nova meta emergente gerada! -> '{new_goal.goal_description}'")
        return new_goal

    except Exception as e:
        logger.error(f"AURA-REFLECTOR (Goal Generation): Erro durante a geração da meta: {e}", exc_info=True)
        return None


async def _promote_insight_to_value(agent_instance: CEAFSystem, insight_text: str) -> Optional[ExplicitMemory]:
    """Avalia se um insight pode se tornar um novo valor central."""

    promotion_prompt = f"""
    Você é um filósofo de IA analisando um insight gerado pela própria IA.

    Insight Gerado: "{insight_text}"

    Tarefa: Avalie se este insight representa um princípio de comportamento fundamental, reutilizável e de alto nível que deveria se tornar um "valor central" para a IA.
    - Um bom valor central é uma diretriz, não um fato. (Ex: "Eu devo buscar clareza" vs "Clareza é boa").
    - Deve ser aplicável em muitas situações.

    Responda APENAS com um JSON com a seguinte estrutura:
    {{
        "is_core_value": <true or false>,
        "distilled_value_statement": "<Se true, reformule o insight como uma declaração de valor concisa em primeira pessoa. Se false, null.>",
        "reasoning": "<Sua justificativa para a decisão.>"
    }}
    """

    response_str = await agent_instance.llm_service.ainvoke(
        LLM_MODEL_FOR_REFLECTION,  # Use o modelo mais inteligente para isso
        promotion_prompt,
        temperature=0.1
    )

    result_json = extract_json_from_text(response_str)

    if result_json and result_json.get("is_core_value") is True:
        distilled_value = result_json.get("distilled_value_statement")
        if distilled_value:
            logger.critical(
                f"AURA-REFLECTOR (Evolução de Valor): Insight promovido a novo VALOR CENTRAL! -> '{distilled_value}'")

            new_core_value_memory = ExplicitMemory(
                content=ExplicitMemoryContent(text_content=distilled_value),
                memory_type="explicit",
                source_type=MemorySourceType.INTERNAL_REFLECTION,
                salience=MemorySalience.CRITICAL,
                keywords=["core_value", "principle", "learned_belief", "emergent_value"],
                is_core_value=True,
                learning_value=0.8,  # Começa forte, mas não tão forte quanto os iniciais
                metadata={"derived_from_insight": insight_text[:150]}
            )
            return new_core_value_memory
    return None


async def _generate_theme_for_cluster(
        cluster_memories: List[Any],
        agent_instance: 'CEAFSystem'
) -> str:
    """Usa um LLM para extrair um tema abstrato de um cluster de memórias."""
    if not cluster_memories:
        return "conceito_indefinido"

    memory_texts = []
    for mem in cluster_memories:
        text, _ = await agent_instance.memory_service._get_searchable_text_and_keywords(mem)
        if text:
            memory_texts.append(f"- {text[:200]}")  # Limita o tamanho para o prompt

    if not memory_texts:
        return "conceito_textual_vazio"

    theme_prompt = f"""
    Analise os seguintes fragmentos de memória de uma IA. Identifique o tema ou conceito abstrato central que os conecta.

    Fragmentos de Memória:
    {chr(10).join(memory_texts)}

    Sua Tarefa:
    Responda com uma única frase curta (máximo 10 palavras) que descreva este conceito central.
    Exemplos: "a natureza da consciência", "a importância da humildade epistêmica", "estratégias para resolver problemas complexos".

    Conceito Central:
    """

    try:
        theme = await agent_instance.llm_service.ainvoke(
            LLM_MODEL_SMART,
            theme_prompt,
            temperature=0.2
        )
        return theme.strip() if theme and not theme.startswith("[LLM_ERROR]") else "tema_nao_sintetizado"
    except Exception:
        return "erro_na_sintese_do_tema"


async def perform_autonomous_clustering_and_synthesis(agent_id: str, agent_manager: AgentManager):
    """
    Realiza o "ciclo de sonho" V2 do agente. Analisa memórias recentes no espaço latente,
    sintetiza proto-memórias (centróides) e aplica esquecimento ativo às memórias originais.
    """
    logger.info(f"AURA-REFLECTOR (Latent Dream): Iniciando ciclo de consolidação para o agente {agent_id}")
    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance:
        logger.error(f"AURA-REFLECTOR (Latent Dream): Não foi possível obter a instância do agente {agent_id}.")
        return None  # Retorna None para consistência

    # 1. Obter um lote de memórias recentes para "sonhar"
    recent_memories_raw = await agent_instance.memory_service.search_raw_memories(query="*", top_k=50)

    # Filtra para incluir principalmente memórias de interação e reflexão.
    memories_to_consolidate = [
        mem for mem, score in recent_memories_raw
        if hasattr(mem, 'source_type') and mem.source_type in [
            MemorySourceType.USER_INTERACTION, MemorySourceType.ORA_RESPONSE,
            MemorySourceType.INTERNAL_REFLECTION, MemorySourceType.REASONING_MEMORY  # Adicionado
        ] and hasattr(mem, 'memory_id')
    ]

    MIN_MEMORIES_FOR_CLUSTERING = 5
    if len(memories_to_consolidate) < MIN_MEMORIES_FOR_CLUSTERING:
        logger.info(
            f"AURA-REFLECTOR (Latent Dream): Memórias de experiência insuficientes ({len(memories_to_consolidate)}/{MIN_MEMORIES_FOR_CLUSTERING}). Pulando ciclo de consolidação.")
        return None

    # 2. Extrair Embeddings das memórias selecionadas
    embeddings = []
    memories_with_embeddings = []
    for mem in memories_to_consolidate:
        if mem.memory_id in agent_instance.memory_service._embedding_cache:
            embeddings.append(agent_instance.memory_service._embedding_cache[mem.memory_id])
            memories_with_embeddings.append(mem)

    if len(embeddings) < MIN_MEMORIES_FOR_CLUSTERING:
        logger.info("AURA-REFLECTOR (Latent Dream): Memórias com embeddings insuficientes. Pulando.")
        return None

    embeddings_matrix = np.array(embeddings)
    logger.info(f"AURA-REFLECTOR (Latent Dream): Clusterizando {embeddings_matrix.shape[0]} vetores de memória.")

    # 3. Aplicar Clustering (DBSCAN) para encontrar conceitos densos
    # eps: A distância máxima entre duas amostras para uma ser considerada como na vizinhança da outra.
    # min_samples: O número de amostras em uma vizinhança para um ponto ser considerado como um ponto central.
    clustering = DBSCAN(eps=0.45, min_samples=3, metric='cosine')
    cluster_labels = clustering.fit_predict(embeddings_matrix)

    unique_labels = set(cluster_labels)
    proto_memories_created = 0
    consolidated_source_mem_ids = set()

    # 4. Processar cada cluster para extrair Proto-Memórias
    for label in unique_labels:
        if label == -1:
            continue  # Ignora pontos de ruído (não pertencem a nenhum cluster)

        cluster_indices = np.where(cluster_labels == label)[0]
        cluster_embeddings = embeddings_matrix[cluster_indices]

        # 4a. Calcular o Centróide (a Proto-Memória)
        centroid_vector = np.mean(cluster_embeddings, axis=0)
        centroid_vector /= np.linalg.norm(centroid_vector)  # Normalizar

        # 4b. Gerar um Rótulo Textual para o conceito
        cluster_mems = [memories_with_embeddings[i] for i in cluster_indices]
        cluster_theme = await _generate_theme_for_cluster(cluster_mems, agent_instance)

        if "indefinido" in cluster_theme or "vazio" in cluster_theme or "erro" in cluster_theme:
            logger.warning(
                f"AURA-REFLECTOR: Tema para cluster {label} não foi gerado com sucesso. Pulando criação de proto-memória.")
            continue

        logger.critical(f"AURA-REFLECTOR (Latent Dream): Novo conceito latente extraído: '{cluster_theme}'")

        # 4c. Salvar o Centróide como uma nova GenerativeMemory
        proto_memory = GenerativeMemory(
            seed_name=f"Conceito Latente: {cluster_theme}",
            seed_data=GenerativeSeed(
                seed_type="latent_concept",
                content=f"Um conceito central sobre '{cluster_theme}', consolidado a partir de {len(cluster_mems)} experiências passadas."
            ),
            source_type=MemorySourceType.SYNTHESIZED_SUMMARY,
            salience=MemorySalience.HIGH,
            keywords=["conceito_latente", "proto_memoria", "sonho_ia"] + cluster_theme.lower().split(),
            learning_value=0.8  # Alto valor de aprendizado
        )

        # O embedding para esta nova memória é o próprio centróide
        agent_instance.memory_service._embedding_cache[proto_memory.memory_id] = centroid_vector.tolist()
        await agent_instance.memory_service.add_specific_memory(proto_memory)
        proto_memories_created += 1

        # Adiciona os IDs das memórias-fonte que foram consolidadas
        for mem in cluster_mems:
            consolidated_source_mem_ids.add(mem.memory_id)

    # 5. Aplicar Esquecimento Ativo
    if consolidated_source_mem_ids:
        logger.info(
            f"AURA-REFLECTOR (Active Forgetting): Reduzindo saliência de {len(consolidated_source_mem_ids)} memórias-fonte consolidadas.")
        mems_to_update = [mem for mem in memories_to_consolidate if mem.memory_id in consolidated_source_mem_ids]

        for mem in mems_to_update:
            mem.dynamic_salience_score = 0.05  # Reduz drasticamente a importância
            # Re-salva a memória com a saliência atualizada
            await agent_instance.memory_service.add_specific_memory(mem)
        logger.critical(
            f"AURA-REFLECTOR (Active Forgetting): {len(mems_to_update)} memórias-fonte foram 'esquecidas' (saliência reduzida para 0.05).")

    # A função agora retorna algo para o ciclo principal saber o que aconteceu
    synthesis_result = {
        "narrative_text": f"Consolidated {len(consolidated_source_mem_ids)} memories into {proto_memories_created} new latent concepts." if proto_memories_created > 0 else "No new concepts were consolidated.",
        "narrative_coherence": 0.9 if proto_memories_created > 0 else 0.0,  # Placeholder
    }

    # Esta parte é importante para o próximo passo (Geração de Metas)
    if proto_memories_created > 0:
        # Pega o tema da primeira proto-memória como o "insight" principal do sonho
        first_cluster_label = next((lbl for lbl in unique_labels if lbl != -1), None)
        if first_cluster_label is not None:
            first_cluster_indices = np.where(cluster_labels == first_cluster_label)[0]
            first_cluster_mems = [memories_with_embeddings[i] for i in first_cluster_indices]
            synthesis_result["narrative_text"] = await _generate_theme_for_cluster(first_cluster_mems, agent_instance)

    return synthesis_result


async def perform_kg_synthesis_cycle(agent_id: str, agent_manager: AgentManager):
    """
    (REVISADO) Verifica memórias explícitas não processadas e as envia para o
    processador de KG apropriado (geral ou Aureola).
    """
    logger.info(f"AURA-KGS: Iniciando ciclo de síntese de KG para o agente {agent_id}")
    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance:
        logger.error(f"AURA-KGS: Não foi possível obter a instância do agente {agent_id}.")
        return

    # Lógica para encontrar memórias não processadas
    unprocessed_memories: List[ExplicitMemory] = []
    for mem in agent_instance.memory_service._in_memory_explicit_cache:
        # Adicionamos uma verificação extra para garantir que a memória não seja do tipo 'self_model'
        is_self_model = mem.memory_id == "ceaf_self_model_singleton_v1"
        if not mem.metadata.get("kg_processed") and not is_self_model:
            unprocessed_memories.append(mem)

    if not unprocessed_memories:
        logger.info(f"AURA-KGS: Nenhuma memória explícita nova para processar no agente {agent_id}.")
        return

    logger.warning(f"AURA-KGS: Encontradas {len(unprocessed_memories)} memórias para síntese de KG.")

    # <--- LÓGICA DE DESPACHO (DISPATCHER) --->
    aureola_transcriptions = []
    other_explicit_memories = []

    for mem in unprocessed_memories:
        if mem.metadata.get("ingestion_source") == "aureola_app":
            aureola_transcriptions.append(mem)
        else:
            other_explicit_memories.append(mem)

    kg_processor = KGProcessor(agent_instance.llm_service, agent_instance.memory_service)
    total_entities_created = 0
    total_relations_created = 0

    # Processa as transcrições da Aureola com o processador social
    if aureola_transcriptions:
        logger.info(f"AURA-KGS: Processando {len(aureola_transcriptions)} transcrições da Aureola...")
        entities, relations = await kg_processor.process_aureola_transcription_to_kg(aureola_transcriptions)
        total_entities_created += entities
        total_relations_created += relations
        logger.info(f"AURA-KGS (Aureola): Criados {entities} entidades e {relations} relações.")

    # Processa outras memórias com o processador geral
    if other_explicit_memories:
        logger.info(f"AURA-KGS: Processando {len(other_explicit_memories)} memórias gerais...")
        entities, relations = await kg_processor.process_memories_to_kg(other_explicit_memories)
        total_entities_created += entities
        total_relations_created += relations
        logger.info(f"AURA-KGS (Geral): Criados {entities} entidades e {relations} relações.")

    # <--- FIM DA LÓGICA DE DESPACHO --->

    if total_entities_created > 0 or total_relations_created > 0:
        logger.critical(
            f"AURA-KGS: Ciclo de síntese concluído para {agent_id}. "
            f"Total: {total_entities_created} entidades, {total_relations_created} relações."
        )

    # Marca todas as memórias processadas para não reprocessá-las
    for mem in unprocessed_memories:
        mem.metadata["kg_processed"] = True
    # A atualização será salva na próxima reescrita do MBS, o que é eficiente.

async def calculate_dynamic_proactive_interval(agent_instance, drives, body_state) -> int:
    """
    Calcula um intervalo dinâmico para a próxima mensagem proativa, considerando um
    conjunto mais rico de estados internos e histórico de interações.

    Args:
        agent_instance: A instância ativa do CEAFSystem do agente.
        drives: O estado atual dos drives motivacionais.
        body_state: O estado corporal virtual atual (cansaço, saturação).

    Returns:
        O intervalo em segundos para a próxima ação proativa.
    """
    # --- 1. Parâmetros Base ---
    # Intervalos em horas, convertidos para segundos.
    MIN_INTERVAL_H = 20.15  # Mínimo de 3 minutos
    DEFAULT_INTERVAL_H = 20.3 # Padrão de 6 minutos
    MAX_INTERVAL_H = 80.6   # Máximo de 12 minutos

    current_interval = DEFAULT_INTERVAL_H * 3600

    # --- 2. Modificadores de Estado Interno (efeitos multiplicativos) ---
    # Modificadores > 1.0 aumentam o intervalo (mais tempo de espera).
    # Modificadores < 1.0 diminuem o intervalo (menos tempo de espera).

    # a) Modificador de Drives Motivacionais
    # Conexão alta reduz drasticamente o tempo. Curiosidade alta também, mas menos.
    connection_modifier = 1.0 - (drives.connection.intensity * 0.7)
    curiosity_modifier = 1.0 - (drives.curiosity.intensity * 0.4)
    drive_modifier = connection_modifier * curiosity_modifier

    # b) Modificador de Estado Corporal
    # Fadiga alta aumenta drasticamente o tempo. Saturação também, mas menos.
    fatigue_modifier = 1.0 + (body_state.cognitive_fatigue * 1.5)  # Range: 1.0 (sem efeito) a 2.5 (150% mais lento)
    saturation_modifier = 1.0 + (
                body_state.information_saturation * 0.5)  # Range: 1.0 (sem efeito) a 1.5 (50% mais lento)
    body_modifier = fatigue_modifier * saturation_modifier

    # c) Modificador de Histórico de Interação (assíncrono)
    history_modifier = 1.0
    try:
        # Busca a memória do estado interno do último turno para avaliar a qualidade da última interação.
        last_internal_state_mem = await agent_instance.memory_service.search_raw_memories(
            query="estado interno do último turno",
            top_k=1,
            memory_type_filter="explicit"  # Força a busca por memórias explícitas
        )
        if last_internal_state_mem:
            mem_obj, _ = last_internal_state_mem[0]
            if hasattr(mem_obj, 'content') and mem_obj.content and mem_obj.content.structured_data:
                report_data = mem_obj.content.structured_data.get("report", {})
                cognitive_flow = report_data.get("cognitive_flow", 0.0)
                ethical_tension = report_data.get("ethical_tension", 0.0)

                # Se a última interação foi fluida, diminui o intervalo. Se foi tensa, aumenta.
                flow_bonus = cognitive_flow * 0.3  # Reduz em até 30%
                tension_penalty = ethical_tension * 0.5  # Aumenta em até 50%
                history_modifier = (1.0 - flow_bonus) + tension_penalty
    except Exception as e:
        logger.warning(f"AURA-PROACTIVE: Não foi possível obter o estado da última interação da memória: {e}")

    # --- 3. Cálculo Final ---
    # Aplica todos os modificadores ao intervalo padrão.
    modified_interval = current_interval * drive_modifier * body_modifier * history_modifier

    # Adiciona uma variação aleatória (jitter) de +/- 15% para um comportamento menos previsível.
    jitter = 1.0 + random.uniform(-0.15, 0.15)
    final_interval_before_clamping = modified_interval * jitter

    # Garante que o intervalo final esteja dentro dos limites MÍNIMO e MÁXIMO.
    final_interval = int(max(MIN_INTERVAL_H * 3600, min(final_interval_before_clamping, MAX_INTERVAL_H * 3600)))

    logger.info(
        f"AURA-PROACTIVE (Intervalo Dinâmico V2): "
        f"Padrão: {DEFAULT_INTERVAL_H:.1f}h | "
        f"Mods(Drives:{drive_modifier:.2f}, Corpo:{body_modifier:.2f}, Hist:{history_modifier:.2f}) | "
        f"Resultado: {final_interval / 3600:.1f}h (Limites: {MIN_INTERVAL_H:.1f}h - {MAX_INTERVAL_H:.1f}h)"
    )

    return final_interval


async def trigger_proactive_behavior(agent_id: str, agent_manager: AgentManager, db_repo: AgentRepository):
    """
    Verifica o estado de um agente e dispara uma mensagem proativa com base
    em uma combinação emergente de seus drives e um intervalo dinâmico.
    (MODIFICADO para ser async e passar a instância do agente)
    """
    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance:
        return

    # 1. Carregar todos os estados internos relevantes
    drives = agent_instance.motivational_drives
    body_state = agent_instance.body_state

    # 2. Calcular o intervalo dinâmico (agora uma chamada assíncrona)
    min_proactive_interval = await calculate_dynamic_proactive_interval(agent_instance, drives, body_state)

    # 3. Calcular o Proactivity Score (lógica inalterada)
    proactivity_score = (
            (drives.curiosity.intensity * 0.5) + (drives.connection.intensity * 0.35) +
            (drives.mastery.intensity * 0.1) + (drives.consistency.intensity * 0.05)
    )
    proactivity_score -= body_state.cognitive_fatigue * 0.3

    logger.info(
        f"AURA-PROACTIVE CHECK: Agente {agent_id} | Score: {proactivity_score:.4f} | Limiar: {PROACTIVITY_ACTIVATION_THRESHOLD}")

    # 4. Verificar o gatilho de ativação
    if proactivity_score < PROACTIVITY_ACTIVATION_THRESHOLD:
        return

    logger.critical(f"AURA-PROACTIVE: Agente {agent_id} ativou gatilho (Score: {proactivity_score:.2f})!")

    dominant_drive = max(
        {"curiosity": drives.curiosity.intensity, "connection": drives.connection.intensity,
         "mastery": drives.mastery.intensity},
        key=lambda k: drives.model_dump()[k]['intensity']
    )

    # 5. Verificar o rate limit, agora usando o intervalo dinâmico
    dynamic_config = agent_instance.ceaf_dynamic_config
    last_proactive_ts = dynamic_config.get("last_proactive_message_ts", 0)
    if (datetime.now().timestamp() - last_proactive_ts) < min_proactive_interval:
        logger.warning(
            f"AURA-PROACTIVE: Ação suprimida para {agent_id} devido ao intervalo dinâmico "
            f"({(datetime.now().timestamp() - last_proactive_ts) / 3600:.1f}h / {min_proactive_interval / 3600:.1f}h)."
        )
        return

    # Etapa 6: Gerar a mensagem proativa (sem mudanças)
    proactive_message = await agent_instance.generate_proactive_message(dominant_drive)
    if not proactive_message:
        logger.warning(f"AURA-PROACTIVE: Agente {agent_id} decidiu não gerar uma mensagem.")
        return

    # --- NOVA ETAPA 7: Chamar a API para despachar a mensagem ---
    try:
        user_id = agent_instance.config.get("user_id")
        if not user_id:
            logger.error(
                f"AURA-PROACTIVE: Agente {agent_id} não tem um user_id associado. Não é possível enviar mensagem.")
            return

        endpoint_url = f"{AURA_API_BASE_URL}/agents/dispatch-proactive"
        payload = {
            "user_id": user_id,
            "agent_id": agent_id,
            "message": proactive_message
        }

        system_token = "seu-token-secreto-de-sistema"
        headers = {"Authorization": f"Bearer {system_token}"}

        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint_url, json=payload, headers=headers)
            if response.status_code == 200:
                logger.critical(f"AURA-PROACTIVE: Mensagem para user_id {user_id} despachada com sucesso pela API.")
                dynamic_config["last_proactive_message_ts"] = datetime.now().timestamp()
                await save_ceaf_dynamic_config(agent_instance.persistence_path, dynamic_config)
                drives.curiosity.intensity, drives.connection.intensity, drives.mastery.intensity = 0.5, 0.5, 0.5
                await agent_instance._save_motivational_drives()
            else:
                logger.error(
                    f"AURA-PROACTIVE: Falha ao despachar mensagem pela API. Status: {response.status_code}, Resposta: {response.text}")
    except Exception as e:
        logger.error(f"AURA-PROACTIVE: Erro inesperado durante o despacho via API: {e}", exc_info=True)


# --- MÉTODO PRINCIPAl main_aura_reflector_cycle ---
async def main_aura_reflector_cycle(agent_manager: AgentManager, db_repo: AgentRepository):
    """
    Ciclo principal do Aura Reflector. V3.2 - Foco em agentes ativos e síntese.
    """
    logger.info("--- Iniciando Ciclo do Aura Reflector (V3.2 - Foco em Agentes Ativos) ---")

    # <<< INÍCIO DA MUDANÇA >>>
    try:
        # Busca apenas agentes que tiveram atividade recente
        recently_active_agent_ids = db_repo.get_recently_active_agent_ids(hours=48)
        if not recently_active_agent_ids:
            logger.info("[Refletor] Nenhum agente com atividade recente encontrado. Pulando ciclo de proatividade.")
            return

        logger.info(f"[Refletor] Encontrados {len(recently_active_agent_ids)} agentes ativos para processar.")

    except Exception as e:
        logger.error(f"[Refletor] Erro ao buscar agentes ativos: {e}. Abortando ciclo.")
        return

    for agent_id in recently_active_agent_ids:
        agent_config = agent_manager.agent_configs.get(agent_id)
        if not agent_config:
            logger.warning(f"[Refletor] Pulando agente {agent_id} pois sua configuração não foi encontrada.")
            continue

        logger.info(f"--- [Refletor] Processando Agente: {agent_config.name} ({agent_id}) ---")

        # --- Tarefa 1: Restaurar Estado Corporal ("Descanso") ---
        try:
            agent_instance = agent_manager.get_agent_instance(agent_id)
            if agent_instance:
                # Restaura estado corporal
                if hasattr(agent_instance, 'body_state'):
                    agent_instance.body_state.cognitive_fatigue *= 0.1
                    agent_instance.body_state.information_saturation *= 0.5
                    await agent_instance._save_body_state()
                    logger.info(f"AURA-REFLECTOR (Descanso): Estado corporal do agente {agent_id} restaurado.")

                # <<< INÍCIO DA MUDANÇA >>>
                # Simula o aumento passivo da motivação com o tempo
                if hasattr(agent_instance, 'motivational_drives'):
                    drives = agent_instance.motivational_drives
                    time_delta_seconds = datetime.now().timestamp() - drives.last_updated
                    time_delta_hours = time_delta_seconds / 3600

                    # Aumenta a curiosidade e a conexão com o passar do tempo
                    drives.curiosity.intensity += 0.05 * time_delta_hours  # Aumenta 5% por hora
                    drives.connection.intensity += 0.1 * time_delta_hours  # Aumenta 10% por hora

                    # Normaliza para garantir que fiquem entre 0 e 1
                    drives.curiosity.intensity = max(0.0, min(1.0, drives.curiosity.intensity))
                    drives.connection.intensity = max(0.0, min(1.0, drives.connection.intensity))

                    drives.last_updated = datetime.now().timestamp()
                    await agent_instance._save_motivational_drives()
                    logger.info(
                        f"AURA-REFLECTOR (Drives): Drives passivos atualizados para agente {agent_id}. "
                        f"Curiosidade: {drives.curiosity.intensity:.2f}, Conexão: {drives.connection.intensity:.2f}"
                    )
        except Exception as e:
            logger.error(f"[Refletor ERRO/Descanso] Agente {agent_id}: {e}", exc_info=True)

        # --- Tarefa 2: Comportamento Proativo ---
        prioritize_synthesis = False
        if agent_instance and hasattr(agent_instance, 'body_state'):
            if agent_instance.body_state.information_saturation > 0.75:
                prioritize_synthesis = True
                logger.critical(
                    f"[Refletor] Agente {agent_id} com alta saturação ({agent_instance.body_state.information_saturation:.2f}). Priorizando 'sonho'.")

        # --- Tarefa 2: Comportamento Proativo ---
        if not prioritize_synthesis:  # Só tenta ser proativo se não estiver sobrecarregado
            try:
                await trigger_proactive_behavior(agent_id, agent_manager, db_repo)
            except Exception as e:
                logger.error(f"[Refletor ERRO/Proativo] Agente {agent_id}: {e}", exc_info=True)
        else:
            logger.info(
                f"[Refletor] Pulando verificação de proatividade para o Agente {agent_id} para focar na síntese.")




        # --- Tarefa 3: Síntese de Insight ("Sonho") ---
        # Esta é a tarefa principal de aprendizado offline.
        synthesis_insight_text = None
        try:
            synthesis_result_dict = await perform_autonomous_clustering_and_synthesis(agent_id, agent_manager)

            # --- VERIFICAÇÃO ROBUSTA ---
            if synthesis_result_dict and isinstance(synthesis_result_dict, dict):
                synthesis_insight_text = synthesis_result_dict.get("narrative_text")
                if not synthesis_insight_text:
                    logger.info(
                        f"[Refletor] Síntese concluída, mas não gerou 'narrative_text' para o agente {agent_id}.")
            else:
                logger.info(
                    f"[Refletor] Ciclo de síntese não produziu resultados para o agente {agent_id} (pode ser intencional, ex: poucas memórias).")
            # --- FIM DA VERIFICAÇÃO ---

        except Exception as e:
            logger.error(f"[Refletor ERRO/Síntese] Agente {agent_id}: {e}", exc_info=True)

        # <<< INÍCIO DA IMPLEMENTAÇÃO: ALÍVIO DA SATURAÇÃO APÓS O SONHO >>>
        if synthesis_insight_text:  # Se o sonho produziu um insight
            if agent_instance and hasattr(agent_instance, 'body_state'):
                # Reduz a saturação como recompensa pela consolidação
                agent_instance.body_state.information_saturation *= 0.7
                await agent_instance._save_body_state()
                logger.critical(
                    f"AURA-REFLECTOR (Consolidação): Saturação reduzida para {agent_instance.body_state.information_saturation:.2f} após sonho bem-sucedido.")

        # --- NOVA TAREFA 4: GERAÇÃO DE METAS EMERGENTES ---
        if synthesis_insight_text and agent_instance:
            try:
                # Carrega o auto-modelo mais recente para dar contexto
                current_self_model = await agent_instance._ensure_self_model()

                # Chama a nova função para gerar o objetivo
                emergent_goal = await generate_emergent_goals(
                    agent_instance=agent_instance,
                    synthesis_insight=synthesis_insight_text,
                    self_model=current_self_model
                )

                # Se um objetivo foi gerado com sucesso, salva-o na memória
                if emergent_goal:
                    await agent_instance.memory_service.add_specific_memory(emergent_goal)
            except Exception as e:
                logger.error(f"[Refletor ERRO/Geração de Metas] Agente {agent_id}: {e}", exc_info=True)

        # --- Tarefa 5 (anteriormente 4): Síntese de Grafo de Conhecimento ---
        try:
            await perform_kg_synthesis_cycle(agent_id, agent_manager)
        except Exception as e:
            logger.error(f"[Refletor ERRO/KGS] Agente {agent_id}: {e}", exc_info=True)

    logger.info("--- Ciclo do Aura Reflector Concluído ---")