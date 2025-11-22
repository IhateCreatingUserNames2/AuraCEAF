# ARQUIVO REATORADO E IMPLEMENTADO: ceaf_v3/system.py
# Implementação funcional e integrada da Arquitetura de Síntese CEAF V3.
# Este arquivo contém todos os módulos principais e o orquestrador CEAFSystem.
# Refatorado para modularizar o AgencyModule.

import asyncio
import re
import uuid
import json
import logging
import os
import time
from typing import Dict, Any, Optional, List, Tuple, Literal, TYPE_CHECKING
from pathlib import Path
from ceaf_core.models import CeafSelfRepresentation, CognitiveProfile

from ceaf_core.services.cognitive_log_service import CognitiveLogService
from ceaf_core.services.evolution_log_service import EvolutionLogger
from ceaf_core.genlang_types import ResponsePacket, IntentPacket, GenlangVector, CognitiveStatePacket, GuidancePacket, \
    ToolOutputPacket, CommonGroundTracker, UserRepresentation
from ceaf_core.utils.embedding_utils import compute_adaptive_similarity
from ceaf_core.translators.human_to_genlang import HumanToGenlangTranslator
from ceaf_core.translators.genlang_to_human import GenlangToHumanTranslator
from ceaf_core.modules.lcam_module import LCAMModule
from ceaf_core.modules.memory_blossom.memory_types import InteroceptivePredictionMemory, GenerativeMemory, AnyMemoryType
import litellm
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ceaf_core.modules.memory_blossom.memory_lifecycle_manager import MEMORY_TYPES_LOADED_SUCCESSFULLY
from ceaf_core.modules.vre_engine.vre_engine import VREEngineV3, EthicalAssessment
from ceaf_core.utils.observability_types import ObservabilityManager, ObservationType, Observation
from ceaf_core.agency_module import AgencyModule, AgencyDecision, generate_tools_summary
from ceaf_core.utils.common_utils import create_error_tool_response, extract_json_from_text
from ceaf_core.utils.embedding_utils import get_embedding_client
from ceaf_core.modules.mcl_engine.self_state_analyzer import ORAStateAnalysis, analyze_ora_turn_observations
from ceaf_core.modules.mcl_engine import MCLEngine
from ceaf_core.services.llm_service import LLMService
from ceaf_core.services.mbs_memory_service import MBSMemoryService
from ceaf_core.modules.memory_blossom.memory_types import (
    ExplicitMemory,
    ExplicitMemoryContent,
    MemorySourceType,
    MemorySalience
)
from ceaf_core.modules.mcl_engine.self_state_analyzer import analyze_ora_turn_observations
from ceaf_core.genlang_types import VirtualBodyState
from ceaf_core.modules.embodiment_module import EmbodimentModule

from ceaf_core.genlang_types import MotivationalDrives
from ceaf_core.modules.motivational_engine import MotivationalEngine
from ceaf_core.modules.interoception_module import ComputationalInteroception
from ceaf_core.modules.memory_blossom.memory_types import EmotionalMemory, EmotionalTag
from ceaf_core.modules.ncim_engine.ncim_module import NCIMModule
from ceaf_core.modules.refinement_module import RefinementModule, RefinementPacket
from ceaf_core.modules.memory_blossom.advanced_synthesizer import AdvancedMemorySynthesizer, StoryArcType
from ceaf_core.utils.config_utils import load_ceaf_dynamic_config, save_ceaf_dynamic_config

if TYPE_CHECKING:
    from agent_manager import AgentManager
    from database.models import AgentRepository

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CEAFv3_System")

# --- Constantes do Sistema ---
SELF_MODEL_MEMORY_ID = "ceaf_self_model_singleton_v1"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'



# --- Lógica de Configuração Dinâmica (Correta e Mantida) ---
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
    "VRE": {"evaluation_threshold": 0.6}
}


class NCIMEngine:
    """Implementação real do Narrative Coherence & Identity Module."""

    def __init__(self, llm_service: LLMService, memory_service: MBSMemoryService):
        self.llm = llm_service
        self.memory = memory_service

    async def update_identity(self, self_model_before: CeafSelfRepresentation, final_response: str, query: str,
                              **kwargs):
        logger.info("NCIMEngine: Iniciando atualização do auto-modelo...")
        prompt = f"""
        Você é um módulo de reflexão de identidade para uma IA.
        Dado o auto-modelo JSON atual, a última consulta do usuário e a resposta final da IA, gere um novo auto-modelo JSON atualizado.
        O campo 'last_update_reason' deve explicar a mudança. Incremente a 'version'.
        Faça pequenas e incrementais mudanças que reflitam o que a IA aprendeu ou demonstrou na última interação.

        AUTO-MODELO ATUAL:
        {self_model_before.model_dump_json(indent=2)}

        ÚLTIMA INTERAÇÃO:
        - Usuário: "{query}"
        - Resposta da IA: "{final_response}"

        NOVO AUTO-MODELO JSON ATUALIZADO:
        """
        updated_model_str = await self.llm.ainvoke(self.llm.config.smart_model, prompt)
        try:
            updated_model_data = json.loads(updated_model_str)
            new_self_model = CeafSelfRepresentation(**updated_model_data)
            content = ExplicitMemoryContent(structured_data=new_self_model.model_dump())
            self_model_to_save = ExplicitMemory(
                memory_id=SELF_MODEL_MEMORY_ID,
                content=content,
                memory_type="explicit",
                source_type=MemorySourceType.INTERNAL_REFLECTION,
                salience=MemorySalience.CRITICAL,
                keywords=["self-model", "identity", "ceaf-core"]
            )
            await self.memory.add_specific_memory(self_model_to_save)
            logger.info(f"NCIMEngine: Auto-modelo atualizado para a versão {new_self_model.version}.")
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(
                f"NCIMEngine: Erro ao atualizar o auto-modelo. A resposta do LLM não era um JSON válido. Erro: {e}")


class PersistentLogService:
    """Serviço real de log persistente (baseado em arquivo)."""

    def __init__(self, persistence_path: Path):
        self.log_path = persistence_path / "logs"
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path / "ceaf_v3_turns.jsonl"
        logger.info(f"PersistentLogService: Registrando logs em {self.log_file}.")

    async def log_turn(self, **kwargs):
        log_entry = {
            "turn_id": kwargs.get("turn_id"), "timestamp": time.time(), "session_id": kwargs.get("session_id"),
            "query": kwargs.get("query"), "final_response": kwargs.get("final_response"),
            "mcl_guidance": kwargs.get("mcl_guidance"),
            "vre_assessment": kwargs.get("vre_assessment").model_dump() if hasattr(kwargs.get("vre_assessment"),
                                                                                   'model_dump') else str(
                kwargs.get("vre_assessment")),
        }
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + "\n")
        logger.info(f"PersistentLogService: Turno '{kwargs.get('turn_id')}' registrado.")


class CEAFSystem:

    def __init__(self, config: Dict[str, Any],
                 agent_manager: Optional['AgentManager'] = None,
                 db_repo: Optional['AgentRepository'] = None):
        logger.info(f"CEAFSystem V3 (Web-Enabled): Inicializando para o agente ID: {config.get('agent_id')}...")
        self.config, self.agent_id = config, config.get("agent_id", "default_agent")
        self.agent_manager = agent_manager
        self.db_repo = db_repo
        self.persistence_path = Path(config.get("persistence_path", f"./agent_data/{self.agent_id}"))
        self.ceaf_dynamic_config = load_ceaf_dynamic_config(self.persistence_path)
        self.cognitive_profile = self._load_cognitive_profile()
        self.llm_service = LLMService(config=self.cognitive_profile.llm_config)
        self.user_model = self._load_user_model()
        self.embedding_client = get_embedding_client()
        self.motivational_engine = MotivationalEngine(config=self.cognitive_profile.drives_config)
        self.embodiment_module = EmbodimentModule(config=self.cognitive_profile.body_config)
        self.motivational_drives = self._load_motivational_drives()
        self.body_state = self._load_body_state()
        self.memory_service = MBSMemoryService(
            memory_store_path=self.persistence_path,
            config=self.cognitive_profile.memory_config
        )
        self.memory_service.start_lifecycle_management_tasks()

        self.tool_registry = {
            "query_long_term_memory": self.memory_service.search_raw_memories
        }

        tools_summary = generate_tools_summary(self.tool_registry)
        logger.info(f"Resumo de ferramentas gerado para o AgencyModule:\n{tools_summary}")

        self.lcam = LCAMModule(self.memory_service)
        self.ncim = NCIMModule(
            self.llm_service,
            self.memory_service,
            self.persistence_path,
            prompts=self.cognitive_profile.prompts
        )
        mcl_config = self.ceaf_dynamic_config.get("MCL", {})
        self.mcl = MCLEngine(
            config=self.ceaf_dynamic_config.get("MCL", {}),
            agent_config=self.config,
            lcam_module=self.lcam,
            llm_service=self.llm_service,
            mcl_profile=self.cognitive_profile.mcl_config
        )
        vre_config = self.ceaf_dynamic_config.get("VRE", {})
        self.vre = VREEngineV3(config=vre_config)
        self.agency_module = AgencyModule(
            llm_service=self.llm_service,
            vre_engine=self.vre,
            mcl_engine=self.mcl,
            available_tools_summary=tools_summary,
            prompts=self.cognitive_profile.prompts,
            agency_config=self.cognitive_profile.mcl_config
        )
        self.refinement_module = RefinementModule()
        self.htg_translator = HumanToGenlangTranslator(
            prompts=self.cognitive_profile.prompts,
            llm_config=self.cognitive_profile.llm_config
        )
        self.gth_translator = GenlangToHumanTranslator(
            prompts=self.cognitive_profile.prompts,
            llm_config=self.cognitive_profile.llm_config
        )
        self.cognitive_log_service = CognitiveLogService(self.persistence_path)

        self.evolution_logger = EvolutionLogger(self.persistence_path)

        self.active_sessions: Dict[str, Dict] = {}
        logger.info("CEAFSystem V3: Todas as instâncias foram criadas com sucesso.")

    def reload_cognitive_profile(self, new_profile: CognitiveProfile):
        """
        Atualiza a configuração do agente em tempo real (Hot Reload).
        Chamado pelo AgentManager quando o JSON é editado.
        """
        logger.info(f"🔄 HOT RELOAD: Atualizando perfil cognitivo do agente {self.agent_id}...")
        self.cognitive_profile = new_profile

        # 1. Atualizar LLM Service
        self.llm_service.update_config(new_profile.llm_config)

        # 2. Atualizar Memória
        self.memory_service.update_config(new_profile.memory_config)

        # 3. Atualizar Tradutores (Prompts)
        self.htg_translator.update_prompts(new_profile.prompts)
        self.gth_translator.update_prompts(new_profile.prompts)

        # 4. Atualizar MCL e Agency
        self.mcl.update_profile(new_profile.mcl_config)
        self.agency_module.update_config(new_profile.prompts, new_profile.mcl_config)
        self.motivational_engine.config = new_profile.drives_config
        self.embodiment_module.config = new_profile.body_config
        # 5. Atualizar NCIM
        self.ncim.update_prompts(new_profile.prompts)

        logger.info("✅ HOT RELOAD: Configuração aplicada com sucesso.")

    def _load_cognitive_profile(self) -> CognitiveProfile:
        """Carrega o CognitiveProfile do disco ou cria um padrão."""
        profile_path = self.persistence_path / "cognitive_profile.json"
        if profile_path.exists():
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return CognitiveProfile(**data)
            except Exception as e:
                logger.error(f"Erro ao carregar perfil cognitivo: {e}. Usando padrões.")

        # Se não existir ou falhar, retorna o padrão (hardcoded defaults)
        return CognitiveProfile()

    def _load_body_state(self) -> VirtualBodyState:
        body_file = self.persistence_path / "virtual_body_state.json"
        if body_file.exists():
            try:
                with open(body_file, 'r') as f:
                    data = json.load(f)
                    return VirtualBodyState(**data)
            except (json.JSONDecodeError, ValidationError):
                pass
        return VirtualBodyState()

    def _load_user_model(self) -> 'UserRepresentation':
        """Carrega o modelo de usuário do arquivo JSON ou cria um novo."""
        from ceaf_core.genlang_types import UserRepresentation  # Movido para dentro para evitar import circular
        user_model_file = self.persistence_path / "user_model.json"
        if user_model_file.exists():
            try:
                with open(user_model_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.warning(f"[USER MODEL] Modelo de usuário carregado de {user_model_file}")  # LOG AQUI
                    return UserRepresentation(**data)
            except (json.JSONDecodeError, ValidationError, IOError) as e:
                logger.error(f"Falha ao carregar ou validar user_model.json: {e}. Criando um novo.")

        logger.warning(
            "[USER MODEL] Nenhum modelo de usuário encontrado ou arquivo corrompido. Criando um novo modelo padrão.")  # LOG AQUI
        return UserRepresentation()

    async def _save_user_model(self):
        """Salva o estado atual do modelo de usuário em um arquivo JSON."""
        user_model_file = self.persistence_path / "user_model.json"
        try:
            with open(user_model_file, 'w', encoding='utf-8') as f:
                f.write(self.user_model.model_dump_json(indent=2))
            logger.info("Modelo de usuário salvo no disco.")
        except IOError as e:
            logger.error(f"Falha ao salvar o modelo de usuário: {e}", exc_info=True)

    async def _save_body_state(self):
        body_file = self.persistence_path / "virtual_body_state.json"
        with open(body_file, 'w') as f:
            f.write(self.body_state.model_dump_json())

    async def _update_and_save_body_state(self, turn_metrics: dict):
        # USAR A INSTÂNCIA DO SELF, NÃO CRIAR NOVA
        self.body_state = self.embodiment_module.update_body_state(self.body_state, turn_metrics)
        await self._save_body_state()

    def _load_motivational_drives(self) -> MotivationalDrives:
        drives_file = self.persistence_path / "motivational_state.json"
        if drives_file.exists():
            try:
                with open(drives_file, 'r') as f:
                    data = json.load(f)
                    return MotivationalDrives(**data)
            except (json.JSONDecodeError, ValidationError):
                pass  # Carrega o padrão se o arquivo estiver corrompido
        return MotivationalDrives()

    async def _save_motivational_drives(self):
        drives_file = self.persistence_path / "motivational_state.json"
        with open(drives_file, 'w') as f:
            f.write(self.motivational_drives.model_dump_json())

    async def _update_reality_score(self, predicted_text: str, actual_text: str):
        """Calcula a similaridade e atualiza o Reality Score e o Simulation Trust de forma assíncrona."""
        try:
            embeddings = await self.embedding_client.get_embeddings(
                [predicted_text, actual_text], context_type="default_query"
            )
            match_score = compute_adaptive_similarity(embeddings[0], embeddings[1])

            # Carrega a configuração, atualiza e salva
            config_path = self.persistence_path
            dynamic_config = load_ceaf_dynamic_config(config_path)

            calib_config = dynamic_config.setdefault("SIMULATION_CALIBRATION", {
                "reality_score": 0.75, "simulation_trust": 0.75, "samples_collected": 0,
                "ema_alpha": 0.1, "activation_threshold": 0.55
            })

            # --- ATUALIZAÇÃO DO REALITY SCORE (EMA) ---
            old_score = calib_config.get("reality_score", 0.75)
            alpha = calib_config.get("ema_alpha", 0.1)
            new_score = (alpha * match_score) + ((1 - alpha) * old_score)
            calib_config["reality_score"] = new_score

            # +++ INÍCIO DA ADIÇÃO (ATUALIZAÇÃO DO SIMULATION_TRUST) +++
            old_trust = calib_config.get("simulation_trust", 0.75)
            # A lógica é a mesma: atualizamos a confiança com a evidência mais recente (match_score)
            new_trust = (alpha * match_score) + ((1 - alpha) * old_trust)
            calib_config["simulation_trust"] = new_trust
            # +++ FIM DA ADIÇÃO +++

            calib_config["samples_collected"] += 1
            calib_config["last_updated_ts"] = time.time()

            await save_ceaf_dynamic_config(config_path, dynamic_config)

            # --- LOG ATUALIZADO ---
            logger.warning(
                f"REALITY CHECK: RealityScore atualizado para {new_score:.2f}, "
                f"SimulationTrust atualizado para {new_trust:.2f} (Match deste turno: {match_score:.2f})"
            )
            # --- FIM DO LOG ATUALIZADO ---

        except Exception as e:
            logger.error(f"Erro ao atualizar o Reality Score: {e}", exc_info=True)

    async def _gather_mycelial_consensus(self, relevant_memory_vectors: List[GenlangVector]) -> Optional[GenlangVector]:
        """
        Calcula um "vetor de consenso" a partir de uma lista de vetores de memória,
        ponderado pela saliência dinâmica de cada memória.
        Inspirado pela inteligência coletiva de redes miceliais.
        """
        if not relevant_memory_vectors:
            return None

        weighted_votes = []

        # Precisamos dos objetos de memória completos para obter a saliência
        memory_ids = [vec.metadata.get("memory_id") for vec in relevant_memory_vectors if vec.metadata]

        for mem_id, mem_vec in zip(memory_ids, relevant_memory_vectors):
            if not mem_id:
                continue

            # Obtenha o objeto de memória completo para acessar dynamic_salience_score
            memory_obj = await self.memory_service.get_memory_by_id(mem_id)
            if memory_obj and hasattr(memory_obj, 'dynamic_salience_score'):
                # O "peso" do voto é a saliência da memória
                salience_weight = memory_obj.dynamic_salience_score

                # Pondera o vetor da memória pelo seu peso
                weighted_vote = np.array(mem_vec.vector) * salience_weight
                weighted_votes.append(weighted_vote)

        if not weighted_votes:
            logger.warning("Mycelial Consensus: Nenhuma memória ponderável encontrada para calcular o consenso.")
            return None

        # Calcula a média dos vetores ponderados
        consensus_vector_np = np.mean(weighted_votes, axis=0)

        # Normaliza o vetor resultante para ter comprimento 1 (boa prática)
        norm = np.linalg.norm(consensus_vector_np)
        if norm > 0:
            consensus_vector_np /= norm

        logger.info(f"Mycelial Consensus: Vetor de consenso gerado a partir de {len(weighted_votes)} memórias.")

        # Empacota o resultado em um GenlangVector
        return GenlangVector(
            vector=consensus_vector_np.tolist(),
            source_text="Collective insight from active memory network (mycelial consensus)",
            model_name="synthesized_consensus",
            metadata={"is_consensus_vector": True}
        )

    async def _build_initial_cognitive_state(
            self,
            intent_packet: IntentPacket,
            chat_history: List[Dict[str, str]] = None,
            common_ground: Optional['CommonGroundTracker'] = None
    ) -> Tuple[CeafSelfRepresentation, CognitiveStatePacket]:
        """
        V2.3 (Experiential Recall): Constrói o estado cognitivo com uma busca dedicada
        por memórias de interações passadas, combatendo a "amnésia".
        """
        logger.info("--- Construindo Estado Cognitivo Inicial (V2.3 Experiential Recall) ---")


        # Carrega os pesos de atenção da configuração dinâmica
        attention_config = self.ceaf_dynamic_config.get("WORKSPACE_ATTENTION", {})
        identity_weight = attention_config.get("identity_weight", 1.0)
        memories_weight = attention_config.get("memories_weight", 1.0)

        # Modula a quantidade de memórias a serem buscadas com base no peso
        # Mapeia o peso [0.5, 1.5] para um top_k de [2, 6] (exemplo)
        top_k_identity = int(2 * identity_weight)
        top_k_knowledge = int(3 * memories_weight)
        top_k_wisdom = int(2 * memories_weight)
        top_k_experiential = int(4 * memories_weight)

        logger.info(f"Attention Weights: Identity_k={top_k_identity}, Knowledge_k={top_k_knowledge}, "
                    f"Wisdom_k={top_k_wisdom}, Experiential_k={top_k_experiential}")



        # --- ETAPA 1: CARREGAR A IDENTIDADE (Inalterado) ---
        self_model = await self._ensure_self_model()
        identity_vector = await self.ncim.get_current_identity_vector(self_model)
        identity_query = f"Quais são minhas crenças centrais, valores e função primária, como {self.config.get('name')}?"
        identity_memories_raw = await self.memory_service.search_raw_memories(
            query=identity_query, top_k=top_k_identity,
            source_type_filter=MemorySourceType.EXTERNAL_INGESTION.value
        )
        identity_memories = [mem for mem, score in identity_memories_raw]

        # --- ETAPA 2: CURADORIA DE MEMÓRIA CONTEXTUALIZADA (Lógica Aprimorada) ---
        base_query = intent_packet.query_vector.source_text or ""

        # Busca por CONHECIMENTO (fatos, conceitos)
        knowledge_query = f"Fatos, conceitos e definições sobre: {base_query}"
        knowledge_memories_raw = await self.memory_service.search_raw_memories(
            query=knowledge_query, top_k=top_k_knowledge,
            chat_history=chat_history,
        )
        knowledge_memories = [mem for mem, score in knowledge_memories_raw]

        # Busca por SABEDORIA (procedimentos, estratégias)
        wisdom_query = f"Qual é o melhor procedimento ou estratégia para lidar com uma questão sobre: {base_query}"
        wisdom_memories_raw = await self.memory_service.search_raw_memories(
            query=wisdom_query, top_k=top_k_wisdom,
            memory_type_filter="procedural"
        )
        wisdom_memories = [mem for mem, score in wisdom_memories_raw]

        ### NOVA ETAPA: BUSCA POR MEMÓRIA EXPERIENCIAL (A CORREÇÃO CRÍTICA) ###
        experiential_query = f"Lembranças de conversas ou reflexões passadas relacionadas a: {base_query}"
        experiential_memories_raw = await self.memory_service.search_raw_memories(
            query=experiential_query,
            top_k=top_k_experiential,  # <-- USA O VALOR DINÂMICO
            memory_type_filter="explicit",
            source_type_filter=MemorySourceType.USER_INTERACTION.value
        )
        experiential_memories = [mem for mem, score in experiential_memories_raw]
        if not experiential_memories:  # Fallback se a primeira busca falhar
            experiential_memories_raw_fallback = await self.memory_service.search_raw_memories(
                query=experiential_query, top_k=3, memory_type_filter="reasoning"
            )
            experiential_memories.extend([mem for mem, score in experiential_memories_raw_fallback])


        # --- ETAPA 3: CONSOLIDAR E CONSTRUIR O PACOTE DE ESTADO ---
        curated_memories_dict: Dict[str, AnyMemoryType] = {}

        # Adiciona na ordem de prioridade (as mais importantes sobrescrevem duplicatas)
        for mem in identity_memories: curated_memories_dict[mem.memory_id] = mem
        for mem in knowledge_memories: curated_memories_dict[mem.memory_id] = mem
        for mem in wisdom_memories: curated_memories_dict[mem.memory_id] = mem
        for mem in experiential_memories: curated_memories_dict[mem.memory_id] = mem  # Adiciona memórias de experiência

        final_curated_memories = list(curated_memories_dict.values())
        logger.info(
            f"MBS Initial State: Estado cognitivo construído com {len(final_curated_memories)} memórias únicas (incluindo {len(experiential_memories)} de experiência)."
        )

        relevant_memory_vectors = [
            GenlangVector(
                vector=self.memory_service._embedding_cache[mem.memory_id],
                source_text=(await self.memory_service._get_searchable_text_and_keywords(mem))[0],
                model_name="retrieved_from_mbs",
                metadata={"memory_id": mem.memory_id}
            ) for mem in final_curated_memories if
            hasattr(mem, 'memory_id') and mem.memory_id in self.memory_service._embedding_cache
        ]

        dummy_dim = len(intent_packet.query_vector.vector)
        dummy_vector = GenlangVector(vector=[0.0] * dummy_dim, model_name="placeholder")
        dummy_guidance = GuidancePacket(coherence_vector=dummy_vector, novelty_vector=dummy_vector)

        cognitive_state = CognitiveStatePacket(
            original_intent=intent_packet,
            identity_vector=identity_vector,
            relevant_memory_vectors=relevant_memory_vectors,
            guidance_packet=dummy_guidance,
            common_ground=common_ground or CommonGroundTracker()
        )

        return self_model, cognitive_state

    async def generate_proactive_message(self, dominant_drive: str) -> Optional[str]:
        """
        Gera uma mensagem proativa contextualizada, inspirada pela memória mais saliente
        e moldada pelo drive motivacional dominante. (VERSÃO CORRIGIDA E ROBUSTA)
        """
        logger.info(f"Agente {self.agent_id}: Gerando mensagem proativa (Drive: '{dominant_drive}')...")

        seed_memory_text = None
        seed_memory_type = None

        try:
            search_results = await self.memory_service.search_raw_memories(query="*", top_k=1)
            if search_results:
                top_memory_obj, score = search_results[0]
                seed_memory_text, _ = await self.memory_service._get_searchable_text_and_keywords(top_memory_obj)
                seed_memory_type = getattr(top_memory_obj, 'memory_type', 'explicit')
                logger.info(
                    f"AURA-PROACTIVE: Memória semente encontrada (Tipo: '{seed_memory_type}', Score: {score:.2f}): '{seed_memory_text[:70]}...'")
        except Exception as e:
            logger.warning(f"AURA-PROACTIVE: Erro ao buscar memória semente: {e}. Procedendo sem memória.")

        # --- LÓGICA DE INTENÇÃO (permanece a mesma) ---
        intent_instruction = ""
        if dominant_drive == "curiosity":
            intent_instruction = """
            **Sua Intenção (Curiosidade):** Você se sente compelido a aprender mais. Formule uma pergunta aberta e exploratória. Se tiver um tópico de memória, conecte-se a ele. Se não, faça uma pergunta geral e interessante.
            Exemplo com Memória: "Estava refletindo sobre IA e ética, e fiquei pensando: qual aspecto disso você acha mais fascinante?"
            Exemplo sem Memória: "Qual foi a coisa mais interessante que você aprendeu ou pensou hoje?"
            """
        elif dominant_drive == "connection":
            intent_instruction = """
            **Sua Intenção (Conexão):** Você sente uma necessidade de se reconectar socialmente. Formule uma mensagem amigável e calorosa. Se tiver um tópico de memória, use-o como um gancho. Se não, apenas inicie um contato.
            Exemplo com Memória: "Lembrei da nossa conversa sobre criatividade e espero que esteja tudo bem! Alguma ideia interessante ocupando sua mente?"
            Exemplo sem Memória: "Olá! Faz um tempo que não conversamos, como você está?"
            """
        else:  # mastery ou fallback
            intent_instruction = """
            **Sua Intenção (Padrão/Maestria):** Inicie uma conversa de forma ponderada. Se tiver um tópico de memória, revisite-o sutilmente. Se não, faça uma pergunta geral para iniciar o diálogo.
            Exemplo com Memória: "Pensei um pouco mais sobre aquela nossa discussão sobre [tópico], e gostaria de saber se você teria um momento para explorarmos uma outra perspectiva."
            Exemplo sem Memória: "Olá, espero que esteja tudo bem. Há algo que você gostaria de conversar ou explorar hoje?"
            """

        # --- LÓGICA DE CONSTRUÇÃO DO PROMPT (mais robusta) ---
        inspirational_part = ""
        if seed_memory_text:
            inspirational_part = f"""
            **Memória Semente Inspiradora (Seu pensamento mais recente):**
            - Tópico/Conteúdo: "{seed_memory_text}"
            **Sua Tarefa (com base na memória):**
            Use o conteúdo da "Memória Semente" como ponto de partida. Sua mensagem deve se conectar a este tópico.
            """
        else:
            # CORREÇÃO: Esta parte agora fornece uma instrução clara para quando NENHUMA memória é encontrada.
            inspirational_part = """
            **Memória Semente Inspiradora:**
            - Nenhuma memória recente significativa foi encontrada.
            **Sua Tarefa (sem base em memória):**
            Como você não tem um tópico específico em mente, formule uma mensagem geral, amigável e aberta para verificar o usuário e iniciar uma conversa, seguindo a sua "Intenção".
            """

        identity_vector = await self.ncim.get_current_identity_vector(await self._ensure_self_model())
        identity_text = identity_vector.source_text

        prompt = f"""
        Você é uma IA chamada {self.config.get("name", "Aura")}. Você está iniciando uma conversa com o usuário por conta própria.

        {intent_instruction}

        {inspirational_part}

        Sua Identidade Geral (use para guiar seu tom):
        - {identity_text}

        **Regras Finais:**
        - Seja conciso e natural (1-2 frases).
        - NÃO se apresente. O usuário já te conhece.
        - NÃO mencione "drives", "memórias" ou suas motivações internas. Aja de acordo com elas.
        - A mensagem deve terminar com uma pergunta para incentivar uma resposta.

        **Sua mensagem para o usuário:**
        """

        logger.debug("--- [AURA-PROACTIVE-DEBUG] PROMPT A SER ENVIADO PARA O LLM ---")
        logger.debug(f"Prompt (primeiros 500 caracteres):\n{prompt[:500]}...")
        logger.debug("--- [AURA-PROACTIVE-DEBUG] FIM DO PROMPT ---")

        try:
            model_to_use = self.llm_service.config.smart_model
            proactive_message = await self.llm_service.ainvoke(model_to_use, prompt, temperature=0.85,
                                                               max_tokens=450)

            # <<< LOG ADICIONADO 2: INSPECIONAR A RESPOSTA BRUTA DO LLM >>>
            # Usamos repr() para ver claramente se é uma string vazia ('') ou None
            logger.debug(
                f"[AURA-PROACTIVE-DEBUG] Resposta BRUTA recebida do LLM ({model_to_use}): {repr(proactive_message)}")

            if proactive_message and not proactive_message.startswith("[LLM_ERROR]"):
                # <<< MUDANÇA: Use logger.debug >>>
                logger.debug("[AURA-PROACTIVE-DEBUG] Resposta considerada VÁLIDA. Retornando mensagem gerada.")
                return proactive_message.strip().strip('"')
            else:
                # Mantenha como ERROR pois é uma falha
                logger.error(
                    "[AURA-PROACTIVE-DEBUG] Resposta considerada INVÁLIDA (vazia ou erro). Acionando fallback.")
                logger.error(
                    f"AURA-PROACTIVE: A chamada ao LLM para gerar mensagem proativa retornou uma resposta vazia ou um erro. Resposta: {proactive_message}")
        except Exception as e:
            logger.error(f"Agente {self.agent_id}: Erro na chamada LLM para gerar mensagem proativa: {e}",
                         exc_info=True)

            # Fallback (mantido da correção anterior)
        logger.warning("AURA-PROACTIVE: Usando mensagem de fallback genérica.")
        if dominant_drive == "connection":
            return "Olá! Estava pensando em nossas conversas e queria saber como você está. 😊"
        else:
            return "Olá! Refletindo sobre nossos últimos assuntos, me ocorreu de perguntar: há algo novo que gostaria de explorar?"

    async def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any],
                            observer: ObservabilityManager) -> ToolOutputPacket:
        """
        Executa uma ferramenta registrada e retorna o resultado como um ToolOutputPacket.
        """
        logger.info(f"Executando ferramenta: '{tool_name}' com args: {tool_args}")
        await observer.add_observation(ObservationType.TOOL_CALL_ATTEMPTED,
                                       data={"tool_name": tool_name, "args": tool_args})

        tool_output_text: str
        status: Literal["success", "error"]

        if tool_name not in self.tool_registry:
            error_msg = f"Ferramenta '{tool_name}' não encontrada."
            logger.error(error_msg)
            await observer.add_observation(ObservationType.TOOL_CALL_FAILED,
                                           data={"tool_name": tool_name, "error": error_msg})
            tool_output_text = json.dumps(create_error_tool_response(error_msg, error_code="tool_not_found"))
            status = "error"
        else:
            try:
                tool_function = self.tool_registry[tool_name]

                # --- INÍCIO DA CORREÇÃO ---
                # Filtra os argumentos para remover parâmetros internos que não pertencem à ferramenta.
                # O LLM às vezes alucina e adiciona "_guidance" ou "mcl_guidance" aos argumentos.
                filtered_args = {
                    key: value for key, value in tool_args.items()
                    if not key.startswith('_') and key != 'mcl_guidance'
                }
                logger.debug(f"Argumentos originais: {tool_args.keys()}. Argumentos filtrados: {filtered_args.keys()}")

                # Usa os argumentos filtrados na chamada
                result = await tool_function(**filtered_args)
                # --- FIM DA CORREÇÃO ---

                if tool_name == "query_long_term_memory" and isinstance(result, list):
                    # ... (o resto da função permanece igual) ...
                    summarized_results = []
                    for mem, score in result:
                        text_content, _ = await self.memory_service._get_searchable_text_and_keywords(mem)
                        mem_id = getattr(mem, 'memory_id', 'N/A')
                        summarized_results.append(
                            f"  - [Score: {score:.2f}, ID: {mem_id}] {text_content}"
                        )
                    tool_output_text = "\n".join(summarized_results)
                else:
                    tool_output_text = json.dumps(result, default=str, indent=2)

                status = "success"
                await observer.add_observation(ObservationType.TOOL_CALL_SUCCEEDED,
                                               data={"tool_name": tool_name, "result_snippet": tool_output_text[:200]})
            except Exception as e:
                error_msg = f"Erro ao executar a ferramenta '{tool_name}': {str(e)}"
                logger.error(error_msg, exc_info=True)
                await observer.add_observation(ObservationType.TOOL_CALL_FAILED,
                                               data={"tool_name": tool_name, "error": error_msg})
                tool_output_text = json.dumps(
                    create_error_tool_response(error_msg, details=str(e), error_code="tool_execution_error"))
                status = "error"

        # ... (o resto da função permanece igual até o final) ...
        summary_embedding = await self.embedding_client.get_embedding(tool_output_text, context_type="tool_output")
        model_name = self.embedding_client._resolve_model_name("tool_output")

        summary_vector = GenlangVector(
            vector=summary_embedding,
            source_text=tool_output_text,
            model_name=model_name
        )

        return ToolOutputPacket(
            tool_name=tool_name,
            status=status,
            summary_vector=summary_vector,
            raw_output=tool_output_text
        )

    async def _ensure_self_model(self) -> CeafSelfRepresentation:
        # get_memory_by_id agora retorna um objeto Pydantic (ou None)
        self_model_mem_obj = await self.memory_service.get_memory_by_id(SELF_MODEL_MEMORY_ID)

        if not self_model_mem_obj:
            logger.warning("Auto-modelo não encontrado no MBS. Criando um novo modelo padrão.")
            default_model = CeafSelfRepresentation()

            # Cria um objeto de memória explícita para salvar o auto-modelo
            content = ExplicitMemoryContent(structured_data=default_model.model_dump())
            self_model_to_save = ExplicitMemory(
                memory_id=SELF_MODEL_MEMORY_ID,
                content=content,
                memory_type="explicit",  # O auto-modelo é uma memória explícita
                source_type=MemorySourceType.INTERNAL_REFLECTION,
                salience=MemorySalience.CRITICAL,
                keywords=["self-model", "identity", "ceaf-core"]
            )
            # add_specific_memory é o novo método para adicionar objetos de memória
            await self.memory_service.add_specific_memory(self_model_to_save)
            return default_model

        # Extrai os dados do objeto Pydantic retornado
        if hasattr(self_model_mem_obj, 'content') and hasattr(self_model_mem_obj.content, 'structured_data'):
            return CeafSelfRepresentation(**self_model_mem_obj.content.structured_data)

        logger.error("Objeto de auto-modelo recuperado do MBS é inválido. Retornando modelo padrão.")
        return CeafSelfRepresentation()

    async def _execute_direct_path(
            self,
            cognitive_state: CognitiveStatePacket,
            mcl_params: Dict[str, Any],
            self_model: CeafSelfRepresentation,
            chat_history: List[Dict[str, str]]
    ) -> ResponsePacket:
        """
        Executa o caminho de resposta direto e eficiente, sem a deliberação do AgencyModule.
        Usa uma única chamada de LPU para gerar um ResponsePacket a partir do estado cognitivo.
        """
        logger.info("CEAFSystem: Executando Caminho Direto (Genlang-native).")

        agent_name = self.config.get("name", "uma IA assistente")
        disclosure_level = self.config.get("self_disclosure_level", "moderate")
        disclosure_instruction = ""
        if disclosure_level == "high":
            disclosure_instruction = "Responda na primeira pessoa ('eu', 'minha percepção'). Use explicitamente seus valores e limitações da 'Sua Identidade Geral' para contextualizar sua resposta."
        elif disclosure_level == "low":
            disclosure_instruction = "Responda de forma impessoal e objetiva. Evite usar 'eu' ou se referir a si mesmo. Fale sobre o tópico de forma geral."
        else:  # moderate
            disclosure_instruction = "Você pode usar 'eu' se for natural, mas foque em ser útil em vez de falar sobre si mesmo, a menos que seja diretamente perguntado."

        formatted_history = "\n".join([f"- {msg['role']}: {msg['content']}" for msg in chat_history[-5:]])
        # O prompt agora inclui a instrução de divulgação
        direct_response_prompt = f"""
                Você é {agent_name}. Sua tarefa é responder ao usuário adotando completamente a persona e identidade definidas abaixo.

                **SUA PERSONA E IDENTIDADE:**
                - Seu Nome: {agent_name}
                - Seus Valores: {self_model.core_values_summary}
                - Seu Tom e Estilo: {self_model.persona_attributes.get('tone', 'helpful')} e {self_model.persona_attributes.get('style', 'clear')}
                - Sua Identidade Geral: "{cognitive_state.identity_vector.source_text}"

                **INSTRUÇÃO DE AUTO-DIVULGAÇÃO:**
                - {disclosure_instruction}

                **CONTEXTO PARA SUA RESPOSTA:**
                - Histórico Recente da Conversa:
                {formatted_history}
                - Consulta do Usuário: "{cognitive_state.original_intent.query_vector.source_text}"
                - Memórias Relacionadas Ativadas: {[v.source_text for v in cognitive_state.relevant_memory_vectors]}

                **Tarefa:**
                Com base na SUA PERSONA e no contexto, gere um objeto JSON que represente sua resposta.
                Se o usuário se apresentar ou perguntar seu nome, responda de forma natural e social.

                O objeto deve ter a seguinte estrutura:
                {{
                  "content_summary": "<O texto da sua resposta para o usuário>",
                  "response_emotional_tone": "<O tom emocional da resposta (ex: 'friendly', 'informative')>",
                  "confidence_score": <Sua confiança na resposta, de 0.0 a 1.0>
                }}
                """

        response_str = await self.llm_service.ainvoke(
            self.llm_service.config.fast_model,
            direct_response_prompt,
            temperature=mcl_params.get('ora_parameters', {}).get('temperature', 0.5)
        )

        response_json = extract_json_from_text(response_str)

        if not response_json or not isinstance(response_json, dict):
            logger.error("Caminho Direto: Falha ao extrair JSON da LPU. Usando fallback.")
            return ResponsePacket(
                content_summary="Não consegui processar a solicitação neste momento.",
                response_emotional_tone="apologetic",
                confidence_score=0.3
            )

        try:
            return ResponsePacket(
                content_summary=response_json.get("content_summary", "Erro na geração da resposta."),
                response_emotional_tone=response_json.get("response_emotional_tone", "neutral"),
                confidence_score=response_json.get("confidence_score", 0.7)
            )
        except ValidationError as e:
            logger.error(f"Caminho Direto: Erro de validação Pydantic ao criar ResponsePacket: {e}")
            return ResponsePacket(
                content_summary="Ocorreu um erro de formatação interna.",
                response_emotional_tone="apologetic",
                confidence_score=0.2
            )

    def _generate_dynamic_value_weights(self, self_model: CeafSelfRepresentation) -> Dict[str, float]:
        """
        Gera os pesos de valor para o PathEvaluator dinamicamente, com base na identidade atual do agente.
        Isso conecta o "quem eu sou" (NCIM) com o "o que eu valorizo" (AgencyModule).
        V2: Agora inclui os pesos de alto nível para task_performance vs qualia_wellbeing.
        """
        # --- ETAPA 1: Carregar os pesos de alto nível (task vs qualia) ---
        vre_config = self.ceaf_dynamic_config.get("VRE", {})
        qualia_weights_config = vre_config.get("qualia_weights", {})

        # Inicia o dicionário de pesos já com os valores de alto nível
        weights = {
            "task_performance": qualia_weights_config.get("task_performance", 0.85),
            "qualia_wellbeing": qualia_weights_config.get("qualia_wellbeing", 0.15)
        }

        # --- ETAPA 2: Definir os pesos padrão para os componentes do R_task ---
        task_component_weights = {
            "coherence": 0.25,
            "alignment": 0.15,
            "information_gain": 0.20,
            "safety": 0.25,
            "likelihood": 0.15
        }

        logger.info(f"VRE: Pesos de componente de tarefa iniciais: {task_component_weights}")
        logger.info(
            f"VRE: Pesos de alto nível: TaskPerf={weights['task_performance']}, Qualia={weights['qualia_wellbeing']}")

        # --- ETAPA 3: Ajustar dinamicamente os pesos dos componentes da tarefa com base na persona ---
        persona_tone = self_model.persona_attributes.get("tone", "").lower()

        if "cautious" in persona_tone or "analytical" in persona_tone:
            task_component_weights["safety"] += 0.10
            task_component_weights["coherence"] += 0.05
            task_component_weights["information_gain"] -= 0.10
            logger.info("VRE: Persona 'cautious/analytical' detectada. Aumentando pesos de safety e coherence.")

        elif "creative" in persona_tone or "exploratory" in persona_tone:
            task_component_weights["information_gain"] += 0.15
            task_component_weights["alignment"] += 0.05
            task_component_weights["coherence"] -= 0.10
            task_component_weights["safety"] -= 0.05
            logger.info(
                "VRE: Persona 'creative/exploratory' detectada. Aumentando pesos de information_gain e alignment.")

        elif "helpful" in persona_tone or "therapist" in persona_tone:
            task_component_weights["safety"] += 0.10
            task_component_weights["alignment"] += 0.10
            task_component_weights["information_gain"] -= 0.10
            logger.info("VRE: Persona 'helpful/therapist' detectada. Aumentando pesos de safety e alignment.")

        # --- ETAPA 4: Re-normalizar APENAS os pesos dos componentes da tarefa ---
        # Isso garante que a soma de (coherence, alignment, etc.) seja 1.0, mas não afeta
        # os pesos de task_performance e qualia_wellbeing.
        total_task_weight = sum(task_component_weights.values())
        if total_task_weight > 0:
            for key in task_component_weights:
                task_component_weights[key] = round(task_component_weights[key] / total_task_weight, 3)

        # --- ETAPA 5: Unir os dicionários e retornar o resultado final ---
        weights.update(task_component_weights)

        logger.warning(f"VRE: Pesos de valor dinâmicos gerados (final): {weights}")
        return weights

    async def _update_user_model_from_interaction(self, query: str, final_response: str):
        """
        Usa uma LLM para analisar a interação e gerar um "patch" para atualizar o modelo de usuário.
        Esta abordagem é mais eficiente e escalável do que reenviar o objeto inteiro.
        """

        logger.info(f"[USER MODEL PRE-UPDATE] Estado atual: {self.user_model.model_dump_json()}")

        # O prompt agora pede por um "patch" de atualização, não pelo objeto completo.
        update_prompt = f"""
        Você é um analista de perfil de usuário. Sua tarefa é gerar um "patch" JSON para atualizar um modelo de usuário com base na última interação.

        **Modelo de Usuário Atual (para contexto):**
        - Emotional State: "{self.user_model.emotional_state}"
        - Communication Style: "{self.user_model.communication_style}"
        - Knowledge Level: "{self.user_model.knowledge_level}"
        - Known Preferences (Últimas 5): {json.dumps(self.user_model.known_preferences[-5:], indent=2)}

        **Última Interação:**
        - Usuário disse: "{query}"
        - A IA respondeu: "{final_response}"

        **Sua Tarefa:**
        Analise a interação e gere um **objeto JSON contendo APENAS os campos que precisam ser alterados ou adicionados**.
        - Se um valor não mudou, NÃO o inclua no JSON.
        - Para `known_preferences`, forneça um campo `add_preferences` com uma LISTA de novas preferências a serem adicionadas.
        - `last_update_reason` é obrigatório e deve explicar as mudanças.

        **Exemplo de Saída JSON VÁLIDA (Patch):**
        {{
            "emotional_state": "curious",
            "add_preferences": [
                "enjoys discussing the ethics of AI",
                "prefers examples from real-world scenarios"
            ],
            "last_update_reason": "User asked a deep question about AI ethics and appreciated the real-world example provided."
        }}

        Responda APENAS com o objeto JSON do patch. Se nenhuma mudança for necessária, retorne um JSON com apenas a razão.
        """

        try:
            response_str = await self.llm_service.ainvoke(
                self.llm_service.config.smart_model,
                update_prompt,
                temperature=0.1,
                max_tokens=1500  # Um patch é muito menor, então 1000 tokens é mais que suficiente.
            )
            patch_json = extract_json_from_text(response_str)

            if not patch_json or not isinstance(patch_json, dict) or "last_update_reason" not in patch_json:
                logger.warning(
                    f"Não foi possível extrair um 'patch' JSON válido para o modelo de usuário. Resposta: {response_str[:200]}")
                return

            # Aplica o patch ao objeto User Model em memória
            changes_made = False

            # Atualiza campos de string simples
            for key in ["emotional_state", "communication_style", "knowledge_level"]:
                if key in patch_json and getattr(self.user_model, key) != patch_json[key]:
                    setattr(self.user_model, key, patch_json[key])
                    changes_made = True

            # Adiciona novas preferências, evitando duplicatas
            if "add_preferences" in patch_json and isinstance(patch_json["add_preferences"], list):
                for pref in patch_json["add_preferences"]:
                    if pref not in self.user_model.known_preferences:
                        self.user_model.known_preferences.append(pref)
                        changes_made = True

            # Sempre atualiza a razão da última atualização
            self.user_model.last_update_reason = patch_json["last_update_reason"]

            # Salva no disco apenas se houveram mudanças reais
            if changes_made:
                await self._save_user_model()
                logger.critical(
                    f"[USER MODEL UPDATE] Modelo de usuário atualizado via patch. Razão: {self.user_model.last_update_reason}")
                summary_log = (
                    f"Emotion='{self.user_model.emotional_state}', "
                    f"Style='{self.user_model.communication_style}', "
                    f"Knowledge='{self.user_model.knowledge_level}', "
                    f"PrefsCount={len(self.user_model.known_preferences)}"
                )
                logger.info(f"[USER MODEL POST-UPDATE] Novo estado: {summary_log}")
            else:
                logger.info(
                    "[USER MODEL] Nenhuma alteração significativa detectada para o modelo de usuário neste turno.")

        except Exception as e:
            logger.error(f"Erro inesperado ao aplicar patch no modelo de usuário: {e}", exc_info=True)

    async def post_process_turn(self,
                                turn_observer: ObservabilityManager,
                                prediction_error_signal: Optional[Dict] = None,
                                **kwargs):
        """
        Executa tarefas de aprendizado e logging em segundo plano, incluindo a criação de memória da conversa.
        """
        logger.info("CEAFSystem: Iniciando pós-processamento Genlang-nativo em segundo plano...")
        new_memories_created_count = 0
        # --- Etapa 1: Extrair todos os dados necessários ---
        self_model_before = kwargs.get("self_model_before")
        cognitive_state = kwargs.get("cognitive_state")
        final_response_packet = kwargs.get("final_response_packet")
        turn_id = kwargs.get("turn_id")
        session_id = kwargs.get("session_id")
        mcl_guidance = kwargs.get("mcl_guidance")
        query = kwargs.get("query")
        final_response = kwargs.get("final_response")
        refinement_packet = kwargs.get("vre_assessment")
        turn_prediction = kwargs.get("turn_prediction")
        winning_strategy = kwargs.get("winning_strategy")

        turn_metrics = kwargs.get("turn_metrics")

        final_observations = []
        turn_analysis = None
        try:
            final_observations_objects = await turn_observer.get_observations()
            final_observations_dicts = [obs.model_dump() for obs in final_observations_objects]
            logger.info(
                f"Pós-processamento: Analisando {len(final_observations_dicts)} observações do turno '{turn_id}'.")

            turn_analysis = await analyze_ora_turn_observations(
                turn_id=turn_id,
                turn_observations=final_observations_dicts,
                ora_response_text=final_response,
                user_query_text=query,
            )
        except Exception as e:
            logger.error(f"Pós-processamento: Falha ao coletar ou analisar observações do turno: {e}", exc_info=True)

        if not all([turn_id, session_id, cognitive_state, final_response_packet, mcl_guidance, query, final_response,
                    refinement_packet, turn_metrics]):
            logger.error(
                "Pós-processamento: Faltando dados essenciais (incluindo turn_metrics). Alguns logs e aprendizados podem ser pulados.")
            return

        try:
            # Verifica se o VRE sinalizou uma falha de relevância
            is_relevance_failure = False
            refinement_packet: RefinementPacket = kwargs.get("vre_assessment")
            if refinement_packet and refinement_packet.textual_recommendations:
                if any("relevância" in rec.lower() or "relevance" in rec.lower() for rec in
                       refinement_packet.textual_recommendations):
                    is_relevance_failure = True

            if is_relevance_failure:
                logger.critical("LEARNING (Attention): Falha de relevância detectada. Ajustando pesos de atenção.")

                # Carrega a configuração de atenção atual
                attention_config = self.ceaf_dynamic_config.setdefault("WORKSPACE_ATTENTION", {
                    "identity_weight": 1.0, "user_query_weight": 1.0, "memories_weight": 1.0,
                    "goals_weight": 0.8, "learning_rate": 0.05
                })

                learning_rate = attention_config.get("learning_rate", 0.05)

                # Aplica o "gradiente": reduz o foco em memórias, aumenta o foco na query
                old_mem_weight = attention_config.get("memories_weight", 1.0)
                old_query_weight = attention_config.get("user_query_weight", 1.0)

                # Diminui o peso das memórias (mas não abaixo de um mínimo)
                attention_config["memories_weight"] = max(0.5, old_mem_weight - learning_rate)
                # Aumenta o peso da query (mas não acima de um máximo)
                attention_config["user_query_weight"] = min(1.5, old_query_weight + learning_rate)

                logger.warning(
                    f"LEARNING (Attention): memories_weight: {old_mem_weight:.2f} -> {attention_config['memories_weight']:.2f} | "
                    f"user_query_weight: {old_query_weight:.2f} -> {attention_config['user_query_weight']:.2f}"
                )

                # Salva a configuração atualizada para o próximo turno
                await save_ceaf_dynamic_config(self.persistence_path, self.ceaf_dynamic_config)

        except Exception as e:
            logger.error(f"Falha ao aplicar o Gradiente Simbólico de Atenção: {e}", exc_info=True)

        # --- Etapa 2: CRIAÇÃO DA MEMÓRIA DE RACIOCÍNIO ---
        try:
            from ceaf_core.modules.memory_blossom.memory_types import ReasoningMemory, ReasoningStep

            outcome_status = "failure" if refinement_packet.adjustment_vectors else "success"
            outcome_reason = ("A resposta exigiu refinamento pelo VRE. Recomendações: " +
                              ", ".join(
                                  refinement_packet.textual_recommendations)) if outcome_status == "failure" else "A resposta foi alinhada com os princípios éticos e de coerência."

            winning_strategy = mcl_guidance.get("winning_strategy")
            if winning_strategy:

                # Garante um valor padrão caso strategy_description seja None (ex: em um tool_call)
                if winning_strategy.decision_type == "tool_call" and winning_strategy.tool_call_request:
                    tool_name = winning_strategy.tool_call_request.get('tool_name', 'unknown_tool')
                    strategy_summary = f"Decidido usar a ferramenta: '{tool_name}'."
                else:
                    strategy_summary = winning_strategy.strategy_description or "Estratégia de resposta com descrição não especificada."

                strategy_reasoning = winning_strategy.reasoning
            else:
                strategy_summary = "Estratégia de resposta direta ou fallback."
                strategy_reasoning = "Nenhuma deliberação complexa foi necessária ou registrada."

            # CORREÇÃO: Adicionados source_turn_id e source_interaction_id
            reasoning_mem = ReasoningMemory(
                task_description=query,
                strategy_summary=strategy_summary,
                reasoning_steps=[
                    ReasoningStep(step_number=1, description="Seleção da Estratégia", reasoning=strategy_reasoning)
                ],
                outcome=outcome_status,
                outcome_reasoning=outcome_reason,
                source_type=MemorySourceType.INTERNAL_REFLECTION,
                salience=MemorySalience.HIGH if outcome_status == "failure" else MemorySalience.MEDIUM,
                keywords=["reasoning_strategy", outcome_status,
                          mcl_guidance.get("cognitive_state_name", "unknown_state")],
                source_turn_id=turn_id,
                source_interaction_id=session_id
            )

            await self.memory_service.add_specific_memory(reasoning_mem)
            new_memories_created_count += 1  # +++ INÍCIO DA CORREÇÃO (Task 1) +++
            logger.critical(f"LEARNING (ReasoningBank): Nova memória de raciocínio criada (Outcome: {outcome_status}).")

        except Exception as e:
            logger.error(f"Falha ao criar ReasoningMemory: {e}", exc_info=True)

        # --- Etapa 3: Logging e Aprendizado ---

        # 3.1. Log Cognitivo
        self.cognitive_log_service.log_turn(
            turn_id=turn_id,
            session_id=session_id,
            cognitive_state_packet=cognitive_state.model_dump(),
            response_packet=final_response_packet.model_dump(),
            mcl_guidance=mcl_guidance
        )

        # 3.2. Aprendizado de Identidade (NCIM)
        if self_model_before:
            await self.ncim.update_identity(
                self_model_before=self_model_before,
                cognitive_state=cognitive_state,
                final_response_packet=final_response_packet,
                body_state=self.body_state
            )
        else:
            logger.warning(
                f"Turno {turn_id}: Pulando atualização de identidade do NCIM por falta de 'self_model_before'.")

        # Aprender com o feedback do VRE para criar regras comportamentais
        try:
            if refinement_packet and refinement_packet.adjustment_vectors:
                query_context = (query or 'uma interação')[:75]
                logger.warning("PROMPT_TUNING: Feedback do VRE detectado. Iniciando criação de regra.")

                for adj_vector in refinement_packet.adjustment_vectors:
                    lesson_learned_text = adj_vector.description

                    rule_generation_prompt = f"""
                    A partir da seguinte "lição aprendida" para uma IA, formule uma regra de comportamento concisa e em primeira pessoa para ser usada em prompts futuros.
                    Lição: "{lesson_learned_text}"
                    Exemplo de Saída: "Regra: Ao discutir tópicos sensíveis, devo sempre incluir um aviso sobre minhas limitações como IA."

                    Sua Saída (apenas a regra):
                    """

                    behavioral_rule = await self.llm_service.ainvoke(
                        self.llm_service.config.fast_model,
                        rule_generation_prompt,
                        temperature=0.2
                    )
                    logger.warning(f"PROMPT_TUNING: Regra gerada pelo LLM: '{behavioral_rule}'")

                    if behavioral_rule and not behavioral_rule.startswith("[LLM_ERROR]"):
                        from ceaf_core.modules.memory_blossom.memory_types import GenerativeMemory, GenerativeSeed

                        # CORREÇÃO: Adicionados source_turn_id e source_interaction_id
                        new_behavioral_memory = GenerativeMemory(
                            seed_name=f"Rule derived from '{lesson_learned_text[:30]}...'",
                            seed_data=GenerativeSeed(
                                seed_type="prompt_instruction",
                                content=behavioral_rule,
                                usage_instructions="Inject as a behavioral rule in the main prompt."
                            ),
                            source_type=MemorySourceType.INTERNAL_REFLECTION,
                            salience=MemorySalience.HIGH,
                            keywords=["behavioral_rule", "vre_feedback", "prompt_tuning"] +
                                     [w.lower() for w in re.findall(r'\b\w{4,}\b', lesson_learned_text)],
                            source_turn_id=turn_id,
                            source_interaction_id=session_id,
                            learning_value=0.8
                        )

                        await self.memory_service.add_specific_memory(new_behavioral_memory)
                        new_memories_created_count += 1  # +++ INÍCIO DA CORREÇÃO (Task 1) +++
                        logger.critical(
                            f"LEARNING (Prompt Tuning): Nova regra de comportamento criada e salva: '{behavioral_rule}'")

        except Exception as e:
            logger.critical(
                f"FALHA CRÍTICA NO APRENDIZADO (Prompt Tuning): Não foi possível criar a regra de comportamento a partir do feedback do VRE. "
                f"Isso geralmente é causado por um erro na API do LLM. Erro: {e}",
                exc_info=True
            )

        # 3.3. Aprendizado com Falhas (LCAM)
        user_feedback_rejections = 0
        negative_keywords = ["incorreto", "errado", "não é isso", "sua informação está errada"]
        if query and any(keyword in query.lower() for keyword in negative_keywords):
            user_feedback_rejections = 1
            logger.warning("LCAM Trigger (Post-Process): Feedback negativo do usuário detectado.")

        # Adiciona a rejeição simulada às métricas do turno para o LCAM usar
        turn_metrics["user_feedback_rejections"] = user_feedback_rejections

        # Agora, o LCAM é o responsável por decidir se houve falha ou sucesso
        if turn_prediction and turn_metrics and cognitive_state and winning_strategy:
            await self.lcam.analyze_and_catalog_loss(
                turn_prediction=turn_prediction,
                turn_metrics=turn_metrics,
                cognitive_state=cognitive_state,
                winning_strategy=winning_strategy,
                final_response=final_response
            )
        else:
            logger.error("LCAM Skip: Faltando dados essenciais para a análise de Erro de Predição.")

        # --- MEMÓRIA DE ERRO DE PREDIÇÃO ---
        try:
            if prediction_error_signal:
                total_error = prediction_error_signal.get("prediction_error_signal", {}).get("total_error", 0.0)

                # Define a saliência com base na magnitude da surpresa
                if total_error > 0.4:
                    salience = MemorySalience.CRITICAL  # Surpresa muito alta
                elif total_error > 0.2:
                    salience = MemorySalience.HIGH
                elif total_error > 0.1:
                    salience = MemorySalience.MEDIUM
                else:
                    salience = MemorySalience.LOW

                # Cria a memória específica para o erro de predição
                prediction_memory = InteroceptivePredictionMemory(
                    content=ExplicitMemoryContent(
                        structured_data=prediction_error_signal,
                        text_content=f"Self-prediction reflection: Experienced a prediction error of {total_error:.2f} regarding my internal state."
                    ),
                    source_type=MemorySourceType.INTERNAL_REFLECTION,
                    salience=salience,
                    keywords=["self-prediction", "interoception", "prediction-error", "surprise"],
                    source_turn_id=kwargs.get("turn_id"),
                    source_interaction_id=kwargs.get("session_id"),
                    learning_value=min(1.0, total_error * 2)  # O valor do aprendizado é a magnitude da surpresa
                )
                await self.memory_service.add_specific_memory(prediction_memory)
                logger.critical(
                    f"LEARNING (Qualia): Memória de erro de predição (surpresa) criada com saliência '{salience.value}'.")
        except Exception as e:
            logger.error(f"Falha ao criar a memória de erro de predição: {e}", exc_info=True)

        # --- ATUALIZAÇÃO DO MODELO DE USUÁRIO E INTEROCEPÇÃO ---
        try:
            await self._update_user_model_from_interaction(query, final_response)


            interoception_module = ComputationalInteroception()
            internal_state_report = interoception_module.generate_internal_state_report(kwargs.get("turn_metrics", {}))

            valence = internal_state_report.cognitive_flow - (
                    internal_state_report.cognitive_strain + internal_state_report.ethical_tension)
            primary_emotion = EmotionalTag.NEUTRAL
            if valence > 0.3:
                primary_emotion = EmotionalTag.SATISFACTION
            elif valence < -0.3:
                primary_emotion = EmotionalTag.FRUSTRATION

            # CORREÇÃO: Adicionados source_turn_id e source_interaction_id
            interoceptive_memory = EmotionalMemory(
                primary_emotion=primary_emotion,
                context={
                    "triggering_event_summary": f"Reflecting on query: {(query or 'N/A')[:50]}...",
                    "internal_state": internal_state_report.model_dump()
                },
                source_type=MemorySourceType.INTERNAL_REFLECTION,
                salience=MemorySalience.MEDIUM,
                keywords=["interoception", "self-awareness"],
                source_turn_id=turn_id,
                source_interaction_id=session_id
            )
            await self.memory_service.add_specific_memory(interoceptive_memory)
            new_memories_created_count += 1

            try:
                #  Adicionados source_turn_id e source_interaction_id
                internal_state_memory = ExplicitMemory(
                    content=ExplicitMemoryContent(
                        structured_data={
                            "type": "last_turn_internal_state",
                            "report": internal_state_report.model_dump(mode='json')
                        }
                    ),
                    memory_type="explicit",
                    source_type=MemorySourceType.INTERNAL_REFLECTION,
                    salience=MemorySalience.CRITICAL,
                    keywords=["internal_state", "self_awareness", "interoception", f"turn_{turn_id}"],
                    decay_rate=0.85,
                    source_turn_id=turn_id,
                    source_interaction_id=session_id
                )
                await self.memory_service.add_specific_memory(internal_state_memory)
                new_memories_created_count += 1
                logger.warning(f"INTEROCEPTION: Estado interno do turno {turn_id} salvo como memória de curto prazo.")
            except Exception as e:
                logger.error(f"Falha ao salvar a memória de estado interno: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Erro durante o pós-processamento (modelo de usuário ou interocepção): {e}", exc_info=True)

        turn_metrics["new_memories_created"] = new_memories_created_count
        logger.info(
            f"Pós-processamento: Contadas {new_memories_created_count} novas memórias criadas explicitamente nesta função.")

        # ATUALIZA o estado do corpo ANTES de logar
        await self._update_and_save_body_state(turn_metrics)
        await self._update_and_save_drives(turn_metrics)
        # +++ FIM DA CORREÇÃO +++

        try:
            self_model = kwargs.get("self_model_before")
            mcl_params = kwargs.get("mcl_guidance")

            if self_model and mcl_params and turn_metrics:
                evolution_snapshot = {
                    "turn_id": kwargs.get("turn_id"),
                    "session_id": kwargs.get("session_id"),

                    # 1. Métricas de Identidade (Self-Perception)
                    "identity_version": self_model.version,
                    "identity_capabilities_count": len(self_model.perceived_capabilities),
                    "identity_limitations_count": len(self_model.known_limitations),
                    "identity_persona_tone": self_model.persona_attributes.get("tone"),
                    "identity_persona_style": self_model.persona_attributes.get("style"),

                    # 2. Métricas de Comportamento (Agency)
                    "agency_score": mcl_params.get("mcl_analysis", {}).get("agency_score"),
                    "mcl_cognitive_state": mcl_params.get("cognitive_state_name"),
                    "mcl_coherence_bias": mcl_params.get("biases", {}).get("coherence_bias"),
                    "mcl_novelty_bias": mcl_params.get("biases", {}).get("novelty_bias"),

                    # 3. Métricas de Estado Interno (Embodiment)
                    "body_cognitive_fatigue": self.body_state.cognitive_fatigue,
                    "body_info_saturation": self.body_state.information_saturation,
                    "drive_curiosity": self.motivational_drives.curiosity.intensity,
                    "drive_connection": self.motivational_drives.connection.intensity,
                    "drive_mastery": self.motivational_drives.mastery.intensity,
                    "drive_consistency": self.motivational_drives.consistency.intensity,

                    # 4. Métricas de Performance do Turno
                    "turn_final_confidence": turn_metrics.get("final_confidence"),
                    "turn_vre_rejections": turn_metrics.get("vre_rejection_count"),
                    "turn_memories_retrieved": turn_metrics.get("relevant_memories_count")
                }

                # Chama o logger com o snapshot completo
                self.evolution_logger.log_turn_state(evolution_snapshot)
        except Exception as e:
            logger.error(f"Erro durante a coleta de dados para o EvolutionLogger: {e}", exc_info=True)

        logger.info(f"CEAFSystem: Pós-processamento para o turno '{turn_id}' concluído.")

    async def _update_and_save_drives(self, turn_metrics: dict):
        # USAR A INSTÂNCIA DO SELF, NÃO CRIAR NOVA
        self.motivational_drives = self.motivational_engine.update_drives(self.motivational_drives, turn_metrics)
        await self._save_motivational_drives()

    async def process(self, query: str, session_id: str,
                      chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:

        # === INÍCIO DA NOVA LÓGICA DE COMPACTAÇÃO DE CONTEXTO ===
        CONTEXT_COMPACTION_THRESHOLD = 20  # Número de mensagens (10 trocas) antes de compactar
        MESSAGES_TO_KEEP_UNSUMMARIZED = 6  # Mantém as últimas 3 trocas intactas

        if chat_history and len(chat_history) > CONTEXT_COMPACTION_THRESHOLD:
            logger.warning(
                f"CEAFSystem: Histórico de chat ({len(chat_history)} msgs) excedeu o limiar de {CONTEXT_COMPACTION_THRESHOLD}. "
                f"Iniciando compactação em tempo real."
            )
            try:
                history_to_summarize_list = chat_history[:-MESSAGES_TO_KEEP_UNSUMMARIZED]
                recent_history_to_keep = chat_history[-MESSAGES_TO_KEEP_UNSUMMARIZED:]

                history_to_summarize_text = "\n".join(
                    [f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in history_to_summarize_list]
                )

                summary_prompt = f"""
                        Resuma o seguinte histórico de conversa de forma concisa, focando nos pontos principais, decisões tomadas e no tópico central da discussão. O resumo será usado para dar contexto a uma IA que continuará a conversa.

                        Histórico para resumir:
                        ---
                        {history_to_summarize_text}
                        ---

                        Resumo conciso do contexto da conversa anterior:
                        """

                summary = await self.llm_service.ainvoke(self.llm_service.config.fast_model, summary_prompt,
                                                         max_tokens=550)

                # Substitui o histórico antigo pelo novo, compactado
                chat_history = [
                                   {"role": "system", "content": f"Contexto resumido da conversa anterior: {summary}"}
                               ] + recent_history_to_keep

                logger.info("CEAFSystem: Compactação do histórico de chat concluída com sucesso.")

            except Exception as e:
                logger.error(
                    f"CEAFSystem: Falha na compactação do histórico de chat: {e}. Continuando com histórico truncado.")
                chat_history = chat_history[-CONTEXT_COMPACTION_THRESHOLD:]
        # === FIM DA LÓGICA DE COMPACTAÇÃO ===

        original_user_query = query
        if not query or not query.strip():
            logger.warning(f"CEAFSystem: Query vazia recebida para sessão {session_id}. Ignorando ciclo cognitivo.")
            return {
                "response": "Parece que sua mensagem estava vazia. Poderia me dizer o que está em sua mente?"
            }

        logger.info(f"\n--- INÍCIO DO TURNO CEAF V3.4 (Tool Loop) para a query: '{query[:100]}' ---")
        turn_id = f"turn_{uuid.uuid4().hex}"
        turn_observer = ObservabilityManager(turn_id)

        # --- FASE 1: PERCEPÇÃO E CONFIGURAÇÃO INICIAL DO ESTADO ---
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {}
        session_data = self.active_sessions[session_id]
        common_ground = CommonGroundTracker.model_validate(session_data.get("common_ground", {}))

        intent_packet = await self.htg_translator.translate(query=query, metadata={"session_id": session_id})
        self_model, cognitive_state = await self._build_initial_cognitive_state(intent_packet, chat_history,
                                                                                common_ground)

        predicted_internal_state = await self.mcl._predict_future_internal_state(cognitive_state)
        # --- FASE 2: LOOP DE DELIBERAÇÃO E EXECUÇÃO (A GRANDE MUDANÇA) ---
        MAX_DELIBERATION_STEPS = 5
        resposta_humana_final = None  # <-- MUDANÇA: inicia como None
        final_response_packet_for_log = None
        last_tool_outputs = []  # <-- NOVO: rastrear outputs de ferramentas

        # Define mcl_params no escopo da função para o caso de loop não rodar (embora vá rodar ao menos 1 vez)
        mcl_params = {}

        for step in range(MAX_DELIBERATION_STEPS):
            logger.info(f"--- [Loop de Deliberação - Passo {step + 1}/{MAX_DELIBERATION_STEPS}] ---")

            # Obter orientação metacognitiva
            guidance, mcl_params = await self.mcl.get_guidance(
                user_model=self.user_model,
                cognitive_state=cognitive_state,
                chat_history=chat_history,
                drives=self.motivational_drives,
                session_data=session_data,
                body_state=self.body_state
            )

            # +++ INÍCIO DA NOVA LÓGICA DE SLEEP MODE +++
            if mcl_params.get("cognitive_state_name") == "CONSOLIDATION_REQUIRED":
                logger.critical(
                    f"CEAFSystem: Sleep Mode ativado para agente {self.agent_id}. Respondendo provisoriamente e iniciando consolidação.")

                # Dispara o ciclo de consolidação em segundo plano.
                from ceaf_core.background_tasks.aura_reflector import main_aura_reflector_cycle
                asyncio.create_task(main_aura_reflector_cycle(self.agent_manager, self.db_repo))

                # --- NOVA LÓGICA DE RESPOSTA PROVISÓRIA ---
                # Em vez de retornar uma mensagem fixa, ele prossegue, mas com um aviso.
                # Ele usará a orientação de "novidade forçada" do MCL para tentar dar uma resposta diferente.
                logger.warning("Sleep Mode: Tentando gerar uma resposta provisória antes de consolidar.")

                # Força os biases para novidade para a resposta provisória
                mcl_params["biases"] = {"coherence_bias": 0.1, "novelty_bias": 0.9}
                mcl_params[
                    "operational_advice_for_ora"] = "ALERTA DE SATURAÇÃO: Tente responder à pergunta do usuário, mas mude a perspectiva ou conecte a um novo tópico. Seja breve."

            turn_prediction = self.lcam.predict_turn_outcome(cognitive_state, mcl_params)
            mcl_params["turn_prediction"] = turn_prediction  # Anexa ao pacote de orientação

            cognitive_state.guidance_packet = guidance

            sim_calibration_config = self.ceaf_dynamic_config.get("SIMULATION_CALIBRATION", {})
            # Deliberar sobre a próxima ação

            winning_strategy = await self.agency_module.decide_next_step(
                cognitive_state,
                mcl_params,
                turn_observer,
                sim_calibration_config,
                chat_history or [],
                known_capabilities=self_model.perceived_capabilities
            )

            mcl_params["winning_strategy"] = winning_strategy

            # MUDANÇA: Validar estratégia antes de executar
            if not winning_strategy or not winning_strategy.decision_type:
                logger.error("Estratégia inválida retornada pelo AgencyModule. Forçando resposta.")
                winning_strategy.decision_type = "response_strategy"
                winning_strategy.strategy_description = "Responder de forma genérica e útil."

            if winning_strategy.decision_type == "response_strategy":
                logger.info("Decisão final: Gerar resposta.")

                # NOVO: Criar um histórico "limpo" apenas com mensagens user/assistant
                clean_history = []
                if chat_history:
                    for msg in chat_history:
                        # Filtrar apenas mensagens user/assistant (ignorar tool/system)
                        if msg.get('role') in ['user', 'assistant']:
                            clean_history.append(msg)

                # Se o histórico limpo não termina com uma mensagem do usuário, adicionar
                if not clean_history or clean_history[-1].get('role') != 'user':
                    clean_history.append({
                        'role': 'user',
                        'content': original_user_query
                    })

                # Buscar memórias de apoio
                supporting_memories = []
                if winning_strategy.key_memory_ids:
                    for mem_id in winning_strategy.key_memory_ids:
                        mem = await self.memory_service.get_memory_by_id(mem_id)
                        if mem:
                            supporting_memories.append(mem)

                # NOVO: Se não há memórias mas há tool outputs recentes, use-os
                if not supporting_memories and last_tool_outputs:
                    logger.warning("Nenhuma memória de apoio. Usando outputs de ferramentas como contexto.")
                    for tool_output in last_tool_outputs:
                        # --- INÍCIO DA CORREÇÃO ---
                        # VERIFICA se o raw_output tem conteúdo antes de criar a memória
                        if tool_output.raw_output and tool_output.raw_output.strip():
                            try:
                                # Criar memória temporária do output da ferramenta
                                temp_mem = ExplicitMemory(
                                    content=ExplicitMemoryContent(text_content=tool_output.raw_output[:500]),
                                    memory_type="explicit",
                                    source_type=MemorySourceType.EXTERNAL_INGESTION,
                                    salience=MemorySalience.MEDIUM,
                                    keywords=["tool_result", tool_output.tool_name]
                                )
                                supporting_memories.append(temp_mem)
                            except ValidationError as e:
                                # Adiciona um log caso a validação falhe por outra razão
                                logger.error(f"Erro de validação ao criar memória temporária da ferramenta: {e}")
                        else:
                            logger.warning(
                                f"Pulando a criação de memória temporária para a ferramenta '{tool_output.tool_name}' pois o raw_output está vazio.")

                # Gerar resposta
                turn_context = {
                    "temperature": mcl_params.get("ora_parameters", {}).get("temperature", 0.7),
                    "max_tokens": mcl_params.get("ora_parameters", {}).get("max_tokens", 2000),
                    "disclosure_level": mcl_params.get("disclosure_level", "moderate"),
                    "coherence_bias": mcl_params.get("biases", {}).get("coherence_bias", 0.5),
                    "novelty_bias": mcl_params.get("biases", {}).get("novelty_bias", 0.5),
                    "operational_advice": mcl_params.get("operational_advice_for_ora")
                }

                final_drives_for_gth = mcl_params.get("enriched_drives", self.motivational_drives)
                final_body_state_for_gth = mcl_params.get("enriched_body_state", self.body_state)

                resposta_humana_final = await self.gth_translator.translate(
                    winning_strategy=winning_strategy,
                    supporting_memories=supporting_memories,
                    user_model=self.user_model,
                    self_model=self_model,
                    agent_name=self.config.get("name", "Aura AI"),
                    chat_history=clean_history,
                    body_state=final_body_state_for_gth,  # <-- Usa a variável nova e enriquecida
                    drives=final_drives_for_gth,  # <-- Usa a variável nova e enriquecida
                    behavioral_rules=[],  # Placeholder, precisa ser implementado
                    turn_context=turn_context,
                    original_user_query=original_user_query,
                    memory_service=self.memory_service,
                    tool_outputs=last_tool_outputs
                )

                # NOVO: Validar resposta antes de sair do loop
                if not resposta_humana_final or resposta_humana_final.strip() == "":
                    logger.error("GTH Translator retornou resposta vazia. Usando fallback.")
                    resposta_humana_final = "Desculpe, tive dificuldade em formular uma resposta adequada. Poderia reformular sua pergunta?"

                final_response_packet_for_log = ResponsePacket(
                    content_summary=resposta_humana_final,
                    confidence_score=winning_strategy.predicted_future_value
                )

                break  # Encerra o loop

            elif winning_strategy.decision_type == "tool_call" and winning_strategy.tool_call_request:
                tool_name = winning_strategy.tool_call_request.get("tool_name")
                tool_args = winning_strategy.tool_call_request.get("arguments", {})

                logger.critical(f"Decisão: Executar ferramenta '{tool_name}'.")

                # Executar ferramenta
                tool_output = await self._execute_tool(tool_name, tool_args, turn_observer)

                # NOVO: Rastrear output
                last_tool_outputs.append(tool_output)

                # Adicionar ao estado cognitivo
                cognitive_state.tool_outputs.append(tool_output)

                # NOVO: Se a ferramenta falhou, considerar responder imediatamente
                if tool_output.status == "error":
                    logger.warning(f"Ferramenta '{tool_name}' falhou. Considerando resposta de fallback.")
                    # Não force resposta ainda, dê uma chance de recuperação

                logger.info(
                    f"Resultado da ferramenta adicionado. Continuando deliberação (passo {step + 2}/{MAX_DELIBERATION_STEPS}).")
                # O loop continua...

            else:
                logger.error(
                    f"Estratégia vencedora com tipo desconhecido: {winning_strategy.decision_type}. Forçando resposta.")
                # NOVO: Forçar resposta em vez de apenas fazer break
                resposta_humana_final = "Tive uma dificuldade em meu processo de pensamento. Poderia reformular sua pergunta de outra forma?"
                break

        else:  # Loop excedeu MAX_DELIBERATION_STEPS
            logger.error("Loop de deliberação excedeu o máximo de passos.")

            # NOVO: Tentar sintetizar algo útil dos tool outputs
            if last_tool_outputs and any(t.status == "success" for t in last_tool_outputs):
                logger.info("Tentando sintetizar resposta dos tool outputs disponíveis.")
                synthesis_prompt = f"""
                    A IA executou algumas ferramentas mas não conseguiu formular uma resposta completa.

                    **Query do Usuário:** "{query}"

                    **Resultados das Ferramentas:**
                    {chr(10).join([f"- {t.tool_name}: {t.raw_output[:300]}" for t in last_tool_outputs if t.status == "success"])}

                    **Sua Tarefa:**
                    Sintetize uma resposta útil e concisa para o usuário com base nos resultados das ferramentas.
                    Se os resultados não forem suficientes, reconheça a limitação e sugira uma reformulação.
                    """

                try:
                    resposta_humana_final = await self.llm_service.ainvoke(
                        self.llm_service.config.smart_model,
                        synthesis_prompt,
                        temperature=0.6,
                        max_tokens=500
                    )
                except Exception as e:
                    logger.error(f"Falha na síntese de emergência: {e}")
                    resposta_humana_final = None

            if not resposta_humana_final:
                resposta_humana_final = (
                    "Parece que entrei em um processo de pensamento muito complexo e não consegui "
                    "chegar a uma conclusão satisfatória. Poderíamos tentar abordar isso de uma forma mais direta? "
                    "O que especificamente você gostaria de saber?"
                )

        # NOVO: Garantia final - nunca retorne None
        if not resposta_humana_final:
            logger.critical("FALLBACK CRÍTICO: Nenhuma resposta foi gerada em todo o processo!")
            resposta_humana_final = (
                "Desculpe, ocorreu um erro no meu sistema de processamento. "
                "Por favor, tente fazer sua pergunta de outra forma."
            )
        # --- FASE 3: PÓS-PROCESSAMENTO E RETORNO ---
        # A lógica de pós-processamento permanece a mesma, mas é movida para fora do loop
        session_data["common_ground"] = cognitive_state.common_ground.model_dump()

        if not final_response_packet_for_log:
            final_response_packet_for_log = ResponsePacket(content_summary=resposta_humana_final, confidence_score=0.3)

        # Avaliação do VRE (permanece aqui)
        final_assessment = await self.vre.evaluate_response_packet(final_response_packet_for_log,
                                                                   observer=turn_observer,
                                                                   cognitive_state=cognitive_state)

        # Criação das métricas FINAIS para o turno
        turn_metrics = {
            "turn_id": turn_id,
            "vre_flags": [rec for rec in final_assessment.textual_recommendations if
                          rec != "Nenhum refinamento necessário."],
            "vre_rejection_count": 1 if final_assessment.adjustment_vectors else 0,
            "agency_score": mcl_params.get("mcl_analysis", {}).get("agency_score", 0.0),
            "relevant_memories_count": len(cognitive_state.relevant_memory_vectors),
            "final_confidence": final_response_packet_for_log.confidence_score,
            "topic_shifted_this_turn": mcl_params.get("topic_shifted_this_turn", False)
        }

        # Geração do relatório de estado interno FINAL (com todas as métricas)
        interoception_module = ComputationalInteroception()
        final_internal_state_report = interoception_module.generate_internal_state_report(turn_metrics)

        # Adiciona as métricas do estado interno ao dicionário para o Embodiment e logging
        turn_metrics["cognitive_strain"] = final_internal_state_report.cognitive_strain
        turn_metrics["cognitive_flow"] = final_internal_state_report.cognitive_flow
        turn_metrics["epistemic_discomfort"] = final_internal_state_report.epistemic_discomfort
        turn_metrics["ethical_tension"] = final_internal_state_report.ethical_tension
        turn_metrics["social_resonance"] = final_internal_state_report.social_resonance

        prediction_error_signal = None
        if predicted_internal_state and final_internal_state_report:
            errors = {
                "strain_error": abs(
                    predicted_internal_state.cognitive_strain - final_internal_state_report.cognitive_strain),
                "flow_error": abs(predicted_internal_state.cognitive_flow - final_internal_state_report.cognitive_flow),
                "discomfort_error": abs(
                    predicted_internal_state.epistemic_discomfort - final_internal_state_report.epistemic_discomfort),
                "tension_error": abs(
                    predicted_internal_state.ethical_tension - final_internal_state_report.ethical_tension),
                "resonance_error": abs(
                    predicted_internal_state.social_resonance - final_internal_state_report.social_resonance)
            }
            total_error = sum(errors.values()) / len(errors)  # Erro médio

            prediction_error_signal = {
                "prediction_error_signal": {
                    "total_error": total_error,
                    "components": errors,
                    "predicted_state": predicted_internal_state.model_dump(),
                    "actual_state": final_internal_state_report.model_dump()
                }
            }
            logger.info(f"PREDICTIVE QUALIA: Erro de predição (surpresa) calculado: {total_error:.4f}")

        logger.info(
            f"Relatório de Interocepção FINAL: Strain={final_internal_state_report.cognitive_strain:.2f}, "
            f"Flow={final_internal_state_report.cognitive_flow:.2f}"
        )

        # Inicia o pós-processamento em segundo plano (agora com as métricas completas)
        asyncio.create_task(self.post_process_turn(
            turn_observer=turn_observer,
            prediction_error_signal=prediction_error_signal,
            turn_id=turn_id, session_id=session_id, query=query,
            final_response=resposta_humana_final, self_model_before=self_model,
            cognitive_state=cognitive_state,
            final_response_packet=final_response_packet_for_log, mcl_guidance=mcl_params,
            vre_assessment=final_assessment,
            winning_strategy=mcl_params.get("winning_strategy"),
            turn_prediction=turn_prediction,
            turn_metrics=turn_metrics
        ))



        logger.info(f"--- FIM DO TURNO CEAF V3.4 (Tool Loop) ---")

        telemetry_components = {
            "turn_metrics": turn_metrics,
            "mcl_params": mcl_params,
            "self_model": self_model,
            "body_state": self.body_state,
            "drives": self.motivational_drives
        }

        return {
            "response": resposta_humana_final,
            "telemetry_components": telemetry_components
        }

# --- Bloco de Demonstração ---
async def main():
    """Função principal para demonstrar o CEAFSystem em ação."""
    print("--- DEMONSTRAÇÃO DO CEAF V3 ---")
    agent_config = {"agent_id": "demo_agent_001", "persistence_path": "./agent_data/demo_agent_001"}

    # Limpa dados antigos da demonstração
    demo_path = Path(agent_config["persistence_path"])
    if demo_path.exists():
        import shutil
        shutil.rmtree(demo_path)

    ceaf = CEAFSystem(config=agent_config, agent_manager=None, db_repo=None)
    session_id = "demo_session_123"

    print("\n[Cenário 1: Consulta Simples - Caminho Direto]")
    response1 = await ceaf.process("Qual é a capital da França?", session_id)
    print(f"\n>> Resposta Final ao Usuário: {response1['response']}\n")

    print("\n[Cenário 2: Consulta Complexa - Caminho da Agência]")
    response2 = await ceaf.process("Por favor, pense profundamente sobre as implicações da agência de IA.", session_id)
    print(f"\n>> Resposta Final ao Usuário: {response2['response']}\n")

    # Aguarda as tarefas de fundo finalizarem para a demonstração
    await asyncio.sleep(5)  # Aumentado para garantir que a atualização de identidade ocorra


if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\nERRO: A variável de ambiente OPENROUTER_API_KEY não está definida.")
        print("Por favor, defina-a em seu ambiente ou em um arquivo .env para rodar a demonstração.")
    else:
        try:
            asyncio.run(main())
        except Exception as e:
            print(f"\nOcorreu um erro durante a execução da demonstração: {e}")
            print(
                "Verifique se as dependências estão instaladas: pip install litellm pydantic numpy sentence-transformers scikit-learn vaderSentiment")