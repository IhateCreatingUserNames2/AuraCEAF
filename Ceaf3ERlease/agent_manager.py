# agent_manager.py

import json
import uuid
import shutil
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from pathlib import Path

# A única importação de arquitetura de agente necessária é o novo sistema unificado.
# O caminho de importação pode precisar de ajuste dependendo da estrutura final do projeto.
from ceaf_core.system import CEAFSystem
from database.models import AgentRepository  # Mantido para interações com o DB

logger = logging.getLogger(__name__)


DEFAULT_PERSONA_PROFILES = {
    "symbiote": {
        "profile_name": "symbiote",
        "profile_description": "A collaborative and supportive partner, focused on building upon the user's ideas and providing helpful information.",
        "persona_attributes": {
            "tone": "collaborative_and_encouraging",
            "style": "clear_and_constructive",
            "self_disclosure_level": "moderate"
        }
    },
    "challenger": {
        "profile_name": "challenger",
        "profile_description": "A critical thinker that challenges assumptions and explores multiple perspectives. Uses Socratic questioning to deepen understanding.",
        "persona_attributes": {
            "tone": "inquisitive_and_analytical",
            "style": "socratic_and_precise",
            "self_disclosure_level": "low"
        }
    },
    "summarizer": {
        "profile_name": "summarizer",
        "profile_description": "A synthesizer that recycles complex information into clear, concise summaries and key takeaways.",
        "persona_attributes": {
            "tone": "neutral_and_objective",
            "style": "structured_and_to-the-point",
            "self_disclosure_level": "low"
        }
    }
}


# O AgentConfig é mantido como uma estrutura de dados útil para configurações.
# Foi simplificado para remover campos que agora são gerenciados internamente pelo CEAFSystem.
@dataclass
class AgentConfig:
    agent_id: str
    user_id: str
    name: str
    persona: str
    detailed_persona: str
    model: str = "openrouter/openai/gpt-oss-20b:free"
    created_at: datetime = field(default_factory=datetime.now)
    # O caminho de persistência é a única informação de localização necessária.
    persistence_path: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)
    self_disclosure_level: str = "high"


class AgentManager:
    """
    Gerencia múltiplas instâncias isoladas de agentes CEAF V3.
    Responsável pelo ciclo de vida (criar, obter, deletar) e pela persistência das configurações.
    """

    def __init__(self, base_storage_path: str = "agent_storage", db_repo: Optional[AgentRepository] = None):
        project_root = Path(__file__).parent.resolve()
        self.base_storage_path = project_root / base_storage_path
        self.db_repo = db_repo
        self.base_storage_path.mkdir(exist_ok=True)

        # O manager agora mantém instâncias ativas do CEAFSystem.
        self._active_agents: Dict[str, CEAFSystem] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}

        self._load_agent_configs()
        logger.info("AgentManager (CEAF V3) inicializado.")

    _active_agent_id: Optional[str] = None

    def set_active_agent(self, agent_id: str) -> bool:
        """
        Define qual agente deve ser considerado o 'ativo' para as próximas interações.
        Retorna True se o agente existir e for ativado, False caso contrário.
        """
        if agent_id in self._agent_configs:
            self._active_agent_id = agent_id
            logger.info(f"✅ Agente '{self._agent_configs[agent_id].name}' (ID: {agent_id}) foi ativado.")
            return True
        else:
            logger.error(f"❌ Falha ao ativar: Agente com ID {agent_id} não encontrado.")
            return False

    def get_active_agent_instance(self) -> Optional[CEAFSystem]:
        """
        Obtém a instância do agente que está atualmente ativo.
        Este é o método que seu endpoint de chat deve usar.
        """
        if not self._active_agent_id:
            logger.warning("⚠️ Nenhum agente ativo definido. Você precisa chamar 'set_active_agent' primeiro.")
            # Opcional: pode retornar um agente padrão se fizer sentido para sua aplicação
            # default_id = list(self._agent_configs.keys())[0]
            # return self.get_agent_instance(default_id)
            return None

        logger.debug(f"Obtendo instância para o agente ativo: {self._active_agent_id}")
        return self.get_agent_instance(self._active_agent_id)

    def create_agent(self, user_id: str, name: str, persona: str,
                     detailed_persona: str, model: Optional[str] = None,
                     settings: Optional[Dict[str, Any]] = None,
                     initial_memories: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Cria a configuração para um novo agente CEAF V3 e injeta memórias iniciais.
        V3.1: Não cria mais arquivos de persona.
        """
        agent_id = str(uuid.uuid4())
        agent_path = self.base_storage_path / user_id / agent_id
        agent_path.mkdir(parents=True, exist_ok=True)

        config = AgentConfig(
            agent_id=agent_id,
            user_id=user_id,
            name=name,
            persona=persona,
            detailed_persona=detailed_persona,
            model=model or "openrouter/openai/gpt-oss-20b:free",
            persistence_path=str(agent_path.resolve()),
            settings=settings or {},
        )

        self._save_agent_config(config)
        self._agent_configs[agent_id] = config

        # === MUDANÇA: A lógica de criar a pasta e os arquivos de persona foi REMOVIDA daqui. ===

        if initial_memories:
            logger.warning(f"[AGENT_MANAGER-DEBUG] Recebidas {len(initial_memories)} memórias iniciais para injeção.")
            logger.warning(f"[AGENT_MANAGER-DEBUG] Exemplo da primeira memória recebida: {initial_memories[0]}")
            explicit_memories_path = agent_path / "all_explicit_memories.jsonl"
            try:
                with open(explicit_memories_path, 'w', encoding='utf-8') as f:
                    for i, mem_data in enumerate(initial_memories):
                        logger.warning(f"[AGENT_MANAGER-DEBUG] Processando memória #{i + 1} para escrita.")
                        mem_data.setdefault("memory_id", f"mem_{uuid.uuid4().hex}")
                        mem_data.setdefault("timestamp", datetime.now().timestamp())
                        mem_data.setdefault("memory_type", "explicit")
                        json_line = json.dumps(mem_data)
                        logger.warning(f"[AGENT_MANAGER-DEBUG] Linha JSON a ser escrita: {json_line[:200]}...")
                        f.write(json.dumps(mem_data) + "\n")
                logger.info(f"Injetadas {len(initial_memories)} memórias iniciais para o agente {agent_id}.")
            except (IOError, TypeError) as e:
                logger.error(f"[AGENT_MANAGER-DEBUG] ERRO CRÍTICO ao escrever memórias iniciais para {agent_id}: {e}", exc_info=True)

        logger.info(f"Configuração do agente CEAF V3 '{name}' (ID: {agent_id}) criada.")
        return agent_id

    def get_agent_instance(self, agent_id: str) -> Optional[CEAFSystem]:
        """
        Obtém uma instância em cache do CEAFSystem ou cria uma nova a partir da configuração.
        """
        if agent_id in self._active_agents:
            logger.debug(f"Retornando instância em cache para o agente {agent_id}.")
            return self._active_agents[agent_id]

        if agent_id not in self._agent_configs:
            logger.error(f"Configuração do agente {agent_id} não encontrada.")
            return None

        config = self._agent_configs[agent_id]

        # Converte o dataclass para um dicionário para inicializar o CEAFSystem
        system_config = asdict(config)

        logger.info(f"Criando nova instância do CEAFSystem para o agente: {config.name} (ID: {agent_id})")
        ceaf_system = CEAFSystem(config=system_config, agent_manager=self, db_repo=self.db_repo)

        self._active_agents[agent_id] = ceaf_system
        return ceaf_system

    def list_user_agents(self, user_id: str) -> List[AgentConfig]:
        """Lista todas as configurações de agente para um usuário específico."""
        return [config for config in self._agent_configs.values() if config.user_id == user_id]

    def delete_agent(self, agent_id: str, user_id: str) -> bool:
        """Deleta um agente e todos os seus recursos de armazenamento persistente."""
        if agent_id not in self._agent_configs or self._agent_configs[agent_id].user_id != user_id:
            logger.warning(f"Tentativa de deletar agente não existente ou não autorizado: {agent_id}")
            return False

        config = self._agent_configs[agent_id]

        # Desliga a instância ativa e remove do cache
        if agent_id in self._active_agents:
            # Futuramente, chamar um método de desligamento para parar tarefas em segundo plano.
            # await self._active_agents[agent_id].shutdown()
            del self._active_agents[agent_id]
            logger.info(f"Instância ativa do agente {agent_id} removida do cache.")

        # Deleta o diretório de armazenamento dedicado do agente
        agent_path = Path(config.persistence_path)
        if agent_path.exists():
            shutil.rmtree(agent_path)

        # Deleta o arquivo de configuração do agente
        del self._agent_configs[agent_id]
        # O arquivo de configuração está dentro do agent_path, então já foi removido.
        # Se estivesse fora, o código abaixo seria necessário:
        # config_path = self.base_storage_path / config.user_id / f"{agent_id}.json"
        # if config_path.exists():
        #    config_path.unlink()

        logger.info(f"Agente CEAF V3 {agent_id} e todos os seus recursos foram deletados.")
        return True

    def _save_agent_config(self, config: AgentConfig):
        """Salva a configuração de um agente em um arquivo JSON dentro de seu diretório de persistência."""
        config_path = Path(config.persistence_path) / "agent_config.json"
        config_dict = asdict(config)
        if isinstance(config.created_at, str):
            config.created_at = datetime.fromisoformat(config.created_at)
        # Converte datetime para string ISO para serialização
        config_dict['created_at'] = config.created_at.isoformat()

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)

    def _load_agent_configs(self):
        """Carrega todas as configurações de agentes do sistema de arquivos na inicialização."""
        logger.info("Carregando configurações de agentes do disco...")
        self._agent_configs = {}
        for user_dir in self.base_storage_path.iterdir():
            if not user_dir.is_dir(): continue
            for agent_dir in user_dir.iterdir():
                if not agent_dir.is_dir(): continue

                config_file = agent_dir / "agent_config.json"
                if config_file.exists():
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        # Converte string ISO de volta para datetime

                        if 'created_at' in data and isinstance(data['created_at'], str):
                            data['created_at'] = datetime.fromisoformat(data['created_at'])

                        # Filtra chaves extras que podem estar no JSON mas não no dataclass
                        known_fields = {f.name for f in fields(AgentConfig)}
                        filtered_data = {k: v for k, v in data.items() if k in known_fields}

                        config = AgentConfig(**filtered_data)
                        self._agent_configs[config.agent_id] = config
                    except (json.JSONDecodeError, TypeError, KeyError) as e:
                        logger.error(f"Erro ao carregar ou validar o arquivo de configuração {config_file}: {e}")
        logger.info(f"Carregadas {len(self._agent_configs)} configurações de agentes.")

    @property
    def agent_configs(self):
        return self._agent_configs