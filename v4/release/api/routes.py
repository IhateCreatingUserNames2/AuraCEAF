# api/routes.py
"""
Rotas da API refatoradas para a Arquitetura de Síntese CEAF V3.

Esta versão unifica a interação com os agentes através do novo `CEAFSystem`.
Toda a lógica complexa de NCF, Agency e pós-processamento foi movida para
dentro do `CEAFSystem`, simplificando drasticamente a camada da API.
"""
import logging
import uuid
import shutil
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import asyncio
import os
from contextlib import asynccontextmanager
from ceaf_core.models import CognitiveProfile

from starlette.responses import StreamingResponse
from billing_logic import MODEL_USER_COSTS_CREDITS, check_and_debit_credits
import litellm

from ceaf_core.modules.memory_blossom import KGEntityRecord
from prebuilt_agents_system import PrebuiltAgentRepository, create_sample_prebuilt_agents
import bcrypt
import jwt
from fastapi import (FastAPI, HTTPException, Depends, status, UploadFile, File, Form)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Header

from whatsapp_bridge.database import WhatsAppUser
from whatsapp_bridge.whatsapp_client import send_whatsapp_message
import json
from pydantic import BaseModel, EmailStr, Field, ValidationError, model_validator
from ceaf_core.modules.memory_blossom.memory_types import ExplicitMemory, ExplicitMemoryContent, MemorySourceType, MemorySalience

CONVERSATION_LOG_PATH = Path(__file__).resolve().parent.parent / "conversation_logs"
CONVERSATION_LOG_PATH.mkdir(exist_ok=True)
CONVERSATION_LOG_FILE = CONVERSATION_LOG_PATH / "chat_logs.jsonl"

SYSTEM_AUTH_TOKEN = "seu-token-secreto-de-sistema"
import io
import zipfile
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
# --- Importações dos Módulos Principais ---
from ceaf_core.background_tasks.aura_reflector import main_aura_reflector_cycle
from agent_manager import AgentManager
from ceaf_core.system import CEAFSystem
from database.models import AgentRepository, User, ChatSession, Message, Agent
from rag_processor import process_and_index_file

# --- Modelos Pydantic (integrados para simplicidade) ---
class UserRegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str

class FileResponseModel(BaseModel):
    filename: str
    content_type: str
    size: int
    url: str

class UserLoginRequest(BaseModel):
    username: str
    password: str

class ProactiveMessageRequest(BaseModel):
    user_id: str
    agent_id: str
    message: str

class CreateAgentRequest(BaseModel):
    name: str
    persona: str
    detailed_persona: str
    model: Optional[str] = "openrouter/openai/gpt-oss-20b:free"
    settings: Optional[Dict[str, Any]] = None

class EntityUpdateRequest(BaseModel):
    label: Optional[str] = None # O novo nome para a pessoa, ex: "Carlos"
    attributes: Optional[Dict[str, Any]] = None # Dados adicionais, ex: {"role": "Gerente de Vendas"}
    description: Optional[str] = None

class TranscriptionEngineType(str, Enum):
    VOSK_OFFLINE = "vosk_offline"
    PICOVOICE_PREMIUM = "picovoice_premium"

class TranscriptionSegment(BaseModel):
    speaker_id: str  # Ex: "voice_print_1", "user", "unknown_1"
    text: str
    timestamp_start: float
    timestamp_end: float
    confidence: float

class TranscriptionIngestRequest(BaseModel):
    segments: List[TranscriptionSegment]
    conversation_timestamp: float
    location_context: Optional[str] = None # Opcional: "Escritório", "Café"
    participants: List[str] # Lista de speaker_ids na conversa
    engine: TranscriptionEngineType

class UpdateAgentRequest(BaseModel):
    name: Optional[str] = None
    persona: Optional[str] = None
    detailed_persona: Optional[str] = None
    avatar_url: Optional[str] = None
    model: Optional[str] = None
    settings: Optional[dict] = None

class AgentBiography(BaseModel):
    class BiographyConfig(BaseModel):
        name: str
        persona: str
        detailed_persona: str
        model: str = "openrouter/openai/gpt-4o-mini"
        settings: Optional[Dict[str, Any]] = None

    class BiographyMemory(BaseModel):
        memory_type: str = "explicit"
        content: Dict[str, str] = Field(..., example={"text_content": "Este é o conteúdo da memória."})
        salience: str = "medium"
        source_type: str = "external_ingestion"
        keywords: List[str] = []
        is_core_value: bool = Field(False, description="Set to true if this memory represents a core value or belief.")

    config: BiographyConfig
    biography: List[BiographyMemory]

class MemoryUploadRequest(BaseModel):
    memories: List[Dict[str, Any]]
    overwrite_existing: bool = False

class BulkMemoryUploadResponse(BaseModel):
    successful_uploads: int
    failed_uploads: int
    errors: List[str]


# Atualize a AgentResponse para incluir mais detalhes
class AgentResponse(BaseModel):
    agent_id: str
    name: str
    persona: str
    detailed_persona: str
    created_at: str # Mantém como string para a resposta da API
    model: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    avatar_url: Optional[str] = None
    is_public_template: bool = False
    clone_count: int = 0
    version: str = "1.0.0"


    @model_validator(mode='before')
    @classmethod
    def convert_datetime_to_iso(cls, data: Any) -> Any:
        if isinstance(data, dict):
            created_at_val = data.get('created_at')
            if isinstance(created_at_val, datetime):
                data['created_at'] = created_at_val.isoformat()
        return data


# Substitua o ClonePrebuiltAgentRequest por este mais genérico
class CloneAgentRequest(BaseModel):
    source_agent_id: str
    custom_name: Optional[str] = None
    clone_memories: bool = True


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    agent_id: str
    timestamp: str
    system_type: str = "ceaf_v3"
    telemetry: Optional[Dict[str, Any]] = None

class MemorySearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10

class PrebuiltAgentResponse(BaseModel):
    id: str
    name: str
    archetype: str
    maturity_level: str
    system_type: str
    short_description: str
    detailed_persona: str
    version: str
    tags: List[str]
    rating: float
    download_count: int


# --- Módulos de Autenticação (integrados para simplicidade) ---
JWT_SECRET = "your-secret-key-change-this-in-production"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
security = HTTPBearer()


def create_access_token(user_id: str, username: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode = {"user_id": user_id, "username": username, "exp": expire}
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {"user_id": payload["user_id"], "username": payload["username"]}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except (jwt.InvalidTokenError, jwt.PyJWTError) as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {str(e)}")


async def run_aura_reflector_periodically(manager: AgentManager,
                                          db_repo: AgentRepository):  # <--- ADICIONE db_repo AQUI
    """
    Executa o ciclo do AuraReflector periodicamente.
    """
    interval_seconds = int(os.getenv("AURA_REFLECTOR_INTERVAL_SECONDS", 60))
    logger.info(f"AuraReflector: Tarefa de fundo iniciada. Ciclo a cada {interval_seconds}s.")

    await asyncio.sleep(10)

    while True:
        try:
            logger.critical(f"AURA_REFLECTOR_HEARTBEAT: Aguardando próximos {interval_seconds}s para o ciclo.")
            await asyncio.sleep(interval_seconds)

            logger.info("--- [CICLO DE REFLEXÃO DE FUNDO] Iniciando AuraReflector ---")
            # PASSE O db_repo AQUI --->
            await main_aura_reflector_cycle(manager, db_repo)
            logger.info("--- [CICLO DE REFLEXÃO DE FUNDO] AuraReflector concluído. ---")
        except asyncio.CancelledError:
            logger.info("AuraReflector: Tarefa cancelada.")
            break
        except Exception as e:
            logger.error(f"AuraReflector: Erro crítico no ciclo de fundo: {e}", exc_info=True)
            await asyncio.sleep(interval_seconds)


async def log_conversation_turn(
    agent_id: str,
    session_id: str,
    user_query: str,
    agent_response: str
):
    """
    Anexa uma consulta de usuário e a resposta do agente a um arquivo JSONL
    (JSON Lines) de forma assíncrona para não bloquear a resposta ao usuário.
    """
    log_entry = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "agent_id": agent_id,
        "session_id": session_id,
        "user_query": user_query,
        "agent_response": agent_response,
    }

    def write_to_file():
        """Função síncrona para escrita em arquivo."""
        try:
            with open(CONVERSATION_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            # Usa o logger principal do arquivo de rotas
            logger.error(f"Falha ao escrever no log de conversas: {e}", exc_info=True)

    # Executa a escrita de arquivo (que é bloqueante) em uma thread separada
    await asyncio.to_thread(write_to_file)


# --- FUNÇÃO AUXILIAR DE CHAT UNIFICADA PARA V3 ---
async def process_chat_message_v3(agent_id: str, message: str, session_id: str, current_user: dict, chat_history: List[Dict[str, str]]) -> dict:
    """Função auxiliar unificada para processar mensagens de chat com o CEAFSystem."""
    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance:
        raise HTTPException(status_code=404, detail="Agente não encontrado.")

    config = agent_manager.agent_configs.get(agent_id)
    if not config or config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Acesso negado a este agente.")

    # --- INÍCIO DA NOVA LÓGICA DE CUSTOS ---
    # 1. Obter o modelo que será usado
    model_to_use = config.model

    # 2. Debitar créditos antes de processar
    with db_repo.SessionLocal() as db_session:
        # Estimativa de tokens (pode ser refinada)
        # Usamos litellm.token_counter para uma contagem precisa
        try:
            input_tokens = litellm.token_counter(model=model_to_use, text=message)
            # Saída é uma estimativa, podemos refinar após a resposta
            estimated_output_tokens = 500
        except Exception:
            # Fallback para contagem simples se o token_counter falhar
            input_tokens = len(message) // 4
            estimated_output_tokens = 500

        # Verifica e debita o custo estimado
        debit_success = await check_and_debit_credits(
            db_session=db_session,
            user_id=current_user["user_id"],
            agent_id=agent_id,
            model_name=model_to_use,
            input_tokens=input_tokens,
            output_tokens=estimated_output_tokens # Usamos a estimativa por enquanto
        )

        if not debit_success:
            raise HTTPException(status_code=402, detail="Créditos insuficientes para esta ação.")
    # --- FIM DA NOVA LÓGICA DE CUSTOS ---

    with db_repo.SessionLocal() as db_session:
        db_session.add(Message(session_id=session_id, role="user", content=message))
        db_session.commit()

    try:
        response_dict = await agent_instance.process(query=message, session_id=session_id, chat_history=chat_history)
        response_text = response_dict.get("response", "Ocorreu um erro ao gerar a resposta.")
    except Exception as e:
        logger.error(f"Erro crítico durante CEAFSystem.process() para o agente {agent_id}: {e}", exc_info=True)
        # ESTORNO: Em caso de falha, podemos estornar os créditos
        # (Lógica de estorno pode ser adicionada aqui se desejado)
        raise HTTPException(status_code=500, detail="Erro interno no processamento do agente.")

    # <<< INÍCIO DA MODIFICAÇÃO >>>
    # Registra a conversa no arquivo JSONL como uma tarefa de fundo
    asyncio.create_task(log_conversation_turn(
        agent_id=agent_id,
        session_id=session_id,
        user_query=message,
        agent_response=response_text
    ))
    # <<< FIM DA MODIFICAÇÃO >>>

    with db_repo.SessionLocal() as db_session:
        db_session.add(Message(session_id=session_id, role="assistant", content=response_text))
        db_session.commit()

    # Opcional: Ajuste de custo pós-resposta para maior precisão
    # (Pode ser uma tarefa em segundo plano para não atrasar a resposta ao usuário)

    return {
        "response": response_text, "session_id": session_id, "agent_id": agent_id,
        "timestamp": datetime.utcnow().isoformat(), "system_type": "ceaf_v3"
    }


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_STORAGE_PATH = PROJECT_ROOT / "agent_data"
DATABASE_FILE_PATH = PROJECT_ROOT / "aura_agents_v3.db"
AGENT_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

# 1. INICIALIZE AS DEPENDÊNCIAS PRIMEIRO
db_repo = AgentRepository(db_url=f"sqlite:///{str(DATABASE_FILE_PATH)}")
agent_manager = AgentManager(base_storage_path=str(AGENT_STORAGE_PATH), db_repo=db_repo)
prebuilt_repo = create_sample_prebuilt_agents()


# 2. AGORA DEFINA A FUNÇÃO LIFESPAN QUE USA AS DEPENDÊNCIAS
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerenciador de ciclo de vida do FastAPI para iniciar tarefas de fundo.
    """
    # Para ter certeza, vamos adicionar um log aqui
    logger.critical(">>> CICLO DE VIDA (LIFESPAN) DA APLICAÇÃO INICIADO <<<")

    reflector_task = asyncio.create_task(run_aura_reflector_periodically(agent_manager, db_repo))

    yield

    logger.info("Encerrando aplicação e tarefas de fundo...")
    reflector_task.cancel()
    try:
        await reflector_task
    except asyncio.CancelledError:
        logger.info("Tarefa do AuraReflector cancelada com sucesso.")


# 3. FINALMENTE, INICIALIZE O APP FASTAPI USANDO A FUNÇÃO LIFESPAN
app = FastAPI(
    title="Aura Multi-Agent API (CEAF V3)",
    description="API para criar e gerenciar agentes sob a Arquitetura de Síntese CEAF V3 unificada.",
    version="3.0.0",
    lifespan=lifespan,
    root_path="/ceaf"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])





FRONTEND_PATH = Path(__file__).resolve().parent.parent / "frontend"

# 1. Monta a pasta 'frontend' para que arquivos CSS, JS, imagens, etc. possam ser acessados
# O caminho "/static" é virtual. Se seu index.html chama <link href="/css/style.css">,
# o FastAPI irá procurar por frontend/css/style.css


@app.get("/prebuilt-agents/list", response_model=List[PrebuiltAgentResponse], tags=["Pre-built Agents Marketplace"])
async def list_prebuilt_agents(current_user: dict = Depends(verify_token)):
    """
    Lista todos os agentes disponíveis no marketplace, incluindo
    agentes pré-construídos ("de fábrica") e templates publicados por usuários.
    """
    try:
        response_data = []
        seen_ids = set()

        # --- PASSO 1: Obter agentes pré-construídos do sistema ---
        system_agents = prebuilt_repo.get_available_agents()
        for agent in system_agents:
            if agent.id not in seen_ids:
                response_data.append(
                    PrebuiltAgentResponse(
                        id=agent.id,
                        name=f"{agent.name} (Sistema)",
                        archetype=agent.archetype.value,
                        maturity_level=agent.maturity_level.value,
                        system_type=agent.system_type,
                        short_description=agent.short_description,
                        detailed_persona=agent.detailed_persona,
                        version=agent.version,
                        tags=agent.tags + ["system"],
                        rating=agent.rating,
                        download_count=agent.download_count
                    )
                )
                seen_ids.add(agent.id)

        # --- PASSO 2: Obter templates publicados por usuários do banco de dados ---
        with db_repo.SessionLocal() as session:
            user_published_agents = session.query(Agent).filter(Agent.is_public_template == 1).all()
            for agent in user_published_agents:
                if agent.id not in seen_ids:
                    response_data.append(
                        PrebuiltAgentResponse(
                            id=agent.id,
                            name=agent.name,
                            archetype=agent.settings.get("archetype", "custom"),
                            maturity_level=agent.settings.get("maturity_level", "newborn"),
                            system_type=agent.settings.get("system_type", "ceaf_v3"),
                            short_description=agent.persona,
                            detailed_persona=agent.detailed_persona or agent.persona,
                            version=agent.version,
                            tags=agent.settings.get("tags", []) + ["user-published"],
                            rating=0.0,
                            download_count=agent.clone_count or 0
                        )
                    )
                    seen_ids.add(agent.id)

        response_data.sort(key=lambda x: x.name)
        return response_data

    except Exception as e:
        logger.error(f"Erro ao listar agentes do marketplace unificado: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Não foi possível buscar os agentes do marketplace.")


@app.post("/agents/dispatch-proactive", tags=["System & Proactivity"])
async def dispatch_proactive_message(
        request: ProactiveMessageRequest,
        authorization: str = Header(None)
):
    """
    Endpoint interno para despachar uma mensagem proativa gerada pelo AuraReflector.
    Este endpoint lida com a lógica de encontrar o canal de comunicação do usuário (ex: WhatsApp).
    """
    # 1. Autenticação interna para garantir que apenas o sistema possa chamar este endpoint
    if not authorization or authorization.split(" ")[1] != SYSTEM_AUTH_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Acesso não autorizado")

    logger.info(
        f"API Dispatcher: Recebida mensagem proativa para user_id {request.user_id} do agente {request.agent_id}")

    # 2. Lógica de Despacho (aqui você decide para onde enviar)
    # Por enquanto, vamos implementar apenas para o WhatsApp

    delivery_channel = "whatsapp"  # Pode ser expandido no futuro

    if delivery_channel == "whatsapp":
        try:
            with db_repo.SessionLocal() as session:
                # Encontra o usuário do WhatsApp com base no user_id do Aura
                whatsapp_user = session.query(WhatsAppUser).filter(WhatsAppUser.aura_user_id == request.user_id).first()

                if not whatsapp_user or not whatsapp_user.phone_number:
                    logger.error(
                        f"API Dispatcher: Não foi possível encontrar um número de WhatsApp associado ao user_id {request.user_id}.")
                    raise HTTPException(status_code=404, detail="Usuário do WhatsApp não encontrado para este user_id.")

                # Envia a mensagem
                await send_whatsapp_message(whatsapp_user.phone_number, request.message)

                logger.info(f"API Dispatcher: Mensagem proativa enviada com sucesso para {whatsapp_user.phone_number}.")

                # Adicionar a mensagem ao histórico de chat (opcional, mas recomendado)
                # Encontrar a sessão mais recente entre este usuário e agente
                latest_session = session.query(ChatSession).filter(
                    ChatSession.user_id == request.user_id,
                    ChatSession.agent_id == request.agent_id
                ).order_by(ChatSession.last_active.desc()).first()

                if latest_session:
                    db_repo.save_message(session_id=latest_session.id, role="assistant",
                                         content=f"[PROACTIVE] {request.message}")

                return {"status": "success", "channel": "whatsapp", "to": whatsapp_user.phone_number}

        except Exception as e:
            logger.error(f"API Dispatcher: Falha ao enviar mensagem via WhatsApp: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Falha ao despachar mensagem.")

    else:
        # Futuramente, você pode adicionar outras lógicas aqui (Telegram, Email, etc.)
        raise HTTPException(status_code=501, detail=f"Canal de entrega '{delivery_channel}' não implementado.")


# --- ROTAS DE AUTENTICAÇÃO E USUÁRIO (Sem alterações) ---
@app.post("/auth/register", status_code=status.HTTP_201_CREATED, tags=["Authentication"])
async def register_user(request: UserRegisterRequest):
    with db_repo.SessionLocal() as session:
        if session.query(User).filter((User.email == request.email) | (User.username == request.username)).first():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email or username already exists")
        hashed_pw = bcrypt.hashpw(request.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        new_user = User(email=request.email, username=request.username, password_hash=hashed_pw)
        session.add(new_user)
        session.commit()
        session.refresh(new_user)
        token = create_access_token(new_user.id, new_user.username)
        return {"user_id": new_user.id, "username": new_user.username, "access_token": token, "token_type": "bearer"}


@app.post("/auth/login", tags=["Authentication"])
async def login_user(request: UserLoginRequest):
    with db_repo.SessionLocal() as session:
        user = session.query(User).filter(User.username == request.username).first()
        if not user or not bcrypt.checkpw(request.password.encode('utf-8'), user.password_hash.encode('utf-8')):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        token = create_access_token(user.id, user.username)
        return {"user_id": user.id, "username": user.username, "access_token": token, "token_type": "bearer"}


@app.get("/auth/me", tags=["Authentication"])
async def get_current_user(current_user: dict = Depends(verify_token)):
    return current_user


# --- NOVAS ROTAS DE CONFIGURAÇÃO COGNITIVA (TWERKING) ---

@app.get("/agents/{agent_id}/config", response_model=CognitiveProfile, tags=["Agent Configuration"])
async def get_agent_config(
        agent_id: str,
        current_user: dict = Depends(verify_token)
):
    """
    Retorna o Perfil Cognitivo completo do agente (Prompts, Pesos, Configs de LLM).
    """
    # Verifica permissão
    config = agent_manager.agent_configs.get(agent_id)
    if not config or config.user_id != current_user['user_id']:
        raise HTTPException(status_code=404, detail="Agente não encontrado ou acesso negado.")

    # Busca o perfil do disco
    profile = agent_manager.get_agent_profile(agent_id)
    if not profile:
        raise HTTPException(status_code=500, detail="Perfil cognitivo não encontrado ou corrompido.")

    return profile


@app.patch("/agents/{agent_id}/config", response_model=CognitiveProfile, tags=["Agent Configuration"])
async def update_agent_config(
        agent_id: str,
        config_update: Dict[str, Any],  # Aceita JSON parcial
        current_user: dict = Depends(verify_token)
):
    """
    Atualiza o Perfil Cognitivo do agente em tempo real (Hot Reload).
    Aceita um JSON parcial ou completo.
    Exemplo: {"llm_config": {"fast_model": "openrouter/google/gemini-flash"}}
    """
    # Verifica permissão
    config = agent_manager.agent_configs.get(agent_id)
    if not config or config.user_id != current_user['user_id']:
        raise HTTPException(status_code=404, detail="Agente não encontrado ou acesso negado.")

    # Tenta atualizar via AgentManager (que cuida do Hot Reload)
    success = agent_manager.update_agent_profile(agent_id, config_update)

    if not success:
        raise HTTPException(status_code=400, detail="Falha ao atualizar a configuração. Verifique o formato do JSON.")

    # Retorna o perfil atualizado para confirmação
    updated_profile = agent_manager.get_agent_profile(agent_id)
    return updated_profile


@app.get("/agents/{agent_id}/files", response_model=List[FileResponseModel], tags=["RAG & Files"])
async def list_agent_files(
    agent_id: str,
    current_user: dict = Depends(verify_token)
):
    """Lista todos os arquivos disponíveis no armazenamento de um agente."""
    config = agent_manager.agent_configs.get(agent_id)
    if not config or config.user_id != current_user['user_id']:
        raise HTTPException(status_code=404, detail="Agente não encontrado ou acesso negado.")

    files_dir = Path(config.persistence_path) / "files"
    if not files_dir.exists():
        return []

    file_list = []
    for f in files_dir.iterdir():
        if f.is_file():  # Ignora subdiretórios como 'vector_store'
            import mimetypes
            content_type, _ = mimetypes.guess_type(f.name)

            file_list.append(FileResponseModel(
                filename=f.name,
                content_type=content_type or "application/octet-stream",
                size=f.stat().st_size,
                url=f"/agent_files/{config.user_id}/{agent_id}/files/{f.name}"
            ))

# --- ROTAS DE GERENCIAMENTO DE AGENTES (Simplificadas para V3) ---
@app.post("/agents", response_model=dict, tags=["Agent Management"])
async def create_agent(request: CreateAgentRequest, current_user: dict = Depends(verify_token)):
    """Cria um novo agente CEAF V3 a partir de parâmetros básicos."""
    try:
        # A lógica do agent_manager já cuida do DB e dos arquivos
        agent_id = agent_manager.create_agent(
            user_id=current_user["user_id"],
            name=request.name,
            persona=request.persona,
            detailed_persona=request.detailed_persona,
            model=request.model,
            settings=request.settings
        )
        db_repo.create_agent(
            agent_id=agent_id, user_id=current_user["user_id"], name=request.name,
            persona=request.persona, detailed_persona=request.detailed_persona,
            model=request.model, settings=request.settings
        )
        return {"agent_id": agent_id, "message": f"Agente '{request.name}' (CEAF V3) criado com sucesso."}
    except Exception as e:
        logger.error(f"Erro ao criar agente: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/from-biography", response_model=dict, tags=["Agent Management"])
async def create_agent_from_biography(
        file: UploadFile = File(...),
        current_user: dict = Depends(verify_token)
):
    """Cria um novo agente com uma biografia rica a partir de um arquivo JSON (formato V2)."""
    if file.content_type != "application/json":
        raise HTTPException(status_code=400, detail="Por favor, envie um arquivo JSON.")

    try:
        content_bytes = await file.read()
        data = json.loads(content_bytes)

        # A validação agora funciona com o novo formato de JSON
        validated_data = AgentBiography(**data)

        config_data = validated_data.config
        # Extrai a lista de memórias como dicionários, que já estão no formato correto
        biography_memories = [mem.model_dump() for mem in validated_data.biography]

    except (ValidationError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=400,
                            detail=f"Estrutura de arquivo de biografia inválida ou não corresponde ao formato esperado: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro inesperado ao processar o arquivo: {e}")

    try:
        # A lógica aqui permanece a mesma: passar os dicionários de memória diretamente
        agent_id = agent_manager.create_agent(
            user_id=current_user["user_id"],
            name=config_data.name,
            persona=config_data.persona,
            detailed_persona=config_data.detailed_persona,
            model=config_data.model,
            settings=config_data.settings,
            initial_memories=biography_memories  # <-- O fio continua conectado corretamente
        )

        db_repo.create_agent(
            agent_id=agent_id, user_id=current_user["user_id"], name=config_data.name,
            persona=config_data.persona, detailed_persona=config_data.detailed_persona,
            model=config_data.model, settings=config_data.settings
        )

        return {
            "agent_id": agent_id,
            "message": "Agente criado com sucesso a partir da biografia.",
            "memories_injected": len(biography_memories)
        }
    except Exception as e:
        logger.error(f"Erro ao criar agente da biografia: {e}", exc_info=True)
        if 'agent_id' in locals():
            agent_manager.delete_agent(agent_id, current_user["user_id"])
        raise HTTPException(status_code=500, detail=f"Falha ao criar agente: {str(e)}")





@app.get("/agents", response_model=List[AgentResponse], tags=["Agent Management"])
async def list_agents(current_user: dict = Depends(verify_token)):
    """Lista todos os agentes do usuário com detalhes completos do DB e config."""
    configs = agent_manager.list_user_agents(current_user["user_id"])

    with db_repo.SessionLocal() as session:
        db_agents_map = {
            agent.id: agent for agent in session.query(Agent).filter(Agent.user_id == current_user["user_id"]).all()
        }

    response_list = []
    for config in configs:
        db_agent = db_agents_map.get(config.agent_id)

        # Garante que created_at seja uma string no formato correto
        created_at_str = config.created_at
        if isinstance(config.created_at, datetime):
            created_at_str = config.created_at.isoformat()

        response_list.append(AgentResponse(
            agent_id=config.agent_id,
            name=config.name,
            persona=config.persona,
            detailed_persona=config.detailed_persona,
            created_at=created_at_str,  # <--- Usa a string segura
            model=config.model,
            settings=config.settings,
            avatar_url=db_agent.avatar_url if db_agent else None,
            is_public_template=bool(db_agent.is_public_template) if db_agent else False,
            clone_count=db_agent.clone_count if db_agent else 0,
            version=db_agent.version if db_agent else "1.0.0"
        ))

    return sorted(response_list, key=lambda x: x.created_at, reverse=True)


@app.get("/agents/{agent_id}", response_model=AgentResponse, tags=["Agent Management"])
async def get_agent(agent_id: str, current_user: dict = Depends(verify_token)):
    config = agent_manager.agent_configs.get(agent_id)
    if not config or config.user_id != current_user['user_id']:
        raise HTTPException(status_code=404, detail="Agente não encontrado ou acesso negado.")

    # Com o validador no modelo Pydantic, você pode apenas passar o dicionário
    return AgentResponse(**config.__dict__)


@app.delete("/agents/{agent_id}", tags=["Agent Management"])
async def delete_agent(agent_id: str, current_user: dict = Depends(verify_token)):
    if not agent_manager.delete_agent(agent_id, current_user["user_id"]):
        raise HTTPException(status_code=404, detail="Agente não encontrado ou acesso negado.")
    return {"message": "Agente e todos os seus dados foram deletados com sucesso."}

# AUREOLA ENDPOINTS

@app.get("/agents/{agent_id}/aureola/unnamed-persons", tags=["Aureola Functions"], response_model=List[KGEntityRecord])
async def get_unnamed_persons(agent_id: str, current_user: dict = Depends(verify_token)):
    """
    Lista todas as entidades de 'pessoa' que ainda não foram nomeadas pelo usuário.
    (label começa com 'Pessoa Desconhecida' ou 'Unknown Person').
    """
    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance or agent_instance.config.get('user_id') != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Acesso negado.")

    unnamed_persons = await agent_instance.memory_service.get_unnamed_persons()
    return unnamed_persons


@app.put("/agents/{agent_id}/aureola/persons/{entity_id_str}", tags=["Aureola Functions"],
         response_model=KGEntityRecord)
async def update_person_entity(
        agent_id: str,
        entity_id_str: str,
        request: EntityUpdateRequest,
        current_user: dict = Depends(verify_token)
):
    """
    Atualiza uma entidade de pessoa, permitindo ao usuário dar um nome
    e adicionar atributos (ex: cargo, relação).
    """
    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance or agent_instance.config.get('user_id') != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Acesso negado.")

    updated_entity = await agent_instance.memory_service.update_kg_entity(
        entity_id_str=entity_id_str,
        update_data=request.model_dump(exclude_unset=True)
    )

    if not updated_entity:
        raise HTTPException(status_code=404, detail="Entidade de pessoa não encontrada.")

    return updated_entity


@app.get("/agents/{agent_id}/aureola/persons/{entity_id_str}/details", tags=["Aureola Functions"])
async def get_person_details(agent_id: str, entity_id_str: str, current_user: dict = Depends(verify_token)):
    """
    Busca os detalhes de uma entidade de pessoa e todas as suas memórias e relações associadas (o "Graph Tree").
    """
    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance or agent_instance.config.get('user_id') != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Acesso negado.")

    details = await agent_instance.memory_service.get_entity_details(entity_id_str)

    if not details:
        raise HTTPException(status_code=404, detail="Entidade de pessoa não encontrada.")

    return details


@app.post("/agents/{agent_id}/ingest/transcription", tags=["Aureola Ingestion"])
async def ingest_transcription(
        agent_id: str,
        request: TranscriptionIngestRequest,
        current_user: dict = Depends(verify_token)
):
    """
    Endpoint para a Aureola fazer a ingestão de uma conversa transcrita.
    Agora inclui o motor de transcrição usado.
    """
    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance or agent_instance.config.get('user_id') != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Acesso negado.")

    full_text_content = "\n".join(
        [f"[{seg.speaker_id}]: {seg.text}" for seg in request.segments]
    )

    try:
        new_memory = ExplicitMemory(
            content=ExplicitMemoryContent(
                text_content=full_text_content,
                structured_data=request.model_dump()
            ),
            memory_type="explicit",
            source_type=MemorySourceType.EXTERNAL_INGESTION,
            salience=MemorySalience.MEDIUM,
            keywords=["conversa", "transcrição", "aureola"] + request.participants,
            metadata={
                "ingestion_source": "aureola_app",
                "conversation_timestamp": request.conversation_timestamp,
                "transcription_engine": request.engine.value  # <--- SALVANDO O MOTOR USADO
            }
        )

        await agent_instance.memory_service.add_specific_memory(new_memory)

        logger.info(f"Ingestão ({request.engine.value}) para o agente {agent_id} bem-sucedida.")

        return {
            "message": "Transcrição ingerida com sucesso.",
            "memory_id": new_memory.memory_id
        }
    except Exception as e:
        logger.error(f"Falha na ingestão da transcrição para o agente {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro ao processar e salvar a transcrição.")


# END OF AUREOLA ENDPOINTS

# --- ROTAS DE CHAT E MEMÓRIA (Unificadas para V3) ---
@app.post("/agents/{agent_id}/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_with_agent(agent_id: str, request: ChatRequest, current_user: dict = Depends(verify_token)):
    """
    Processa uma mensagem de chat com um agente.
    V3.3 Refatorado: A lógica de estado da sessão agora é gerenciada internamente pelo CEAFSystem.
    """
    session_id = request.session_id
    chat_history = []

    # --- Lógica de persistência de longo prazo (permanece na API) ---
    with db_repo.SessionLocal() as db_session:
        # Garante que um session_id exista para o log
        if session_id:
            # Se uma sessão existe, atualize seu tempo de 'last_active'
            session_obj = db_session.query(ChatSession).filter(ChatSession.id == session_id).first()
            if session_obj:
                session_obj.last_active = datetime.utcnow()
            else:
                # Se o ID foi passado mas não existe no DB, cria um novo
                session_id = str(uuid.uuid4())  # Gera um novo ID para evitar conflito
                new_session = ChatSession(id=session_id, user_id=current_user["user_id"], agent_id=agent_id)
                db_session.add(new_session)
        else:
            # Se nenhuma sessão foi iniciada, cria uma nova
            session_id = str(uuid.uuid4())
            new_session = ChatSession(id=session_id, user_id=current_user["user_id"], agent_id=agent_id)
            db_session.add(new_session)

        db_session.commit()

        # Carrega o histórico de mensagens do DB para dar contexto ao agente
        recent_messages = db_session.query(Message).filter(Message.session_id == session_id).order_by(
            Message.created_at.desc()).limit(10).all()
        recent_messages.reverse()
        chat_history = [{"role": msg.role, "content": msg.content} for msg in recent_messages]

        # Salva a mensagem do usuário no DB
        db_session.add(Message(session_id=session_id, role="user", content=request.message))
        db_session.commit()

    # --- Lógica de Processamento Cognitivo (agora simplificada) ---
    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance:
        raise HTTPException(status_code=404, detail="Agente não encontrado.")

    config = agent_manager.agent_configs.get(agent_id)
    if not config or config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Acesso negado a este agente.")

    try:
        # A chamada ao core agora retorna mais dados
        result_dict = await agent_instance.process(
            query=request.message,
            session_id=session_id,
            chat_history=chat_history
        )
        response_text = result_dict.get("response", "Ocorreu um erro ao gerar a resposta.")

        # <<< INÍCIO DA MODIFICAÇÃO >>>
        # Extrai os componentes de telemetria brutos
        telemetry_components = result_dict.get("telemetry_components", {})

    except Exception as e:
        logger.error(f"Erro crítico durante CEAFSystem.process(): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro interno no processamento do agente.")

        # --- Persistência Final e Log (código existente) ---
    with db_repo.SessionLocal() as db_session:
        db_session.add(Message(session_id=session_id, role="assistant", content=response_text))
        db_session.commit()

    asyncio.create_task(log_conversation_turn(agent_id=agent_id, session_id=session_id, user_query=request.message,
                                              agent_response=response_text))

    # --- Construção do Pacote de Telemetria para o Frontend ---
    telemetry_data = {}
    if telemetry_components:
        # Extrai os sub-dicionários para facilitar o acesso
        turn_metrics = telemetry_components.get("turn_metrics", {})
        mcl_params = telemetry_components.get("mcl_params", {})
        self_model = telemetry_components.get("self_model")
        body_state = telemetry_components.get("body_state")
        drives = telemetry_components.get("drives")

        telemetry_data = {
            "body_cognitive_fatigue": body_state.cognitive_fatigue if body_state else 0,
            "body_info_saturation": body_state.information_saturation if body_state else 0,
            "cognitive_flow": turn_metrics.get("cognitive_flow", 0.0),
            "cognitive_strain": turn_metrics.get("cognitive_strain", 0.0),
            "drive_curiosity": drives.curiosity if drives else 0.5,
            "drive_connection": drives.connection if drives else 0.5,
            "drive_mastery": drives.mastery if drives else 0.5,
            "drive_consistency": drives.consistency if drives else 0.5,
            "mcl_coherence_bias": mcl_params.get("biases", {}).get("coherence_bias", 0.5),
            "mcl_novelty_bias": mcl_params.get("biases", {}).get("novelty_bias", 0.5),
            "mcl_cognitive_state": mcl_params.get("cognitive_state_name", "STABLE_OPERATION"),
            "identity_version": self_model.version if self_model else 0,
            "turn_final_confidence": turn_metrics.get("final_confidence", 0.7),
        }

    # Constrói a resposta final da API, agora com a telemetria
    return ChatResponse(
        response=response_text,
        session_id=session_id,
        agent_id=agent_id,
        timestamp=datetime.utcnow().isoformat(),
        telemetry=telemetry_data
    )


@app.post("/agents/{agent_id}/memories/search", tags=["Memory Management"])
async def search_agent_memories(agent_id: str, request: MemorySearchRequest,
                                current_user: dict = Depends(verify_token)):
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent or agent.config.get('user_id') != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Agente não encontrado ou acesso negado.")
    results = await agent.memory_service.search_raw_memories(query=request.query, top_k=request.limit)
    return {"agent_id": agent_id, "query": request.query, "results": results}

@app.get("/agents/{agent_id}/memories/{memory_id}", tags=["Memory Management"])
async def get_specific_memory(
    agent_id: str,
    memory_id: str,
    current_user: dict = Depends(verify_token)
):
    """Busca uma memória específica pelo seu ID."""
    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance or agent_instance.config.get('user_id') != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Agente não encontrado ou acesso negado.")

    memory = await agent_instance.memory_service.get_memory_by_id(memory_id)

    if not memory:
        raise HTTPException(status_code=404, detail="Memória não encontrada.")

    # Extrai o texto pesquisável para uma resposta simples e limpa
    searchable_text, _ = await agent_instance.memory_service._get_searchable_text_and_keywords(memory)

    return {
        "memory_id": memory_id,
        "content": searchable_text or "Conteúdo textual não disponível.",
        "memory_type": getattr(memory, 'memory_type', 'unknown')
    }

@app.put("/agents/{agent_id}/profile", response_model=dict, tags=["Agent Management"])
async def update_agent_profile(
        agent_id: str,
        request: UpdateAgentRequest,
        current_user: dict = Depends(verify_token)
):
    """Atualiza o perfil de um agente existente (nome, persona, modelo, etc.)."""
    config = agent_manager.agent_configs.get(agent_id)
    if not config or config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Agente não encontrado ou acesso negado.")

    update_data = request.model_dump(exclude_unset=True)
    updated_fields = list(update_data.keys())

    # Atualiza a configuração em memória e no arquivo
    for key, value in update_data.items():
        if hasattr(config, key):
            setattr(config, key, value)
    agent_manager._save_agent_config(config)

    # Atualiza o banco de dados
    with db_repo.SessionLocal() as session:
        db_agent = session.query(Agent).filter(Agent.id == agent_id).first()
        if db_agent:
            for key, value in update_data.items():
                if hasattr(db_agent, key):
                    setattr(db_agent, key, value)
            session.commit()

    return {"message": "Perfil do agente atualizado com sucesso.", "updated_fields": updated_fields}


@app.post("/agents/{agent_id}/memories/upload", response_model=BulkMemoryUploadResponse, tags=["Memory Management"])
async def upload_agent_memories(
        agent_id: str,
        request: MemoryUploadRequest,
        current_user: dict = Depends(verify_token)
):
    """Faz upload de memórias em massa para um agente."""
    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance or agent_instance.config.get('user_id') != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Agente não encontrado ou acesso negado.")

    successful = 0
    failed = 0
    errors = []
    for i, mem_data in enumerate(request.memories):
        try:
            # Converte o dicionário genérico em um objeto de memória Pydantic
            content = ExplicitMemoryContent(text_content=mem_data.get("content"))
            new_mem = ExplicitMemory(
                content=content,
                memory_type="explicit",
                source_type=MemorySourceType.EXTERNAL_INGESTION,
                salience=MemorySalience.MEDIUM,
                keywords=mem_data.get("keywords", [])
            )
            await agent_instance.memory_service.add_specific_memory(new_mem)
            successful += 1
        except Exception as e:
            failed += 1
            errors.append(f"Memória #{i + 1}: {str(e)}")

    return BulkMemoryUploadResponse(successful_uploads=successful, failed_uploads=failed, errors=errors)


@app.get("/agents/{agent_id}/memories/export", tags=["Memory Management"])
async def export_agent_memories(
        agent_id: str,
        format: str = "json",  # json, csv
        current_user: dict = Depends(verify_token)
):
    """Exporta todas as memórias de um agente."""
    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance or agent_instance.config.get('user_id') != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Agente não encontrado ou acesso negado.")

    # Assumindo que MBS tem um método para obter todas as memórias.
    # Se não tiver, você precisará adicionar um.
    # Exemplo de como poderia ser:
    # all_memories = await agent_instance.memory_service.get_all_raw_memories()
    all_memories_raw = await agent_instance.memory_service.search_raw_memories(query="*", top_k=9999)
    all_memories = [mem.model_dump(mode='json') for mem, score in all_memories_raw]

    if format == "json":
        return JSONResponse(content={"agent_id": agent_id, "memories": all_memories})

    elif format == "csv":
        if not PANDAS_AVAILABLE:
            raise HTTPException(status_code=501, detail="A biblioteca 'pandas' é necessária para exportar como CSV.")

        # Achatando os dados para CSV
        flat_memories = []
        for mem in all_memories:
            flat_mem = {"memory_id": mem.get("memory_id"), "timestamp": mem.get("timestamp"),
                        "memory_type": mem.get("memory_type")}
            if mem.get("memory_type") == "explicit" and mem.get("content"):
                flat_mem["content"] = mem["content"].get("text_content")
            # Adicionar mais lógicas para outros tipos de memória
            flat_memories.append(flat_mem)

        df = pd.DataFrame(flat_memories)
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        return StreamingResponse(iter([stream.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": f"attachment; filename={agent_id}_memories.csv"})

    else:
        raise HTTPException(status_code=400, detail="Formato não suportado. Use 'json' ou 'csv'.")


@app.get("/models/openrouter", tags=["Models"])
async def get_openrouter_models(current_user: dict = Depends(verify_token)):
    """Fornece uma lista curada de modelos de IA com seus custos para o usuário."""
    # Defina aqui os modelos que você quer oferecer no frontend
    recommended_models = {
        "Featured (Best Value)": [
            "openrouter/openai/gpt-4o-mini",
            "openrouter/google/gemini-2.5-flash",
        ],
        "Advanced (Maximum Power)": [
            "openrouter/openai/gpt-oss-20b:free",
            "openrouter/anthropic/claude-3.5-sonnet",
        ],
        "Free & Beta": [
            "openrouter/deepseek/deepseek-r1-0528:free",
            "openrouter/horizon-beta",
        ],
    }

    models_with_costs = {}
    for category, model_list in recommended_models.items():
        category_list = []
        for model_name in model_list:
            # Busca o custo da nova tabela centralizada
            cost = MODEL_USER_COSTS_CREDITS.get(model_name, MODEL_USER_COSTS_CREDITS.get("default", 0))
            # Formatando para ser mais legível para o usuário (ex: "1.5k" em vez de 1500)
            formatted_cost = f"{cost / 1000:.1f}k" if cost >= 1000 else str(cost)

            category_list.append({"name": model_name, "cost_display": f"~{formatted_cost} / 1M tokens"})
        models_with_costs[category] = category_list

    return models_with_costs


@app.post("/agents/{agent_id}/avatar", tags=["Agent Management"])
async def upload_agent_avatar(
        agent_id: str,
        file: UploadFile = File(...),
        current_user: dict = Depends(verify_token)
):
    """Faz upload de uma imagem de avatar para um agente."""
    config = agent_manager.agent_configs.get(agent_id)
    if not config or config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Agente não encontrado ou acesso negado.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Por favor, envie um arquivo de imagem.")

    agent_path = Path(config.persistence_path)
    avatar_dir = agent_path / "avatar"
    avatar_dir.mkdir(exist_ok=True)

    file_extension = Path(file.filename).suffix or ".png"
    avatar_path = avatar_dir / f"profile{file_extension}"

    with open(avatar_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # URL relativa para o frontend
    avatar_url = f"/agent_files/{config.user_id}/{agent_id}/avatar/profile{file_extension}"

    # Atualiza o DB
    with db_repo.SessionLocal() as session:
        db_agent = session.query(Agent).filter(Agent.id == agent_id).first()
        if db_agent:
            db_agent.avatar_url = avatar_url
            session.commit()

    return {"message": "Avatar atualizado com sucesso.", "avatar_url": avatar_url}


@app.post("/agents/{agent_id}/publish", tags=["Agent Marketplace"], status_code=status.HTTP_201_CREATED)
async def publish_agent_to_marketplace(
        agent_id: str,
        changelog: str = Form("Primeira versão pública."),
        current_user: dict = Depends(verify_token)
):
    """Publica um agente privado, criando um template público como um 'snapshot' inteligente."""
    source_config = agent_manager.agent_configs.get(agent_id)
    if not source_config or source_config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Agente não encontrado ou não autorizado.")

    public_agent_id = str(uuid.uuid4())
    source_path = Path(source_config.persistence_path)
    public_path = source_path.parent / public_agent_id

    try:
        shutil.copytree(source_path, public_path)

        # Atualiza o arquivo de configuração dentro do novo diretório
        public_config_path = public_path / "agent_config.json"
        with open(public_config_path, 'r+') as f:
            config_dict = json.load(f)
            config_dict['agent_id'] = public_agent_id
            config_dict['persistence_path'] = str(public_path.resolve())
            f.seek(0)
            json.dump(config_dict, f, indent=2)
            f.truncate()

        # Cria a entrada no banco de dados para o template público
        with db_repo.SessionLocal() as session:
            public_template = Agent(
                id=public_agent_id, user_id=current_user["user_id"], name=source_config.name,
                persona=source_config.persona, detailed_persona=source_config.detailed_persona,
                model=source_config.model, settings=source_config.settings,
                is_public_template=1, parent_agent_id=agent_id,
                version="1.0.0", changelog=changelog
            )
            session.add(public_template)
            session.commit()

        # Recarrega as configurações no AgentManager para incluir o novo template
        agent_manager._load_agent_configs()

        return {
            "message": f"Agente '{source_config.name}' publicado com sucesso!",
            "public_template_id": public_agent_id,
        }
    except Exception as e:
        if public_path.exists():
            shutil.rmtree(public_path)
        logger.error(f"Erro ao publicar agente {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Falha ao publicar agente: {str(e)}")


@app.post("/agents/clone", response_model=dict, tags=["Agent Marketplace"])
async def clone_agent(request: CloneAgentRequest, current_user: dict = Depends(verify_token)):
    """Clona um agente (pré-construído ou de template público) para a conta do usuário."""
    source_agent_id = request.source_agent_id
    agent_config_data = {}
    initial_memories_data = []

    # --- Tenta clonar de um agente pré-construído primeiro ---
    try:
        clone_data_prebuilt = prebuilt_repo.clone_agent_for_user(
            agent_id=source_agent_id,
            user_id=current_user["user_id"],
            custom_name=request.custom_name
        )
        agent_config_data = clone_data_prebuilt["agent_config"]
        initial_memories_data = clone_data_prebuilt.get("initial_memories", [])

    except ValueError:
        # --- Se não for pré-construído, tenta encontrar um template público de usuário ---
        source_config = agent_manager.agent_configs.get(source_agent_id)

        # Verifica se o agente existe no AgentManager e se ele está marcado como público no DB
        if source_config:
            source_db_agent = db_repo.get_agent(source_agent_id)
            if source_db_agent and source_db_agent.is_public_template:

                agent_config_data = {
                    "name": request.custom_name or source_config.name,
                    "persona": source_config.persona,
                    "detailed_persona": source_config.detailed_persona,
                    "system_type": source_config.settings.get("system_type", "ceaf_v3"),
                    "model": source_config.model,
                    "settings": source_config.settings
                }

                # <<< INÍCIO DA CORREÇÃO CRÍTICA >>>
                # Carrega as memórias do arquivo .jsonl do agente de origem
                # em vez de passar uma string.
                if request.clone_memories:
                    source_memories_path = Path(source_config.persistence_path) / "all_explicit_memories.jsonl"
                    if source_memories_path.exists():
                        with open(source_memories_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    initial_memories_data.append(json.loads(line))
                                except json.JSONDecodeError:
                                    logger.warning(
                                        f"Skipping malformed memory line during clone from {source_agent_id}")
                # <<< FIM DA CORREÇÃO CRÍTICA >>>

            else:
                raise HTTPException(status_code=404, detail="Agente de origem não é um template público válido.")
        else:
            raise HTTPException(status_code=404, detail="Agente de origem não encontrado.")

    if not agent_config_data:
        raise HTTPException(status_code=500, detail="Não foi possível preparar os dados de configuração para clonagem.")

    # --- A lógica de criação do novo agente agora é unificada ---
    try:
        # Cria o novo agente para o usuário com a configuração e memórias corretas
        new_agent_id = agent_manager.create_agent(
            user_id=current_user["user_id"],
            name=agent_config_data["name"],
            persona=agent_config_data["persona"],
            detailed_persona=agent_config_data["detailed_persona"],
            model=agent_config_data["model"],
            settings=agent_config_data.get("settings", {}),
            initial_memories=initial_memories_data if request.clone_memories else []
        )

        db_repo.create_agent(
            agent_id=new_agent_id, user_id=current_user["user_id"], name=agent_config_data["name"],
            persona=agent_config_data["persona"], detailed_persona=agent_config_data["detailed_persona"],
            model=agent_config_data["model"],
            settings=agent_config_data.get("settings", {}),
            is_public=False  # O clone é sempre privado para o novo usuário
        )

        # Incrementa o contador de clones do original
        with db_repo.SessionLocal() as session:
            source_db_agent = session.query(Agent).filter(Agent.id == source_agent_id).first()
            if source_db_agent:
                source_db_agent.clone_count = (source_db_agent.clone_count or 0) + 1
                session.commit()

        return {"agent_id": new_agent_id, "name": agent_config_data["name"], "message": "Agente clonado com sucesso!"}

    except Exception as e:
        logger.error(f"Erro ao criar o agente clonado: {e}", exc_info=True)
        # Limpeza em caso de falha na criação
        if 'new_agent_id' in locals():
            agent_manager.delete_agent(new_agent_id, current_user["user_id"])
        raise HTTPException(status_code=500, detail="Ocorreu um erro inesperado durante a criação do clone.")


# --- NOVA ROTA DE UPLOAD DE ARQUIVOS (RAG) ---
@app.post("/agents/{agent_id}/files/upload", tags=["RAG & Files"])
async def upload_file_for_rag(agent_id: str, file: UploadFile = File(...), current_user: dict = Depends(verify_token)):
    """Faz upload de um arquivo para ser indexado na memória de arquivos do agente (RAG)."""
    config = agent_manager.agent_configs.get(agent_id)
    if not config or config.user_id != current_user['user_id']:
        raise HTTPException(status_code=404, detail="Agente não encontrado ou acesso negado.")

    temp_dir = Path("./temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_file_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Arquivo '{file.filename}' salvo temporariamente em '{temp_file_path}'. Iniciando indexação.")

        agent_storage_path = Path(config.persistence_path)
        await process_and_index_file(agent_storage_path, temp_file_path)

        return {"message": f"Arquivo '{file.filename}' processado e indexado para o agente '{config.name}'."}
    except Exception as e:
        logger.error(f"Falha no upload e indexação do arquivo para o agente {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Falha ao processar o arquivo.")
    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()


# --- ROTAS DE SERVIÇO ---
@app.get("/health", tags=["Service"])
async def health_check():
    return {"status": "healthy", "architecture_version": "CEAF V3"}


app.mount("/agent_files", StaticFiles(directory=AGENT_STORAGE_PATH), name="agent_files")

# 2. Rota específica para servir o index.html na raiz
@app.get("/", include_in_schema=False)
async def serve_frontend():
    index_path = FRONTEND_PATH / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Aura Multi-Agent API (CEAF V3) está em execução.", "docs_url": "/docs"}


# 3. Rota catch-all para servir arquivos estáticos (CSS, JS, imagens)
# IMPORTANTE: Isso precisa ser a ÚLTIMA rota definida
@app.get("/{full_path:path}", include_in_schema=False)
async def serve_static_or_404(full_path: str):
    # Ignora rotas de API
    if full_path.startswith(
            ("api/", "auth/", "agents/", "prebuilt-agents/", "models/", "health", "docs", "openapi.json")):
        raise HTTPException(status_code=404, detail="Not found")

    # Tenta servir o arquivo estático
    file_path = FRONTEND_PATH / full_path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)

    # Se não encontrou, retorna 404
    raise HTTPException(status_code=404, detail="File not found")