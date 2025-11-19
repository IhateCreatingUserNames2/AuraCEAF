# whatsapp_bridge/bridge_main.py
import os
import logging
from fastapi import FastAPI, Request, HTTPException, Response, BackgroundTasks
from dotenv import load_dotenv

from db_setup import SessionLocal, WhatsAppUser
import aura_client
from whatsapp_client import send_whatsapp_message

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = FastAPI()

VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")

user_agent_cache = {}
user_marketplace_cache = {}


# (As funções de webhook e extract_message_info permanecem as mesmas)
@app.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        logging.info("✅ Webhook verificado com sucesso!")
        return Response(content=challenge, status_code=200)
    else:
        logging.error("❌ Falha na verificação do Webhook.")
        raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
        logging.info(f"📦 Webhook recebido: {data}")
        message_info = extract_message_info(data)
        if message_info:
            phone_number, message_body = message_info
            background_tasks.add_task(process_incoming_message, phone_number, message_body)
        else:
            logging.info("ℹ️ Webhook recebido não é uma mensagem de texto de usuário. Ignorando.")
    except Exception as e:
        logging.error(f"❌ Erro crítico ao processar o corpo do webhook: {e}", exc_info=True)
    return Response(status_code=200)


def extract_message_info(data: dict):
    try:
        if data.get("object") != "whatsapp_business_account": return None
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                if change.get("field") == "messages":
                    value = change.get("value", {})
                    for message_data in value.get("messages", []):
                        if message_data.get("type") == "text":
                            return message_data.get("from"), message_data.get("text", {}).get("body")
    except Exception as e:
        logging.error(f"🚨 Erro ao extrair info da mensagem: {e}", exc_info=True)
    return None


async def process_incoming_message(phone_number: str, message: str):
    logging.info(f"Processando mensagem de {phone_number}: {message}")
    db = SessionLocal()
    try:
        user = db.query(WhatsAppUser).filter(WhatsAppUser.phone_number == phone_number).first()

        # 1. Non-authenticated commands
        if message.lower().startswith("!register"):
            await handle_register_command(phone_number, message, db)
            return
        if message.lower().startswith("!login"):
            await handle_login_command(phone_number, message, user, db)
            return

        # 2. Check if user is logged in
        if not user:
            await send_whatsapp_message(phone_number,
                                        "🌟 Bem-vindo(a) ao AURA!\n\n"
                                        "Parece que este número não está vinculado a uma conta.\n\n"
                                        "🔹 Se você é *novo*, crie uma conta com:\n`!register <email> <usuario> <senha>`\n\n"
                                        "🔹 Se você *já tem uma conta*, vincule este número com:\n`!login <usuario> <senha>`")
            return

        # Altere a lógica do !logout para ser menos destrutiva
        if message.lower() == "!logout":
            if user:
                # Em vez de deletar o usuário, apenas invalidamos o token e o agente selecionado.
                # Isso mantém o vínculo entre o número de telefone e o user_id.
                user.selected_agent_id = None
                user.aura_auth_token = "logged_out"
                db.commit()
            await send_whatsapp_message(phone_number, "✅ Sessão encerrada. Para se conectar novamente, use `!login`.")
            return

        if message.lower() == "!marketplace":
            await handle_marketplace_command(phone_number, user)
            return

        if message.lower().startswith("!clone"):
            await handle_clone_command(phone_number, message, user)
            return

        if message.lower() == "!agents":
            await handle_agents_command(phone_number, user)
            return

        if message.lower().startswith("!select"):
            await handle_select_command(phone_number, message, user, db)
            return

        # ==================== NEW COMMANDS LOGIC START ====================
        if message.lower() == "!modelos":
            await handle_models_command(phone_number, user)
            return

        if message.lower().startswith("!modelo "):
            await handle_set_model_command(phone_number, message, user)
            return
        # ===================== NEW COMMANDS LOGIC END =====================

        if message.lower() in ["!exit", "!menu"]:
            if user.selected_agent_id:
                user.selected_agent_id = None
                db.commit()
                await send_whatsapp_message(phone_number, "🤖 Você voltou ao menu principal.")
            else:
                await send_whatsapp_message(phone_number, "ℹ️ Você já está no menu principal.")
            return

        if message.lower() == "!help":
            status_text = "Nenhum agente selecionado."
            if user.selected_agent_id:
                try:
                    agents = await aura_client.get_my_agents(user.aura_auth_token)
                    current_agent = next((a for a in agents if a['agent_id'] == user.selected_agent_id), None)
                    if current_agent: status_text = f"Agente atual: *{current_agent['name']}*"
                except:
                    status_text = f"Agente atual: `{user.selected_agent_id}`"

            help_text = f"""🤖 *Comandos AURA no WhatsApp*

*Estado Atual:*
{status_text}

*Marketplace de Agentes:*
`!marketplace` - Vê agentes públicos para clonar.
`!clone <número>` - Clona um agente do marketplace.

*Seus Agentes:*
`!agents` - Lista seus agentes.
`!select <número>` - Conversa com um de seus agentes.
`!exit` ou `!menu` - Volta para a seleção de agentes.

*Configuração do Agente (selecionado):*
`!modelos` - Lista os modelos de IA disponíveis.
`!modelo <nome_do_modelo>` - Altera o modelo do agente atual.

*Gerenciamento de Conta:*
`!register <email> <user> <pass>` - Cria uma conta.
`!login <user> <pass>` - Conecta uma conta.
`!logout` - Desconecta sua conta.
`!help` - Mostra esta mensagem."""
            await send_whatsapp_message(phone_number, help_text)
            return

        # 4. Chat message
        if not user.selected_agent_id:
            await send_whatsapp_message(phone_number, "❌ Nenhum agente selecionado. Use `!agents` ou `!marketplace`.")
            return

        await handle_chat_message(phone_number, message, user)

    except Exception as e:
        logging.error(f"Erro ao processar mensagem de {phone_number}: {e}", exc_info=True)
        await send_whatsapp_message(phone_number, "❌ Ocorreu um erro interno. A equipe foi notificada.")
    finally:
        db.close()


# (handle_register_command e handle_login_command permanecem os mesmos)
async def handle_register_command(phone_number: str, message: str, db):
    """Handles the !register command."""
    parts = message.split()
    if len(parts) != 4:
        await send_whatsapp_message(phone_number, "❌ Comando inválido. Use: `!register <email> <usuario> <senha>`")
        return

    _, email, username, password = parts
    try:
        await send_whatsapp_message(phone_number, "⏳ Criando sua conta, um momento...")
        reg_data = await aura_client.register_user(email, username, password)

        if reg_data and "access_token" in reg_data:
            user = db.query(WhatsAppUser).filter(WhatsAppUser.phone_number == phone_number).first()
            if user:
                user.aura_user_id = reg_data["user_id"]
                user.aura_auth_token = reg_data["access_token"]
                user.selected_agent_id = None
            else:
                user = WhatsAppUser(phone_number=phone_number, aura_user_id=reg_data["user_id"],
                                    aura_auth_token=reg_data["access_token"])
                db.add(user)
            db.commit()
            await send_whatsapp_message(phone_number,
                                        "✅ Registro realizado com sucesso! Você já está logado. Use `!marketplace` para encontrar seu primeiro agente.")
        else:
            error_msg = reg_data.get("error", "Não foi possível completar o registro.")
            await send_whatsapp_message(phone_number, f"❌ Falha no registro: {error_msg}")

    except Exception as e:
        logging.error(f"Erro no registro para {phone_number}: {e}", exc_info=True)
        await send_whatsapp_message(phone_number, "❌ Erro durante o registro. Tente novamente.")


async def handle_login_command(phone_number: str, message: str, user, db):
    parts = message.split()
    if len(parts) != 3:
        await send_whatsapp_message(phone_number, "❌ Comando inválido. Use: !login <usuario> <senha>")
        return
    _, username, password = parts
    try:
        auth_data = await aura_client.login(username, password)
        if auth_data:
            if user:
                user.aura_user_id = auth_data["user_id"]
                user.aura_auth_token = auth_data["access_token"]
                user.selected_agent_id = None
            else:
                user = WhatsAppUser(phone_number=phone_number, aura_user_id=auth_data["user_id"],
                                    aura_auth_token=auth_data["access_token"])
                db.add(user)
            db.commit()
            await send_whatsapp_message(phone_number,
                                        "✅ Login realizado com sucesso! Use `!agents` para ver seus agentes.")
        else:
            await send_whatsapp_message(phone_number, "❌ Login falhou. Verifique seu usuário e senha.")
    except Exception as e:
        logging.error(f"Erro no login para {phone_number}: {e}")
        await send_whatsapp_message(phone_number, "❌ Erro no login. Tente novamente.")


# (handle_agents_command, handle_select_command, etc. permanecem os mesmos)
async def handle_agents_command(phone_number: str, user):
    """Lida com o comando !agents, agora com atalhos numéricos."""
    try:
        agents = await aura_client.get_my_agents(user.aura_auth_token)
        if agents:
            unique_agents = {agent['agent_id']: agent for agent in agents}.values()
            sorted_agents = sorted(list(unique_agents), key=lambda x: x['name'])
            user_agent_cache[phone_number] = sorted_agents
            response_text = "🤖 *Seus Agentes:*\n\n"
            for i, agent in enumerate(sorted_agents, 1):
                response_text += f"*{i}.* {agent['name']}\n"
            response_text += "\n💬 Use `!select <número>` para conversar com um agente."
            await send_whatsapp_message(phone_number, response_text)
        else:
            await send_whatsapp_message(phone_number,
                                        "❌ Você ainda não tem agentes. Use `!marketplace` para clonar seu primeiro agente!")
    except Exception as e:
        logging.error(f"Erro ao buscar agentes para {phone_number}: {e}", exc_info=True)
        await send_whatsapp_message(phone_number, "❌ Erro ao buscar agentes. Tente novamente.")


async def handle_select_command(phone_number: str, message: str, user, db):
    """Lida com o comando !select, agora aceitando números ou IDs."""
    parts = message.split()
    if len(parts) != 2:
        await send_whatsapp_message(phone_number, "❌ Comando inválido. Use: `!select <número_da_lista>`")
        return
    selection = parts[1]
    try:
        agent_id_to_select, agent_name = (None, None)
        if selection.isdigit():
            agent_index = int(selection) - 1
            if phone_number in user_agent_cache and 0 <= agent_index < len(user_agent_cache[phone_number]):
                agent_to_select = user_agent_cache[phone_number][agent_index]
                agent_id_to_select, agent_name = agent_to_select['agent_id'], agent_to_select['name']
            else:
                await send_whatsapp_message(phone_number,
                                            f"❌ Número `{selection}` inválido. Use `!agents` primeiro para ver a lista.")
                return
        else:
            agents = await aura_client.get_my_agents(user.aura_auth_token)
            selected_agent = next((a for a in agents if a['agent_id'] == selection), None)
            if not selected_agent:
                await send_whatsapp_message(phone_number, f"❌ Agente com ID `{selection}` não encontrado.")
                return
            agent_id_to_select, agent_name = selected_agent['agent_id'], selected_agent['name']

        if agent_id_to_select:
            user.selected_agent_id = agent_id_to_select
            db.commit()
            await send_whatsapp_message(phone_number, f"✅ Agora você está conversando com *{agent_name}*.")
            if phone_number in user_agent_cache: del user_agent_cache[phone_number]
    except Exception as e:
        logging.error(f"Erro ao selecionar agente para {phone_number}: {e}", exc_info=True)
        await send_whatsapp_message(phone_number, "❌ Erro ao selecionar agente.")


async def handle_marketplace_command(phone_number: str, user):
    """Handles the !marketplace command."""
    try:
        await send_whatsapp_message(phone_number, "🔍 Buscando agentes no marketplace... um momento.")
        public_agents = await aura_client.get_public_agents(user.aura_auth_token)
        if not public_agents:
            await send_whatsapp_message(phone_number, "😕 Nenhum agente público encontrado no marketplace no momento.")
            return

        unique_agents = {agent['id']: agent for agent in public_agents}.values()
        sorted_agents = sorted(list(unique_agents), key=lambda x: x['name'])
        user_marketplace_cache[phone_number] = sorted_agents

        response_text = "✨ *Marketplace de Agentes Públicos:*\n\n"
        for i, agent in enumerate(sorted_agents, 1):
            persona = agent.get('short_description', 'Sem descrição.')
            response_text += f"*{i}. {agent['name']}*\n_{persona}_\n\n"
        response_text += " cloning... Para clonar um agente para sua conta, use `!clone <número>`."
        await send_whatsapp_message(phone_number, response_text)
    except Exception as e:
        logging.error(f"Erro ao buscar marketplace para {phone_number}: {e}", exc_info=True)

        # LINHA DE DEBUG ADICIONADA:
        # Converte a exceção em texto e envia para o usuário
        error_details = f"DEBUG: Falha ao contatar a API. Erro: {str(e)}"

        await send_whatsapp_message(phone_number, error_details)


async def handle_clone_command(phone_number: str, message: str, user):
    """Handles the !clone command."""
    parts = message.split()
    if len(parts) != 2 or not parts[1].isdigit():
        await send_whatsapp_message(phone_number, "❌ Comando inválido. Use o *número* do agente: `!clone <número>`")
        return

    agent_index = int(parts[1]) - 1
    if phone_number not in user_marketplace_cache or not (0 <= agent_index < len(user_marketplace_cache[phone_number])):
        await send_whatsapp_message(phone_number, f"❌ Número inválido. Use `!marketplace` para ver a lista.")
        return

    try:
        agent_to_clone = user_marketplace_cache[phone_number][agent_index]
        source_agent_id = agent_to_clone['id']
        custom_name = agent_to_clone['name']  # Clone with the original name

        await send_whatsapp_message(phone_number,
                                    f" cloning... Clonando *{agent_to_clone['name']}* para sua conta... ⏳")
        clone_result = await aura_client.clone_agent(user.aura_auth_token, source_agent_id, custom_name)

        if clone_result and "agent_id" in clone_result:
            await send_whatsapp_message(phone_number,
                                        f"✅ Sucesso! O agente '{clone_result.get('name', custom_name)}' foi adicionado à sua conta. Use `!agents` para vê-lo.")
            if phone_number in user_marketplace_cache: del user_marketplace_cache[phone_number]
            if phone_number in user_agent_cache: del user_agent_cache[phone_number]
        else:
            error_message = clone_result.get("error", "Ocorreu um erro durante a clonagem.")
            await send_whatsapp_message(phone_number, f"❌ Falha ao clonar: {error_message}")
    except Exception as e:
        logging.error(f"Erro ao clonar agente para {phone_number}: {e}", exc_info=True)
        await send_whatsapp_message(phone_number, "❌ Ocorreu um erro interno durante a clonagem.")


async def handle_models_command(phone_number: str, user):
    """Handles the !modelos command."""
    try:
        await send_whatsapp_message(phone_number, "🔍 Buscando modelos de IA disponíveis...")
        models_data = await aura_client.get_available_models(user.aura_auth_token)
        if not models_data:
            await send_whatsapp_message(phone_number, "😕 Não foi possível buscar a lista de modelos.")
            return

        response_text = "🤖 *Modelos de IA Disponíveis (Custo em Créditos)*\n\n"
        for category, models in models_data.items():
            response_text += f"*{category}*\n"
            for model_info in models:
                # Format for better readability in WhatsApp
                response_text += f"∙ `{model_info['name']}` ({model_info['cost']})\n"
            response_text += "\n"

        response_text += "Para alterar o modelo do seu agente *atualmente selecionado*, use:\n`!modelo <nome_completo_do_modelo>`"
        await send_whatsapp_message(phone_number, response_text)

    except Exception as e:
        logging.error(f"Erro ao buscar modelos para {phone_number}: {e}", exc_info=True)
        await send_whatsapp_message(phone_number, "❌ Erro ao buscar a lista de modelos.")


async def handle_set_model_command(phone_number: str, message: str, user):
    """Handles the !modelo <model_name> command."""
    if not user.selected_agent_id:
        await send_whatsapp_message(phone_number, "❌ Nenhum agente selecionado. Use `!select <número>` primeiro.")
        return

    parts = message.split(maxsplit=1)
    if len(parts) < 2 or not parts[1]:
        await send_whatsapp_message(phone_number,
                                    "❌ Comando inválido. Especifique o nome do modelo: `!modelo openrouter/openai/gpt-4o-mini`")
        return

    new_model_name = parts[1].strip()
    try:
        await send_whatsapp_message(phone_number, f"⏳ Alterando o modelo do agente para `{new_model_name}`...")
        result = await aura_client.update_agent_model(user.aura_auth_token, user.selected_agent_id, new_model_name)

        if result and "error" not in result:
            await send_whatsapp_message(phone_number,
                                        f"✅ Modelo do agente atualizado com sucesso para `{new_model_name}`!")
        else:
            error_msg = result.get("error", "Não foi possível atualizar o modelo.")
            await send_whatsapp_message(phone_number, f"❌ Falha ao atualizar: {error_msg}")
    except Exception as e:
        logging.error(f"Erro ao definir modelo para {phone_number}: {e}", exc_info=True)
        await send_whatsapp_message(phone_number, "❌ Erro interno ao tentar atualizar o modelo do agente.")


async def handle_chat_message(phone_number: str, message: str, user):
    aura_response = await aura_client.chat_with_agent(user.aura_auth_token, user.selected_agent_id, message)
    if aura_response and "response" in aura_response:
        await send_whatsapp_message(phone_number, aura_response["response"])
    else:
        await send_whatsapp_message(phone_number, "❌ Desculpe, houve um erro ao se comunicar com o agente.")


if __name__ == "__main__":
    import uvicorn

    required_env = ["WHATSAPP_PERMANENT_TOKEN", "WHATSAPP_PHONE_NUMBER_ID", "WHATSAPP_VERIFY_TOKEN"]
    if any(not os.getenv(env) for env in required_env):
        logging.error(f"❌ Variáveis de ambiente faltando: {[env for env in required_env if not os.getenv(env)]}")
        exit(1)
    logging.info("🚀 Iniciando WhatsApp Bridge...")
    uvicorn.run(app, host="0.0.0.0", port=8001)