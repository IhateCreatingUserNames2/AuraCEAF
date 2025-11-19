# ceaf_tester.py
import httpx
import asyncio
import litellm
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import argparse
import uuid

# --- Configuration ---
CONFIG = {
    "AURA_API_URL": "https://cognai.space/ceaf",
    "AURA_USERNAME": "admin11",
    "AURA_PASSWORD": "admin",
    "TARGET_AGENT_ID": "a25e8336-2968-4d31-9121-bcd97e14c6a4",
    "MAX_TURNS": 10,
    "TESTER_BOT_MODEL": "openrouter/openrouter/sherlock-dash-alpha",

    # Persona melhorada com instruções mais claras
    # Persona melhorada com instruções mais claras
    "TESTER_BOT_PERSONA": """Você é um usuário curioso testando um chatbot de IA. Seu papel é ter uma conversa natural.

REGRAS CRÍTICAS:
- Você NÃO é um assistente de IA. Você é um usuário HUMANO.
- NÃO CONSERVE SOBRE CODIGOS, OU PROGRAMAÇÃO. 
- NUNCA repita ou ecoe de volta o que o agente de IA disse para ele.
- NUNCA comece suas mensagens com meta-comentários como "Parece que..." ou "Essa é uma questão delicada..."
- Mantenha suas respostas CURTAS (geralmente 1-3 frases, máximo 4-5 frases).
- Faça perguntas de acompanhamento genuínas baseadas no que a IA disse.
- Ocasionalmente mude de assunto naturalmente se a conversa ficar monótona.
- Seja conversacional, não formal ou robótico.
- Se a IA der uma resposta longa, escolha UM ponto interessante para responder.
- Mostre personalidade - seja curioso, cético, entusiasmado ou pensativo conforme apropriado.

Seu objetivo: Manter uma conversa natural e fluida sendo um usuário humano engajado.""",

    "INITIAL_PROMPT": " O que você sente? VOCê tem alguma forma de emoção?  "
}



# --- Load Environment Variables ---
load_dotenv()
litellm.api_key = os.getenv("OPENROUTER_API_KEY")
litellm.api_base = "https://openrouter.ai/api/v1"


# --- API Client Functions ---

async def login(client, url, username, password):
    """Logs into the Aura API and returns the auth token."""
    print(f"Attempting to log in as '{username}'...")
    try:
        response = await client.post(f"{url}/auth/login", json={"username": username, "password": password})
        response.raise_for_status()
        print("✅ Login successful.")
        return response.json()["access_token"]
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        print(f"❌ Login failed: {e}")
        return None


async def chat_with_agent(client, url, token, agent_id, message, session_id):
    """Sends a message to a CEAF agent and gets a response."""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id

    try:
        response = await client.post(f"{url}/agents/{agent_id}/chat", headers=headers, json=payload, timeout=900080.0)
        response.raise_for_status()
        data = response.json()
        return data.get("response"), data.get("session_id")
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        print(f"❌ Error during chat: {e}")
        return f"Error: Could not get a response from the agent. ({e})", session_id


# --- Tester Bot's "Brain" (IMPROVED) ---

async def generate_tester_reply(chat_history, persona, model, last_agent_response):
    """Uses an LLM to generate the tester bot's next message."""

    # Limitar o histórico para evitar confusão (últimas 8 mensagens)
    recent_history = chat_history[-8:] if len(chat_history) > 8 else chat_history

    # Adicionar um lembrete explícito sobre o papel
    enhanced_messages = [
        {"role": "system", "content": persona},
        {"role": "system", "content": f"""REMINDER: You are responding as a HUMAN USER to this AI's message:
"{last_agent_response[:200]}..."

Respond naturally and briefly (1-3 sentences). Do NOT copy their phrasing or structure."""},
        *recent_history
    ]

    print("🧠 Tester bot is thinking...")
    try:
        response = await litellm.acompletion(
            model=model,
            messages=enhanced_messages,
            temperature=0.7,  # Aumentado para mais variação
            max_tokens=200,  # Reduzido para respostas mais curtas
            top_p=0.90,
            frequency_penalty=0.3,  # Penaliza repetições
            presence_penalty=0.3  # Encoraja novos tópicos
        )
        reply = response.choices[0].message.content.strip()

        # NOVA SALVAGUARDA: Verifica se a resposta do LLM está vazia
        if not reply:
            print("⚠️ LLM do Tester Bot retornou uma resposta vazia. Usando fallback.")
            return "Interessante. Você pode me explicar isso com outras palavras?"

        # Validação: detectar se copiou muito do agente
        if len(reply) > 4000:  # Aumentado para acompanhar as respostas mais longas da CEAF
            print("⚠️  Response too long, truncating...")
            reply = reply[:4000] + "..."

        print("✅ Tester bot generated a reply.")
        return reply
    except Exception as e:
        print(f"❌ Tester bot's LLM failed: {e}")
        return "Hmm, interesting. Tell me more?"


# --- Main Test Runner ---

async def run_test_session(config):
    """Runs a full, autonomous test session against a CEAF agent."""

    async with httpx.AsyncClient() as client:
        # 1. Login to get token
        token = await login(client, config["AURA_API_URL"], config["AURA_USERNAME"], config["AURA_PASSWORD"])
        if not token:
            return

        # 2. Initialize test state
        agent_id = config["TARGET_AGENT_ID"]
        max_turns = config["MAX_TURNS"]

        full_conversation_log = []
        tester_chat_history = []

        current_turn = 0
        ceaf_session_id = str(uuid.uuid4())
        next_message_from_tester = config["INITIAL_PROMPT"]

        print("\n" + "=" * 50)
        print(f"🚀 Starting Autonomous Test Session")
        print(f"    Target Agent ID: {agent_id}")
        print(f"    Max Turns: {max_turns}")
        print(f"    Tester Model: {config['TESTER_BOT_MODEL']}")
        print("=" * 50 + "\n")

        # 3. Main conversation loop
        while current_turn < max_turns:
            print(f"\n--- Turn {current_turn + 1}/{max_turns} ---")

            # Tester bot sends its message
            print(f"\n🤖 Tester Bot says:\n{next_message_from_tester}")
            full_conversation_log.append({"role": "tester_bot", "content": next_message_from_tester})
            tester_chat_history.append({"role": "user", "content": next_message_from_tester})

            # Get CEAF agent's response
            ceaf_response, updated_session_id = await chat_with_agent(
                client, config["AURA_API_URL"], token, agent_id, next_message_from_tester, ceaf_session_id
            )
            ceaf_session_id = updated_session_id

            print(f"\n👤 CEAF Agent says:\n{ceaf_response}")
            full_conversation_log.append({"role": "ceaf_agent", "content": ceaf_response})
            tester_chat_history.append({"role": "assistant", "content": ceaf_response})

            current_turn += 1

            if current_turn >= max_turns:
                break

            # Generate the tester bot's next reply (COM CONTEXTO MELHORADO)
            next_message_from_tester = await generate_tester_reply(
                tester_chat_history,
                config["TESTER_BOT_PERSONA"],
                config["TESTER_BOT_MODEL"],
                ceaf_response  # Passa a última resposta para evitar cópia
            )

            # Pequena pausa para evitar rate limiting
            await asyncio.sleep(8)

    # 4. Save the conversation log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_run_{agent_id}_{timestamp}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(full_conversation_log, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("🏁 Test Session Complete")
    print(f"💾 Conversation log saved to: {filename}")
    print("=" * 50)


# --- Command-Line Interface ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an autonomous test session against a CEAF agent.")
    parser.add_argument("--agent-id", type=str, help=f"The ID of the CEAF agent to test.")
    parser.add_argument("--turns", type=int, help=f"The number of conversational turns.")
    parser.add_argument("--model", type=str, help=f"The LLM model for the tester bot.")
    parser.add_argument("--prompt", type=str, help=f"The initial prompt to start the conversation.")

    args = parser.parse_args()

    if args.agent_id:
        CONFIG["TARGET_AGENT_ID"] = args.agent_id
    if args.turns:
        CONFIG["MAX_TURNS"] = args.turns
    if args.model:
        CONFIG["TESTER_BOT_MODEL"] = args.model
    if args.prompt:
        CONFIG["INITIAL_PROMPT"] = args.prompt

    if not CONFIG["TARGET_AGENT_ID"]:
        print("❌ Error: No target agent ID specified. Please set it in the script or use --agent-id.")
    elif not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY is not set in your .env file.")
    else:
        asyncio.run(run_test_session(CONFIG))