# whatsapp_bridge/aura_client.py
import httpx
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
AURA_API_BASE_URL = os.getenv("AURA_API_URL", "http://localhost:8009/ceaf")
logging.info(f"AURA_CLIENT INICIADO. Usando URL da API: {AURA_API_BASE_URL}")

async def login(username, password):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{AURA_API_BASE_URL}/auth/login",
            json={"username": username, "password": password}
        )
        if response.status_code == 200:
            return response.json()
        return None

async def register_user(email, username, password):
    payload = {"email": email, "username": username, "password": password}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{AURA_API_BASE_URL}/auth/register", json=payload)
            if response.status_code == 201:
                return response.json()
            else:
                return {"error": response.json().get("detail", "Unknown registration error.")}
        except httpx.RequestError as e:
            return {"error": f"Network error during registration: {e}"}

async def get_my_agents(token):
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{AURA_API_BASE_URL}/agents", headers=headers)
        if response.status_code == 200:
            return response.json()
        return []

async def chat_with_agent(token, agent_id, message):
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message}
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(f"{AURA_API_BASE_URL}/agents/{agent_id}/chat", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        return None

async def get_public_agents(token: str):
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{AURA_API_BASE_URL}/prebuilt-agents/list", headers=headers)
        if response.status_code == 200:
            return response.json()
        return []

async def clone_agent(token: str, source_agent_id: str, custom_name: str):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"source_agent_id": source_agent_id, "custom_name": custom_name, "clone_memories": True}
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(f"{AURA_API_BASE_URL}/agents/clone", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
             return {"error": "Source agent not found or is not public."}
        else:
            try:
                return {"error": response.json().get("detail", "Unknown cloning error.")}
            except:
                return {"error": "An unknown error occurred during cloning."}

# ==================== NEW FUNCTIONS START ====================

async def get_available_models(token: str):
    """Fetches the curated list of available models from the API."""
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{AURA_API_BASE_URL}/models/openrouter", headers=headers)
        if response.status_code == 200:
            return response.json()
        return None

async def update_agent_model(token: str, agent_id: str, new_model: str):
    """Updates the model for a specific agent."""
    headers = {"Authorization": f"Bearer {token}"}
    # The API uses the /profile endpoint for this update
    payload = {"model": new_model}
    async with httpx.AsyncClient() as client:
        response = await client.put(f"{AURA_API_BASE_URL}/agents/{agent_id}/profile", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            try:
                return {"error": response.json().get("detail", "Failed to update model.")}
            except:
                return {"error": "Failed to update model due to an unknown API error."}

# ===================== NEW FUNCTIONS END =====================