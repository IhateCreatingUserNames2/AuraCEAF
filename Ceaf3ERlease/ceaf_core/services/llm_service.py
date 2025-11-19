import asyncio
import os
import logging
import litellm
from typing import Tuple
import aiohttp
import json

logger = logging.getLogger("LLMService")

# Configuração do LiteLLM
litellm.api_key = os.getenv("OPENROUTER_API_KEY")
litellm.api_base = "https://openrouter.ai/api/v1"

# Constantes de modelo
LLM_MODEL_FAST = "openrouter/openrouter/sherlock-think-alpha"
LLM_MODEL_SIMULATION = "openrouter/openrouter/sherlock-think-alpha"
LLM_MODEL_SMART = "openrouter/openrouter/sherlock-think-alpha"


class LLMService:
    """Wrapper para chamadas ao LLM, seguindo o princípio "LLMs como Ferramentas"."""

    async def ainvoke_with_logprobs(self, model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 2000):
        """
        Invoca um modelo LLM e TENTA solicitar logprobs.

        IMPORTANTE: OpenRouter via LiteLLM NÃO suporta logprobs!
        Esta função faz uma requisição direta à API do OpenRouter para obter logprobs.

        Retorna o objeto de resposta com logprobs quando disponível.
        """
        logger.info(f"LLMService: Invocando modelo '{model}' com tentativa de logprobs...")

        # Remove o prefixo "openrouter/" se presente
        clean_model = model.replace("openrouter/", "")

        try:
            # Faz requisição DIRETA ao OpenRouter (não via LiteLLM)
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "CEAFv3"
            }

            payload = {
                "model": clean_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "logprobs": True,
                "top_logprobs": 1,
                "provider": {
                    "require_parameters": True
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"OpenRouter API error: {resp.status} - {error_text}")
                        resp.raise_for_status()
                    response_data = await resp.json()

            # Converte a resposta para um formato compatível com LiteLLM
            # (para manter compatibilidade com o resto do código)
            class SimpleResponse:
                def __init__(self, data):
                    self.choices = [SimpleChoice(data['choices'][0])]

            class SimpleChoice:
                def __init__(self, choice_data):
                    self.message = SimpleMessage(choice_data['message'])
                    # Logprobs pode não estar presente mesmo com a requisição
                    self.logprobs = choice_data.get('logprobs')

            class SimpleMessage:
                def __init__(self, message_data):
                    self.content = message_data['content']

            response = SimpleResponse(response_data)

            # Verifica se logprobs veio na resposta
            if response.choices[0].logprobs is not None:
                logger.info(f"✓ Logprobs recebido com sucesso para modelo {clean_model}")
            else:
                logger.warning(f"⚠ Modelo {clean_model} não retornou logprobs (pode não suportar)")

            return response

        except Exception as e:
            error_message = f"Erro na chamada direta com logprobs: {type(e).__name__} - {e}"
            logger.error(f"LLMService: {error_message}", exc_info=True)

            # FALLBACK: Tenta usar LiteLLM sem logprobs
            logger.warning("Fallback: usando LiteLLM sem logprobs")
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                headers={
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "CEAFv3"
                }
            )

            # Adiciona logprobs=None para consistência
            if hasattr(response.choices[0], 'logprobs'):
                pass
            else:
                response.choices[0].logprobs = None

            return response

    async def ainvoke(self, model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 2000,
                      retries: int = 2) -> str:
        """Invoca um modelo LLM com retentativas para robustez."""
        logger.info(f"LLMService: Invocando modelo '{model}' com max_tokens={max_tokens}...")

        for attempt in range(retries + 1):
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    headers={"HTTP-Referer": "http://localhost", "X-Title": "CEAFv3"}
                )
                content = response.choices[0].message.content.strip()
                logger.info(f"LLMService: Resposta recebida com sucesso na tentativa {attempt + 1}.")
                return content

            except (litellm.APIError, litellm.RateLimitError, litellm.Timeout) as e:
                error_message = f"Erro da API LLM na tentativa {attempt + 1}/{retries + 1}: {e}"
                logger.warning(error_message)
                if attempt >= retries:
                    logger.error("LLMService: Máximo de retentativas atingido. Retornando mensagem de erro.")
                    return f"[LLM_ERROR] {e}"
                await asyncio.sleep(2 ** attempt)  # Backoff exponencial

            except Exception as e:
                error_message = f"Erro genérico na chamada LLM na tentativa {attempt + 1}: {type(e).__name__} - {e}"
                logger.error(f"LLMService: {error_message}", exc_info=True)
                if attempt >= retries:
                    return f"[LLM_ERROR] {error_message}"
                await asyncio.sleep(2 ** attempt)

        return "[LLM_ERROR] Falha na chamada ao LLM após múltiplas tentativas."


# ============================================================================
# FUNÇÃO AUXILIAR OTIMIZADA PARA AGENCY_MODULE.PY
# ============================================================================

async def _invoke_simulation_llm_with_logprobs(
        llm_service: LLMService,
        model: str,
        prompt: str
) -> Tuple[str, float]:
    """
    Função auxiliar otimizada para chamar o LLM de simulação com logprobs.
    Retorna (texto_da_resposta, score_de_confiança).

    NOTA: Como OpenRouter não suporta logprobs de forma confiável,
    esta função agora usa heurísticas baseadas no comprimento e
    coerência da resposta para estimar a confiança.
    """
    try:
        response = await llm_service.ainvoke_with_logprobs(
            model=model,
            prompt=prompt,
            temperature=0.6
        )

        text_content = response.choices[0].message.content.strip()
        likelihood_score = 0.5  # Fallback padrão

        # Tenta extrair logprobs se disponível
        choice = response.choices[0]
        logprobs_extracted = False

        if hasattr(choice, 'logprobs') and choice.logprobs is not None:
            logprobs_obj = choice.logprobs

            # Método 1: OpenAI padrão (logprobs.content[].logprob)
            if hasattr(logprobs_obj, 'content') and logprobs_obj.content:
                token_logprobs = [
                    item.logprob
                    for item in logprobs_obj.content
                    if hasattr(item, 'logprob') and item.logprob is not None
                ]

                if token_logprobs:
                    import numpy as np
                    probabilities = [np.exp(lp) for lp in token_logprobs]
                    likelihood_score = float(np.mean(probabilities))
                    logprobs_extracted = True
                    logger.info(f"✓ Logprobs extraídos: {likelihood_score:.4f}")

            # Método 2: Formato de dicionário
            elif isinstance(logprobs_obj, dict) and 'content' in logprobs_obj:
                token_logprobs = []
                for item in logprobs_obj['content']:
                    if isinstance(item, dict) and 'logprob' in item:
                        token_logprobs.append(item['logprob'])

                if token_logprobs:
                    import numpy as np
                    probabilities = [np.exp(lp) for lp in token_logprobs]
                    likelihood_score = float(np.mean(probabilities))
                    logprobs_extracted = True
                    logger.info(f"✓ Logprobs extraídos: {likelihood_score:.4f}")

        # Se não conseguiu extrair logprobs, usa heurística baseada em confiança
        if not logprobs_extracted:
            # Heurística: respostas mais longas e bem formatadas = mais confiança
            word_count = len(text_content.split())

            # Score base pela completude da resposta
            if word_count < 5:
                likelihood_score = 0.3  # Resposta muito curta = baixa confiança
            elif word_count < 20:
                likelihood_score = 0.5  # Resposta curta = confiança média
            elif word_count < 50:
                likelihood_score = 0.65  # Resposta normal = boa confiança
            else:
                likelihood_score = 0.7  # Resposta longa = alta confiança

            # Ajusta pela presença de marcadores de incerteza
            uncertainty_markers = ['talvez', 'possivelmente', 'não tenho certeza',
                                   'maybe', 'perhaps', 'not sure', 'might', 'could']
            if any(marker in text_content.lower() for marker in uncertainty_markers):
                likelihood_score *= 0.85  # Reduz confiança em 15%

            logger.info(
                f"⚠ Usando heurística de confiança (logprobs indisponível): {likelihood_score:.4f} "
                f"(palavras: {word_count})"
            )

        return text_content, likelihood_score

    except Exception as e:
        logger.warning(
            f"⚠ Simulação com logprobs falhou completamente: {e}. "
            f"Usando fallback sem logprobs.",
            exc_info=True
        )
        # Fallback completo
        text_content = await llm_service.ainvoke(model, prompt, temperature=0.6)

        # Aplica mesma heurística de confiança
        word_count = len(text_content.split())
        likelihood_score = min(0.4 + (word_count / 100), 0.7)

        return text_content, likelihood_score