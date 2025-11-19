# Em: ceaf_core/translators/human_to_genlang.py

import asyncio
import json
import re
from pydantic import ValidationError
from ceaf_core.genlang_types import IntentPacket, GenlangVector
from ceaf_core.utils.embedding_utils import get_embedding_client
from ceaf_core.services.llm_service import LLMService, LLM_MODEL_FAST
from ceaf_core.utils.common_utils import extract_json_from_text
import logging

logger = logging.getLogger("CEAFv3_System")


class HumanToGenlangTranslator:
    def __init__(self):
        self.embedding_client = get_embedding_client()
        self.llm_service = LLMService()

    # Copie e cole esta função inteira, substituindo a existente.
    async def translate(self, query: str, metadata: dict) -> IntentPacket:
        """
        Versão V1.2: Usa um prompt robusto com exemplos (few-shot) para garantir
        uma análise de intenção consistente e evitar falhas de parsing de JSON.
        """
        logger.info(f"--- [HTG Translator v1.2] Analisando query humana com LPU robusta: '{query[:50]}...' ---")

        # === MUDANÇA: PROMPT REFINADO COM EXEMPLOS E INSTRUÇÕES MAIS CLARAS ===
        analysis_prompt = f"""
                You are a linguistic analyst. Your task is to analyze the user's query and extract its core components into a structured JSON object.

                **Instructions:**
                1.  **core_query:** Rephrase the user's query into a clear, self-contained question or statement.
                2.  **intent_description:** Describe the user's primary goal (e.g., "seeking an opinion", "requesting factual information", "making a social greeting").
                3.  **emotional_tone_description:** Describe the user's likely emotional state (e.g., "curious", "frustrated", "friendly", "neutral").
                4.  **key_entities:** Extract the 1-3 most important nouns or concepts.

                **Example 1:**
                User Query: "e o que você pensa sobre isso?"
                JSON Output:
                {{
                  "core_query": "What is your opinion on the previous topic of conversation?",
                  "intent_description": "seeking the assistant's opinion on the preceding context",
                  "emotional_tone_description": "follow-up curiosity",
                  "key_entities": ["opinion", "previous topic"]
                }}

                **Example 2:**
                User Query: "quais seus valores centrais?"
                JSON Output:
                {{
                  "core_query": "What are your core values?",
                  "intent_description": "requesting information about the assistant's core principles",
                  "emotional_tone_description": "inquisitive",
                  "key_entities": ["core values", "principles"]
                }}

                **Your Task:**
                Analyze the following user query and respond ONLY with the valid JSON object.

                User Query: "{query}"
                JSON Output:
                """
        # ==================== FIM DA MUDANÇA ====================

        analysis_json = None
        analysis_str = await self.llm_service.ainvoke(LLM_MODEL_FAST, analysis_prompt, temperature=0.0)

        try:
            extracted_json = extract_json_from_text(analysis_str)
            if isinstance(extracted_json, dict):
                required_keys = ["core_query", "intent_description", "emotional_tone_description", "key_entities"]
                if all(key in extracted_json for key in required_keys):
                    analysis_json = extracted_json
                else:
                    logger.warning(
                        f"HTG Translator: Invalid JSON structure (missing keys). Raw: '{analysis_str[:150]}'")
            else:
                logger.warning(
                    f"HTG Translator: Failed to extract a dictionary from LLM response. Raw: '{analysis_str[:150]}'")

        except Exception as e:
            logger.error(f"HTG Translator: Exception during JSON parsing. Error: {e}. Raw: '{analysis_str[:150]}'")

        # Fallback aprimorado: Se a análise falhar, usa a query bruta, mas ainda tenta extrair keywords.
        if not analysis_json:
            logger.error("HTG Translator: Falha na análise da LPU. Usando fallback aprimorado.")
            fallback_keywords = list(set(re.findall(r'\b\w{3,15}\b', query.lower())))
            analysis_json = {
                "core_query": query,
                "intent_description": "unknown_intent",
                "emotional_tone_description": "unknown_emotion",
                "key_entities": fallback_keywords[:3]  # Pega até 3 palavras-chave
            }

        texts_to_embed = [
                             analysis_json.get("core_query", query),
                             analysis_json.get("intent_description", "unknown"),
                             analysis_json.get("emotional_tone_description", "unknown")
                         ] + analysis_json.get("key_entities", [])

        embeddings = await self.embedding_client.get_embeddings(texts_to_embed, context_type="default_query")

        query_vector = GenlangVector(vector=embeddings[0], source_text=analysis_json.get("core_query", query),
                                     model_name=self.embedding_client._resolve_model_name("default_query"))
        intent_vector = GenlangVector(vector=embeddings[1], source_text=analysis_json.get("intent_description"),
                                      model_name=self.embedding_client._resolve_model_name("default_query"))
        emotional_vector = GenlangVector(vector=embeddings[2],
                                         source_text=analysis_json.get("emotional_tone_description"),
                                         model_name=self.embedding_client._resolve_model_name("default_query"))
        entity_vectors = [GenlangVector(vector=emb, source_text=text,
                                        model_name=self.embedding_client._resolve_model_name("default_query")) for
                          text, emb in zip(analysis_json.get("key_entities", []), embeddings[3:])]

        intent_packet = IntentPacket(
            query_vector=query_vector,
            intent_vector=intent_vector,
            emotional_valence_vector=emotional_vector,
            entity_vectors=entity_vectors,
            metadata=metadata
        )

        logger.info(
            f"--- [HTG Translator] Análise completa. Intenção: '{intent_vector.source_text}', Entidades: {[e.source_text for e in entity_vectors]} ---")
        return intent_packet