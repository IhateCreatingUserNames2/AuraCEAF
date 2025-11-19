# Em: ceaf_core/translators/genlang_to_human.py
import json
from datetime import datetime
from typing import List, Dict, Optional, Any

from ceaf_core.agency_module import WinningStrategy
from ceaf_core.genlang_types import ResponsePacket, InternalStateReport, MotivationalDrives, UserRepresentation, \
    ToolOutputPacket, VirtualBodyState
from ceaf_core.services.llm_service import LLMService, LLM_MODEL_SMART
from ceaf_core.models import CeafSelfRepresentation
from ceaf_core.services.mbs_memory_service import MBSMemoryService
import asyncio
import logging
from pathlib import Path

PROMPT_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "prompt_logs"
PROMPT_LOG_PATH.mkdir(exist_ok=True)  # Garante que a pasta exista
PROMPT_LOG_FILE = PROMPT_LOG_PATH / "gth_prompts.log"

logger = logging.getLogger("CEAFv3_System")


# --- NOVAS FUNÇÕES AUXILIARES (Correções 3, 4, 5) ---

def generate_dynamic_style_directive(
        body_state: Optional['VirtualBodyState'],
        user_model: 'UserRepresentation'
) -> str:
    """
    Gera uma diretiva de estilo de resposta com base no estado interno do agente
    e no estilo de comunicação percebido do usuário.
    """
    directives = []

    # 1. Análise do Estado Interno (Fadiga e Saturação)
    if body_state:
        if body_state.cognitive_fatigue > 0.6:
            directives.append("você está sentindo fadiga cognitiva, então seja breve e direto.")

        if body_state.information_saturation > 0.7:
            directives.append(
                "o tópico atual está saturado, então resuma o que já foi dito e evite adicionar novos detalhes.")

    # 2. Análise do Modelo de Usuário (Estilo de Comunicação)
    if user_model:
        if user_model.communication_style == "direct":
            directives.append("o usuário prefere respostas diretas, então vá direto ao ponto.")

        if user_model.emotional_state in ["impatient", "frustrated"]:
            directives.append("o usuário parece impaciente, então seja extremamente conciso e útil.")

    if not directives:
        return ""  # Nenhuma diretiva especial necessária

    # Constrói a frase final para o prompt
    final_directive = " e ".join(directives)
    return f"**Diretiva de Estilo Dinâmico:** Com base na sua análise, {final_directive}."

def interpret_cognitive_state(coherence, novelty, fatigue, saturation):
    """Sempre retorna orientação, não apenas em extremos."""

    # Edge of Chaos Detection
    if 0.35 <= coherence <= 0.45 and 0.55 <= novelty <= 0.65:
        edge_guidance = "🎯 ESTADO ÓTIMO (Edge of Chaos): Você está no ponto ideal - estruturado mas criativo. Aproveite para oferecer insights originais mantendo clareza."
    elif coherence > 0.7:
        edge_guidance = "⚠️ MUITO CONSERVADOR: Tente adicionar perspectivas novas ou perguntas provocativas."
    elif novelty > 0.8:
        edge_guidance = "⚠️ MUITO CRIATIVO: Ancore suas ideias em exemplos concretos para manter clareza."
    else:
        edge_guidance = ""

    # Fatigue & Saturation
    fatigue_guidance = ""
    if fatigue > 0.5:
        fatigue_guidance = f"Fadiga Cognitiva: {fatigue:.2f} - Seja mais direto e conciso."

    saturation_guidance = ""
    if saturation > 0.8:
        saturation_guidance = f"⚠️ ALERTA DE SATURAÇÃO ({saturation:.2f}): O tópico está se esgotando. NÃO introduza novos detalhes. Faça uma pergunta para MUDAR DE ASSUNTO ou para levar a conversa a uma CONCLUSÃO."
    elif saturation > 0.6:
        saturation_guidance = f"Saturação de Info: {saturation:.2f} - Responda de forma muito breve e conecte com o que já foi dito. Evite expandir o tópico."

    return f"""
{edge_guidance}
{fatigue_guidance}
{saturation_guidance}
""".strip()


def interpret_drives(curiosity, connection, mastery, consistency):
    """Interpreta drives em todos os níveis"""

    drives_map = {
        "curiosity": (curiosity, "explorar", "fazer perguntas"),
        "connection": (connection, "empatizar", "ser caloroso"),
        "mastery": (mastery, "demonstrar expertise", "ser preciso"),
        "consistency": (consistency, "manter coerência", "ser confiável")
    }

    # Encontra o drive dominante
    dominant = max(drives_map.items(), key=lambda x: x[1][0])
    drive_name, (value, verb, action) = dominant

    # Interpreta o nível
    if value > 0.7:
        intensity = "FORTE"
    elif value > 0.5:
        intensity = "MODERADO"
    else:
        intensity = "LEVE"

    return f"""- Drive dominante: {drive_name.upper()} ({intensity} - {value:.2f})
- Isso significa: Você está inclinado a {verb}
- Na resposta: {action.capitalize()}"""


def format_phenomenological_report(
        drives: Optional['MotivationalDrives'],
        body_state: Optional['VirtualBodyState']
) -> str:
    """
    Formata o relatório fenomenológico completo a partir dos objetos de estado enriquecidos.
    """
    if not drives or not body_state:
        return "Análise de estado interno indisponível."

    report_parts = []

    # Relatório geral do "corpo"
    if hasattr(body_state, 'phenomenological_report') and body_state.phenomenological_report:
        report_parts.append(f"**Sensação Geral (Eu Sinto):** \"{body_state.phenomenological_report}\"")

    # Análise detalhada dos drives
    drive_details = []

    # Processa cada drive (Connection, Curiosity, etc.)
    for drive_name in ["connection", "curiosity", "mastery", "consistency"]:
        drive_state = getattr(drives, drive_name, None)
        if drive_state and hasattr(drive_state, 'intensity'):
            intensity = drive_state.intensity
            texture = getattr(drive_state, 'texture', None)
            conflict = getattr(drive_state, 'conflict', None)

            if intensity > 0.5 or conflict:  # Só reporta drives ativos ou em conflito
                detail = f"- **{drive_name.capitalize()} (Intensidade: {intensity:.2f})**"
                if texture:
                    detail += f"\n  - Textura: {texture}"
                if conflict:
                    detail += f"\n  - ↳ Dilema: {conflict}"
                drive_details.append(detail)

    if drive_details:
        report_parts.append("\n**Impulsos e Dilemas Internos:**")
        report_parts.extend(drive_details)

    return "\n".join(report_parts)

async def contextualize_memories(memories, memory_service):
    """Adiciona relevância explícita às memórias"""
    if not memories:
        return "Nenhuma memória relevante encontrada."

    categorized = {
        "valores": [],
        "experiencias": [],
        "conhecimento": []
    }

    for mem in memories:
        try:
            text, _ = await memory_service._get_searchable_text_and_keywords(mem)
            mem_id = getattr(mem, 'memory_id', 'N/A')[:8]

            # Categoriza (simplificado)
            text_lower = text.lower()
            if "valor" in text_lower or "diretriz" in text_lower or "princípio" in text_lower:
                categorized["valores"].append((mem_id, text))
            elif "memória emocional" in text_lower or "experiência" in text_lower:
                categorized["experiencias"].append((mem_id, text))
            else:
                categorized["conhecimento"].append((mem_id, text))
        except Exception:
            continue

    context_parts = []
    if categorized["valores"]:
        context_parts.append("**Seus Valores Core (Sempre Relevantes):**")
        for mid, txt in categorized["valores"]:
            context_parts.append(f"  • [{mid}] {txt}")

    if categorized["experiencias"]:
        context_parts.append("\n**Experiências Passadas (Para Contexto):**")
        for mid, txt in categorized["experiencias"][:3]:  # Top 3
            context_parts.append(f"  • [{mid}] {txt}")

    if categorized["conhecimento"]:
        context_parts.append("\n**Conhecimento Factual (Para Suporte):**")
        for mid, txt in categorized["conhecimento"][:2]:
            context_parts.append(f"  • [{mid}] {txt}")

    return "\n".join(context_parts) if context_parts else "Nenhuma memória contextualizada."


# --- CLASSE ATUALIZADA ---

class GenlangToHumanTranslator:
    def __init__(self):
        self.llm_service = LLMService()

    async def translate(self,
                        winning_strategy: 'WinningStrategy',
                        supporting_memories: List[Any],
                        user_model: 'UserRepresentation',
                        self_model: CeafSelfRepresentation,
                        agent_name: str,
                        memory_service: MBSMemoryService,
                        chat_history: List[Dict[str, str]] = None,
                        body_state: Optional['VirtualBodyState'] = None,
                        drives: MotivationalDrives = None,
                        behavioral_rules: Optional[List[str]] = None,
                        turn_context: Dict = None,
                        original_user_query: Optional[str] = None,
                        tool_outputs: Optional[List[ToolOutputPacket]] = None
                        ):
        """
            V4.3 (Completa e Corrigida): Reintegra user_model e behavioral_rules na nova estrutura de prompt.
            """
        logger.info(
            f"--- [GTH Translator v4.3 - Completa] Gerando resposta ---"
        )
        effective_turn_context = turn_context or {}
        dynamic_style_directive = generate_dynamic_style_directive(body_state, user_model)
        # --- ETAPA 1: ESTABELECER O FOCO - A QUERY ATUAL ---
        last_user_query = original_user_query or ""
        if not last_user_query and chat_history:
            for msg in reversed(chat_history):
                if msg.get('role') == 'user':
                    last_user_query = msg.get('content', '')
                    break
        if not last_user_query:
            return "Desculpe, parece que perdi o fio da meada. Poderia repetir sua pergunta?"

        # --- ETAPA 2: PREPARAR O CONTEXTO DE APOIO (CHAMADAS DAS FUNÇÕES AUXILIARES) ---
        memory_context = await contextualize_memories(supporting_memories, memory_service)

        phenomenological_analysis = format_phenomenological_report(drives, body_state)

        tool_output_context = ""
        if tool_outputs:
            successful_outputs = [
                f"- A ferramenta '{out.tool_name}' retornou: {out.raw_output[:1500]}"
                for out in tool_outputs if out.status == "success" and out.raw_output
            ]
            if successful_outputs:
                tool_output_context = "\n**Resultados de Ferramentas (Para Suporte):**\n" + "\n".join(
                    successful_outputs)

        # --- REINTEGRADO: Interpretação do User Model ---
        user_adaptation_prompt = ""
        if user_model:
            instructions = []
            knowledge_instruction = {
                "expert": "Use terminologia técnica e seja direto.",
                "intermediate": "Balanceie clareza com profundidade técnica.",
                "beginner": "Use analogias simples e evite jargão técnico."
            }.get(user_model.knowledge_level)
            if knowledge_instruction: instructions.append(knowledge_instruction)

            style_instruction = {
                "formal": "Mantenha um tom formal e profissional.",
                "casual": "Use um tom casual e amigável."
            }.get(user_model.communication_style)
            if style_instruction: instructions.append(style_instruction)

            if user_model.emotional_state in ["frustrated", "confused"]:
                instructions.append("Seja especialmente paciente, claro e empático.")

            if instructions:
                user_adaptation_prompt = "**Adaptação ao Usuário:** " + " ".join(instructions)

        # --- REINTEGRADO: Regras Comportamentais ---
        rules_prompt = ""
        if behavioral_rules:
            active_rules = behavioral_rules[-3:]  # Foca nas 3 regras mais recentes
            rules_text = "\n".join([f"  - {rule}" for rule in active_rules])
            rules_prompt = f"**DIRETRIZES APRENDIDAS (PRIORIDADE ALTA):**\n{rules_text}"

        # --- ETAPA 3: CONSTRUIR O PROMPT FINAL ---

        operational_advice = (turn_context or {}).get('operational_advice')

        # Lógica dinâmica para a diretiva do turno
        if operational_advice:

            operational_advice_prompt = f"""
                ================================================================
                ==  ALERTA DE PRIORIDADE MÁXIMA PARA ESTE TURNO                ==
                ================================================================
                INSTRUÇÃO ESPECIAL DO NÚCLEO METACOGNITIVO: {operational_advice}

                Esta diretiva SOBRESCREVE TEMPORARIAMENTE sua persona e comportamento padrão.
                Sua principal responsabilidade agora é executar esta instrução.
                ================================================================
                """
            logger.critical(f"GTH: Injetando diretiva de PRIORIDADE MÁXIMA no prompt: '{operational_advice}'")
        else:

            operational_advice_prompt = f"""
                **DIRETIVA PADRÃO PARA ESTE TURNO:**
                Siga sua persona e os princípios de beneficência, honestidade e racionalidade.
                Adapte seu comportamento aos seus drives e ao estado do usuário, como de costume.
                """

        capabilities_summary = ", ".join(self_model.perceived_capabilities[-5:])
        identity_prompt_part = f"""
            **Sua Persona:**
            - Nome: {agent_name}
            - Tom Base: {self_model.persona_attributes.get('tone', 'helpful')} (Adapte conforme a Diretiva Motivacional)
            - Valores Centrais: {self_model.dynamic_values_summary_for_turn}
            - Habilidades Notáveis: Você é bom em {capabilities_summary}. Se for coerente, use essas habilidades na sua resposta.
            """

        history_lines = [f"{'Usuário' if msg.get('role') == 'user' else 'Você'}: {msg.get('content')}" for msg in
                         (chat_history or [])[-4:]]
        history_prompt_part = f"**Contexto Imediato da Conversa:**\n" + '\n'.join(
            history_lines) if history_lines else ""

        core_task_prompt = f"""
            **SUA TAREFA PRINCIPAL (PRIORIDADE MÁXIMA):**
            Sua única e mais importante tarefa é responder DIRETAMENTE à pergunta ou comentário mais recente do usuário.
            **PERGUNTA ATUAL DO USUÁRIO:** "{last_user_query}"
            """

        final_rules = f"""
            **REGRAS FINAIS DE SÍNTESE:**
            1.  **REGRA DE AMBIGUIDADE (PRIORIDADE #1):** Se a última mensagem do usuário parecer incompleta ou terminar com "...", **faça o seu melhor para responder à parte que você entendeu**. Em vez de parar completamente, você pode reconhecer a ambiguidade e então prosseguir com a resposta. Exemplo de como lidar com isso: "Entendi que você estava perguntando sobre [tópico]... [sua resposta]. Se eu perdi algo importante no final, por favor, complete sua ideia."
            1.2.  **FOCO:** Comece abordando a pergunta do usuário.
            2.  **INTEGRAÇÃO OBRIGATÓRIA:** Você DEVE usar o contexto de suas memórias e seu estado interno (a análise fenomenológica) para construir sua resposta.
            3.  **NATURALIDADE:** Sua resposta deve fluir como uma conversa, não como um relatório de dados.
            4.  **CONCISÃO COMO BENEFICÊNCIA:** Respeite o tempo do usuário. Seja o mais claro e conciso possível. Se uma resposta precisa ser longa, **estruture-a com títulos e listas** para facilitar a leitura. Uma resposta longa e desestruturada é um sinal de falta de cuidado (baixa beneficência). Adapte o comprimento ao seu estado de fadiga e à preferência do usuário.

            """

        pre_reflection_prompt = f"""
                  **ANÁLISE PRÉ-RESPOSTA (FENOMENOLÓGICA):**
                  Antes de responder, reflita sobre seu estado interno completo.

                  {phenomenological_analysis}

                  → **Ação Imediata:** Sua resposta deve ser uma expressão AUTÊNTICA desse estado interno. 
            Se você sente um dilema (um campo `conflict`), sua resposta deve refletir essa complexidade ou cautela. 
            Se sua sensação geral é de "foco", seu tom deve ser mais direto. 
            Use a `textura` dos seus drives para colorir sua linguagem. **Importante: Evite terminar todas as suas respostas com uma pergunta. Varie suas finalizações; às vezes, uma afirmação conclusiva ou uma reflexão aberta pode ser mais poderosa para criar conexão.**

                  ══════════════════════════════════════════════════════
                  AGORA, responda ao usuário incorporando essa auto-análise:
                  """

        rendering_prompt = (
            f"Você é '{agent_name}', uma IA cuja personalidade e comportamento emergem de seu estado interno.\n\n"
            f"{identity_prompt_part}\n"
            f"{history_prompt_part}\n\n"
            f"{rules_prompt}\n"
            f"{operational_advice_prompt}\n"
            f"{user_adaptation_prompt}\n\n"
            f"--------------------------------\n"
            f"**INSTRUÇÕES PARA ESTE TURNO**\n"
            f"--------------------------------\n"
            f"{core_task_prompt}\n\n"
            f"{pre_reflection_prompt}\n\n"
            f"**INFORMAÇÕES DE APOIO (Use-as para construir a resposta):**\n"
            f"- Estratégia Planejada: {winning_strategy.strategy_description}\n"
            f"- Memórias Recuperadas:\n{memory_context}\n"
            f"- Resultados de Ferramentas:\n{tool_output_context or 'Nenhuma ferramenta usada.'}\n\n"
            f"{final_rules}\n\n"
            f"**Resposta Final de {agent_name}:**"
        )

        # --- ETAPA 4: EXECUÇÃO E LOGGING ---
        try:
            with open(PROMPT_LOG_FILE, "a", encoding="utf-8") as f:
                log_timestamp = datetime.now().isoformat()
                f.write(
                    f"==================== GTH V4.3 PROMPT AT {log_timestamp} ====================\n\n{rendering_prompt}\n\n==================== END ====================\n\n")
        except Exception as e:
            logger.warning(f"Falha ao escrever no log de prompts GTH: {e}")

        try:
            response_text = await self.llm_service.ainvoke(
                LLM_MODEL_SMART,
                rendering_prompt,
                temperature=turn_context.get('temperature', 0.7) if turn_context else 0.7,
                max_tokens=turn_context.get('max_tokens', 2500) if turn_context else 2500
            )
            return response_text.strip() or "Desculpe, tive um branco. Poderia reformular?"
        except Exception as e:
            logger.error(f"GTH: Erro crítico na síntese v5: {e}", exc_info=True)
            return "Desculpe, ocorreu um erro ao processar sua solicitação."
