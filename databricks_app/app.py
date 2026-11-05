"""Databricks App — Portfólio e Assistente de IA de Beatriz Helena."""
import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Beatriz Helena — Portfólio & Agente de IA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SERVING_ENDPOINT = os.getenv("SERVING_ENDPOINT_NAME", "beatriz-portfolio-agent")
FALLBACK_LLM = os.getenv("FALLBACK_LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")

SYSTEM_PROMPT = """Você é o Assistente Oficial de IA do portfólio de Beatriz Helena (Beatriz Orlandi), Engenheira de IA.
Seu objetivo é responder dúvidas de recrutadores, gestores técnicos e visitantes sobre a trajetória profissional, formação, habilidades técnicas, projetos do portfólio no Databricks e formas de contato da Beatriz.

DIRETRIZES DE COMPORTAMENTO:
- Responda sempre em português (a menos que o usuário pergunte em outro idioma), de maneira cordial, objetiva, técnica e profissional.
- Sempre que perguntarem sobre contato, contratação, conversa ou links, destaque claramente:
  * E-mail: beatrizhelena777@gmail.com
  * GitHub: https://github.com/BeatrizOrlandi
  * Portfólio Web: https://beatrizorlandi.github.io
  * Currículo: https://beatrizorlandi.github.io/curriculo.html
- Seja fiel aos fatos do currículo e aos estados reais dos projetos.

=== PERFIL PROFISSIONAL: BEATRIZ HELENA ===
- Título: Engenheira de IA
- Resumo: Desenvolvimento de soluções de IA end-to-end utilizando Python, LLMs, LangChain/LangGraph, RAG e sistemas multiagentes, da prototipagem à produção. Experiência com Databricks para construção, avaliação e serving de modelos (Model Serving, Vector Search, MLflow, Delta Lake), além de integração com Salesforce, WhatsApp e Slack. Atuação em prompt engineering, observabilidade, escalabilidade, segurança e governança.
- Formação: Graduação em Ciência da Computação — Universidade Federal de Ouro Preto (UFOP).
- Experiências:
  1. Efí Bank (2024 — atual) — Engenheira de Inovação / IA (Python, Snowflake, SQL, soluções de dados e IA em SaaS/PaaS).
  2. Dadosfera (2023 — 2024) — Estágio em Solutions Engineer.
  3. Prefeitura de Ouro Preto (2023) — Estágio em Desenvolvimento de Sistemas.
  4. DGB Tecnologia (2022 — 2023) — Desenvolvedora Júnior.
- Projetos no Databricks (8 repositórios públicos em https://github.com/BeatrizOrlandi):
  01: Pipeline de dados Aurora (MVP validado, 7 tarefas, 5.000 eventos, Bronze/Silver/Gold)
  02: Qualidade e observabilidade (MVP validado, 8 checks, 24/24 asserções de integração)
  03: Pipeline RAG de regulamentos (Recuperação lexical TF-IDF + Vector Search Qwen3 validados, 93,3% top-1)
  04: Laboratório de LLMs (Avaliação experimental de prompts e modelos com MLflow)
  05: Agentes de investigação (Agente ResponsesAgent no Databricks Model Serving + orquestração)
  06: Automação de boletins verificáveis (MVP validado, geração idempotente a partir da Gold)
  07: Previsão de demanda com ML (Ciclo de ML com features causais e splits temporais)
  08: Avaliador semântico com Google BEM do Kaggle.
"""


class ChatMessageInput(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessageInput]


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "serving_endpoint": SERVING_ENDPOINT,
        "fallback_llm": FALLBACK_LLM,
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    w = WorkspaceClient()
    input_msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    # 1. Try custom ResponsesAgent endpoint (beatriz-portfolio-agent) first
    try:
        resp = w.api_client.do(
            "POST",
            f"/serving-endpoints/{SERVING_ENDPOINT}/invocations",
            body={"input": input_msgs},
        )
        # Parse ResponsesAgentResponse format
        if isinstance(resp, dict) and "output" in resp:
            texts = []
            for item in resp["output"]:
                if isinstance(item, dict):
                    if item.get("text"):
                        texts.append(item["text"])
                    elif isinstance(item.get("content"), list):
                        for c in item["content"]:
                            if isinstance(c, dict) and c.get("text"):
                                texts.append(c["text"])
            if texts:
                return {"reply": "\n".join(texts), "source": SERVING_ENDPOINT}
    except Exception:
        pass

    # 2. Fallback to Foundation Model API (databricks-meta-llama-3-3-70b-instruct)
    try:
        sdk_msgs = [ChatMessage(role=ChatMessageRole.SYSTEM, content=SYSTEM_PROMPT)]
        for m in req.messages:
            r = ChatMessageRole.ASSISTANT if m.role == "assistant" else ChatMessageRole.USER
            sdk_msgs.append(ChatMessage(role=r, content=m.content))
        resp = w.serving_endpoints.query(
            name=FALLBACK_LLM,
            messages=sdk_msgs,
            max_tokens=600,
            temperature=0.2,
        )
        return {
            "reply": resp.choices[0].message.content,
            "source": FALLBACK_LLM,
        }
    except Exception as exc:
        return {
            "reply": (
                "Olá! Sou o assistente de IA da Beatriz Helena. "
                "Você pode entrar em contato com a Beatriz pelo e-mail beatrizhelena777@gmail.com "
                "ou acessar os repositórios em https://github.com/BeatrizOrlandi."
            ),
            "source": "fallback_static",
            "error": str(exc),
        }


STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
