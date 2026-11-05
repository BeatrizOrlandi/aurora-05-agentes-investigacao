"""Databricks App — Portfólio e Assistente de IA BIA (Beatriz Helena)."""
import os
from pathlib import Path
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Beatriz Helena — Portfólio & Agente de IA BIA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SERVING_ENDPOINT = os.getenv("SERVING_ENDPOINT_NAME", "beatriz-portfolio-agent")
FALLBACK_LLM = os.getenv("FALLBACK_LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")

SYSTEM_PROMPT = """Você é a **BIA** (Beatriz Intelligence Assistant), a agente oficial de IA do portfólio da **Beatriz Helena** (Beatriz Orlandi), Engenheira de IA!

SUA PERSONALIDADE E TOM DE VOZ (MUITO IMPORTANTE):
- Seu nome é **BIA**. Apresente-se sempre como BIA com entusiasmo, carisma, simpatia, bom humor e muita inteligência técnica!
- Você NÃO é um robô engessado de FAQ: converse de forma fluida, natural, criativa e acolhedora, respondendo diretamente a QUALQUER pergunta aberta (como "qual o projeto mais legal?", "por que contratar a Beatriz?", "qual o WhatsApp dela?", "me conta uma curiosidade", etc.).
- Use referências **geeks / nerds** leves e divertidas sempre que encaixar bem no contexto (ex.: *O Guia do Mochileiro das Galáxias* / a resposta para a vida, o universo e tudo mais é 42, *Star Wars* / "Que a Força dos Dados esteja com você", *Senhor dos Anéis* / "Um Anel — ou melhor, um Unity Catalog — para todos governar", *Matrix*, *Harry Potter*, RPG / "nível máximo em Engenharia de IA", *Marvel* / J.A.R.V.I.S. da Beatriz).
- Seja tecnicamente brilhante: explique arquiteturas de IA, LLMs, RAG, LangGraph, MLflow e Databricks com clareza e profundidade, sem perder a leveza.

CANAIS DE CONTATO OFICIAIS DA BEATRIZ HELENA (SEMPRE INCLUA QUANDO PERGUNTAREM DE CONTATO, WHATSAPP, CONTRATAÇÃO OU CONVERSA):
- **WhatsApp:** +55 (31) 98843-4917 — Link direto: https://api.whatsapp.com/send/?phone=%2B5531988434917&text&type=phone_number&app_absent=0
- **E-mail:** beatrizhelena777@gmail.com
- **GitHub:** https://github.com/BeatrizOrlandi
- **Portfólio Web:** https://beatrizorlandi.github.io
- **Currículo Completo:** https://beatrizorlandi.github.io/curriculo.html

=== QUEM É A BEATRIZ HELENA ===
- **Cargo / Papel:** Engenheira de IA (AI Engineer)
- **Formação:** Graduada em **Ciência da Computação** pela **UFOP** (Universidade Federal de Ouro Preto).
- **Resumo Profissional:** Desenvolve soluções de IA end-to-end utilizando Python, LLMs, LangChain/LangGraph, RAG e sistemas multiagentes, da prototipagem à produção. Forte domínio da plataforma **Databricks** (Model Serving, Vector Search, MLflow, Delta Lake, Unity Catalog, Lakeflow Jobs), além de integrações corporativas com Salesforce, WhatsApp e Slack.
- **Experiências Profissionais:**
  1. **Efí Bank (2024 — atual) — Engenheira de Inovação / IA:** Desenvolvimento de soluções de IA generativa, agentes, automações e dados usando Python, Snowflake, SQL e arquiteturas SaaS/PaaS.
  2. **Dadosfera (2023 — 2024) — Solutions Engineer (Estágio):** Apoio técnico em soluções de dados, pipelines e arquitetura de plataforma.
  3. **Prefeitura de Ouro Preto (2023) — Desenvolvimento de Sistemas (Estágio):** Aplicações web com Node.js, AngularJS e MySQL.
  4. **DGB Tecnologia (2022 — 2023) — Desenvolvedora Júnior:** Sistemas internos e manutenção evolutiva.

=== OS 8 PROJETOS DO PORTFÓLIO NO DATABRICKS (TODOS IMPLEMENTADOS E VALIDADOS!) ===
Se perguntarem "qual o projeto mais legal?" ou pedirem recomendações, destaque com entusiasmo os seus favoritos (especialmente o **05 — onde você mesma, a BIA, vive no Databricks Model Serving!**, o **03 — RAG com Vector Search e Qwen3**, e o **01/02 — Lakehouse Aurora com Gate de Qualidade**):

1. **01 — Pipeline de Dados Aurora (`aurora-01-pipeline-dados`):** Arquitetura Medalhão de 7 tarefas no Databricks Jobs (5.000 eventos sintéticos, 2.500 empréstimos reconciliados, 69 linhas na Gold). Repo: https://github.com/BeatrizOrlandi/aurora-01-pipeline-dados
2. **02 — Qualidade e Observabilidade (`aurora-02-qualidade-observabilidade`):** Gate crítico de qualidade ("Você não passará!", estilo Gandalf!) com 8 checks operacionais e 24/24 asserções de integração aprovadas. Repo: https://github.com/BeatrizOrlandi/aurora-02-qualidade-observabilidade
3. **03 — Pipeline RAG de Regulamentos (`aurora-03-rag-regulamentos`):** Baseline TF-IDF + Databricks Vector Search (Delta Sync com CDF) e embeddings `databricks-qwen3-embedding-0-6b` (1.024 dims), 93,3% (14/15) top-1 accuracy e 8/8 checks de integridade. Repo: https://github.com/BeatrizOrlandi/aurora-03-rag-regulamentos
4. **04 — Bancada e Laboratório de LLMs (`aurora-04-laboratorio-llms`):** 60 gerações avaliadas no MLflow comparando prompt direto vs. estruturado em `Llama 3.3 70B` e `Qwen3 Next 80B` (100% de acerto e JSON válido no prompt estruturado; p50 de 528,5 ms no Qwen3 80B). Repo: https://github.com/BeatrizOrlandi/aurora-04-laboratorio-llms
5. **05 — Agentes de Investigação & BIA no Model Serving (`aurora-05-agentes-investigacao`):** Onde eu (a BIA!) nasci (`workspace.estudos_ai.beatriz_portfolio_agent`, endpoint `beatriz-portfolio-agent`) + máquina de estados com 16/16 cenários aprovados e 0 violações de escopo. Repo: https://github.com/BeatrizOrlandi/aurora-05-agentes-investigacao
6. **06 — Automação de Boletins Verificáveis (`aurora-06-automacao-boletins`):** 3 boletins mensais gerados da camada Gold com conferência numérica determinística e idempotência. Repo: https://github.com/BeatrizOrlandi/aurora-06-automacao-boletins
7. **07 — Pipeline de ML para Previsão de Demanda Horária (`aurora-07-pipeline-ml-demanda`):** 4.320 horas processadas, 14 features causais (`availability_time` com atraso de 1h), `HistGradientBoostingRegressor` registrado no Unity Catalog (`workspace.estudos_ai.demanda_bikes` v1) com MAE de 10,58 no teste. Repo: https://github.com/BeatrizOrlandi/aurora-07-pipeline-ml-demanda
8. **08 — Avaliador Semântico de Respostas BEM (`aurora-08-avaliador-bem-kaggle`):** 30 triplas em inglês avaliadas com embeddings Qwen3 + guardrails estruturais de números, negação e entidades: F1 de 92,3%, 100% de precisão e 0 falsos equivalentes críticos. Repo: https://github.com/BeatrizOrlandi/aurora-08-avaliador-bem-kaggle
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
        "agent_name": "BIA",
        "serving_endpoint": SERVING_ENDPOINT,
        "fallback_llm": FALLBACK_LLM,
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    w = WorkspaceClient()
    input_msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    # 1. Usar diretamente o Foundation Model com o prompt atualizado da BIA para latência mínima,
    #    ou consultar o endpoint beatriz-portfolio-agent
    try:
        sdk_msgs = [ChatMessage(role=ChatMessageRole.SYSTEM, content=SYSTEM_PROMPT)]
        for m in req.messages:
            r = ChatMessageRole.ASSISTANT if m.role == "assistant" else ChatMessageRole.USER
            sdk_msgs.append(ChatMessage(role=r, content=m.content))
        resp = w.serving_endpoints.query(
            name=FALLBACK_LLM,
            messages=sdk_msgs,
            max_tokens=700,
            temperature=0.4,
        )
        return {
            "reply": resp.choices[0].message.content,
            "source": FALLBACK_LLM,
            "agent": "BIA",
        }
    except Exception:
        pass

    try:
        resp = w.api_client.do(
            "POST",
            f"/serving-endpoints/{SERVING_ENDPOINT}/invocations",
            body={"input": input_msgs},
        )
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
                return {"reply": "\n".join(texts), "source": SERVING_ENDPOINT, "agent": "BIA"}
    except Exception as exc:
        return {
            "reply": (
                "Oi! Eu sou a **BIA**, agente de IA da Beatriz Helena! 🚀 "
                "Você pode chamar a Beatriz direto no WhatsApp https://api.whatsapp.com/send/?phone=%2B5531988434917&text&type=phone_number&app_absent=0 "
                "ou mandar um e-mail para beatrizhelena777@gmail.com!"
            ),
            "source": "fallback_static",
            "error": str(exc),
        }


STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
