# Databricks notebook source
# MAGIC %md
# MAGIC # Registrar, Avaliar com MLflow Tracing e Fazer Deploy do Agente BIA (Model Serving)
# MAGIC Registra o `PortfolioAssistantAgent` (`ResponsesAgent` instrumentado com `@mlflow.trace`) no Unity Catalog (`workspace.estudos_ai.beatriz_portfolio_agent`), gera traces detalhados no experimento MLflow (`/Shared/portfolio-estudos/05_agentes/experimentos/portfolio_agent`), persiste histórico de traces em `workspace.estudos_ai.bia_agent_traces` e atualiza o endpoint `beatriz-portfolio-agent`.

# COMMAND ----------
import json
import os
import time
from datetime import datetime, timezone
import mlflow
from mlflow.models.resources import DatabricksServingEndpoint
from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    TrafficConfig,
    Route,
)

w = WorkspaceClient()
current_user = spark.sql("SELECT current_user() AS user").first().user
assert current_user, "Usuário autenticado necessário"

spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.estudos_ai COMMENT 'Laboratório pessoal Aurora; IA e Agentes'")

AGENT_DIR = f"/Workspace/Users/{current_user}/portfolio-estudos/05_agentes/portfolio_agent"
AGENT_FILE = f"{AGENT_DIR}/agent.py"
EXPERIMENT_PATH = f"/Users/{current_user}/portfolio-estudos/05_agentes/experimentos/portfolio_agent"
w.workspace.mkdirs(EXPERIMENT_PATH.rsplit("/", 1)[0])

MODEL_NAME = "workspace.estudos_ai.beatriz_portfolio_agent"
ENDPOINT_NAME = "beatriz-portfolio-agent"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
exp = mlflow.set_experiment(EXPERIMENT_PATH)
exp_id = exp.experiment_id
os.environ["MLFLOW_EXPERIMENT_ID"] = str(exp_id)

try:
    from mlflow.entities import MlflowExperimentLocation
    mlflow.tracing.set_destination(MlflowExperimentLocation(experiment_id=str(exp_id)))
except Exception:
    pass

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT)]
input_example = {
    "input": [
        {"role": "user", "content": "Olá BIA! Quem é a Beatriz Helena, qual é o projeto mais legal dela no Databricks e qual o WhatsApp dela?"}
    ]
}

with mlflow.start_run(run_name="register_bia_agent_with_tracing") as run:
    mlflow.log_params({
        "agent_name": "BIA",
        "llm_endpoint": LLM_ENDPOINT,
        "uc_model_name": MODEL_NAME,
        "serving_endpoint": ENDPOINT_NAME,
        "mlflow_tracing_enabled": True,
        "experiment_id": str(exp_id),
    })
    info = mlflow.pyfunc.log_model(
        name="agent",
        python_model=AGENT_FILE,
        resources=resources,
        input_example=input_example,
        pip_requirements=[
            f"mlflow=={mlflow.__version__}",
            "databricks-sdk>=0.81.0",
            "pydantic>=2.0.0",
        ],
        registered_model_name=MODEL_NAME,
    )

client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = str(max(int(v.version) for v in versions))
client.set_registered_model_alias(MODEL_NAME, "prod", latest_version)

# Carregar o modelo instrumentado e executar bateria de perguntas para registrar Traces no MLflow Experiment
loaded = mlflow.pyfunc.load_model(info.model_uri)

TRACE_PROMPTS = [
    "Oi BIA! Tudo bem? Quem é você e quem é a Beatriz Helena?",
    "Qual é o projeto mais legal da Beatriz no Databricks?",
    "Você tem o WhatsApp e o e-mail da Beatriz para eu entrar em contato?",
    "Como funciona a arquitetura do projeto 03 de RAG com Databricks Vector Search e Qwen3?",
    "Por que devo contratar a Beatriz para o meu time de Engenharia de IA?",
]

trace_records = []
with mlflow.start_run(run_name="bia_trace_evaluation_suite") as eval_run:
    for idx, q in enumerate(TRACE_PROMPTS, start=1):
        t0 = time.time()
        pred = loaded.predict({"input": [{"role": "user", "content": q}]})
        latency_ms = round((time.time() - t0) * 1000.0, 1)
        reply_str = str(pred)
        trace_records.append({
            "trace_index": idx,
            "experiment_id": str(exp_id),
            "run_id": eval_run.info.run_id,
            "model_version": latest_version,
            "question": q,
            "response_preview": reply_str[:600],
            "latency_ms": float(latency_ms),
            "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        })
    mlflow.log_metrics({
        "traced_queries_count": len(TRACE_PROMPTS),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in trace_records) / len(trace_records), 1),
    })

spark.createDataFrame(trace_records).write.mode("overwrite").saveAsTable("workspace.estudos_ai.bia_agent_traces")

# Atualizar o endpoint no Model Serving incluindo variáveis de ambiente de Tracing
served_entity_name = f"beatriz_portfolio_agent-{latest_version}"
served_entities = [
    ServedEntityInput(
        name=served_entity_name,
        entity_name=MODEL_NAME,
        entity_version=latest_version,
        workload_size="Small",
        scale_to_zero_enabled=True,
        environment_vars={
            "MLFLOW_EXPERIMENT_ID": str(exp_id),
            "ENABLE_MLFLOW_TRACING": "true",
        },
    )
]
traffic_config = TrafficConfig(
    routes=[Route(served_model_name=served_entity_name, traffic_percentage=100)]
)

existing_endpoints = {e.name for e in w.serving_endpoints.list()}
if ENDPOINT_NAME not in existing_endpoints:
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(
            name=ENDPOINT_NAME,
            served_entities=served_entities,
            traffic_config=traffic_config,
        ),
    )
    action = "created"
else:
    w.serving_endpoints.update_config(
        name=ENDPOINT_NAME,
        served_entities=served_entities,
        traffic_config=traffic_config,
    )
    action = "updated"

result = {
    "status": "success",
    "model_name": MODEL_NAME,
    "model_version": latest_version,
    "experiment_id": str(exp_id),
    "experiment_path": EXPERIMENT_PATH,
    "traces_logged": len(trace_records),
    "endpoint_name": ENDPOINT_NAME,
    "endpoint_action": action,
}
print(json.dumps(result, ensure_ascii=False))
dbutils.notebook.exit(json.dumps(result, ensure_ascii=False))
