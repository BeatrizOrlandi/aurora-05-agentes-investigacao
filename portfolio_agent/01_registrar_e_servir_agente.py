# Databricks notebook source
# MAGIC %md
# MAGIC # Registrar e Fazer Deploy do Agente do Portfólio (Model Serving)
# MAGIC Registra o `PortfolioAssistantAgent` (`ResponsesAgent`) no Unity Catalog (`workspace.estudos_ai.beatriz_portfolio_agent`), configura passthrough authentication para o endpoint LLM `databricks-meta-llama-3-3-70b-instruct` e cria/atualiza o endpoint `beatriz-portfolio-agent` no Databricks Model Serving.

# COMMAND ----------
import json
import os
import time
from pathlib import Path
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

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT)]
input_example = {
    "input": [
        {"role": "user", "content": "Olá! Quem é a Beatriz Helena, quais projetos ela já validou no Databricks e como entro em contato?"}
    ]
}

with mlflow.start_run(run_name="register_portfolio_agent") as run:
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

# Teste rápido do modelo carregado no próprio notebook antes do endpoint
loaded = mlflow.pyfunc.load_model(info.model_uri)
test_pred = loaded.predict(input_example)

# Criar ou atualizar o endpoint no Model Serving (com scale_to_zero_enabled=True)
served_entity_name = f"beatriz_portfolio_agent-{latest_version}"
served_entities = [
    ServedEntityInput(
        name=served_entity_name,
        entity_name=MODEL_NAME,
        entity_version=latest_version,
        workload_size="Small",
        scale_to_zero_enabled=True,
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
    "run_id": run.info.run_id,
    "endpoint_name": ENDPOINT_NAME,
    "endpoint_action": action,
    "sample_prediction": str(test_pred)[:500],
}
print(json.dumps(result, ensure_ascii=False))
dbutils.notebook.exit(json.dumps(result, ensure_ascii=False))
