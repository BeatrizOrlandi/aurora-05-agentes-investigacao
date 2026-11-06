# Painel Executivo AI/BI (Lakeview) — Aurora & BIA

Este diretório contém a definição serializada (`.lvdash.json`) do painel executivo **Databricks AI/BI (Lakeview)** que consolida em tempo real as tabelas governadas no **Unity Catalog**:

- `workspace.estudos_ai.aurora_llm_resultados` — Ablação de prompts diretos vs. estruturados (JSON Schema) entre **Meta Llama 3.3 70B** e **Qwen3 Next 80B Instruct**.
- `workspace.estudos_ai.bia_agent_traces` — Telemetria de spans (`@mlflow.trace`) das conversas atendidas pelo agente **BIA** no **Databricks Model Serving** (`beatriz-portfolio-agent`).
- `workspace.estudos_ai.aurora_agent_execucoes` — Auditoria dos 16 cenários da máquina de estados de investigação operacional (100% de rotas e guardrails respeitados).
- `workspace.estudos_gold.bikes_predicoes` — Curva horária de demanda prevista pelo `HistGradientBoostingRegressor` (`MAE 10,58`) contra valores observados.
