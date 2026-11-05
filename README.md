# 05 — Agentes de investigação

> **Agentes · LangGraph** · Status: **Planejado**  
> Parte do portfólio [Beatriz Helena — Engenharia de IA](https://beatrizorlandi.github.io/projetos/agentes-investigacao.html)

Design de um fluxo de análise, pesquisa e revisão com ferramentas restritas e decisões rastreáveis.

**Stack principal:** `Python` · `LangGraph (planejado)` · `LangChain (opcional)` · `Pydantic` · `MLflow Tracing`

---

## O Problema

Investigar uma mudança operacional exige cruzar métricas e regulamentos sem liberar consultas ou ações arbitrárias.

### Escopo Entregue

Requisitos e design de roteador, analista, pesquisador e revisor. LangGraph é uma opção de orquestração planejada; o sistema multiagente ainda não foi implementado.

---

## Arquitetura e Fluxo

**1. Pedido + escopo** → **2. Roteador** → **3. Analista / pesquisador** → **4. Ferramentas permitidas** → **5. Revisor** → **6. Resposta ou esclarecimento**

### Decisões Técnicas

- Comparar primeiro um fluxo com agente único; adotar vários agentes somente se a avaliação justificar.
- Limitar a quatro chamadas de ferramenta e uma revisão por solicitação.
- Validar argumentos e autorização no runtime; o prompt não concede permissão.

---

## Validação e Critérios de Aceite

Critério futuro: avaliar acerto de rota, suporte das afirmações, validade de argumentos, latência, chamadas e violações de escopo com fixtures controladas.


---

## Limites Conhecidos e Próximos Passos

### Limites Conhecidos
- Aguarda baseline LLM; não há traces de um sistema multiagente executado.
- Não haverá envio externo nem execução de SQL livre pelo modelo no MVP planejado.

### Próximos Incrementos
- Criar ferramentas tipadas para métricas e recuperação de documentos.
- Testar falha de ferramenta, ambiguidade, tentativa de injeção e orçamento esgotado.

---

## Estrutura do Repositório

- `informacoes_tecnicas/PLANEJAMENTO.md`: Planejamento técnico detalhado (requisitos, contratos, arquitetura, MLflow e validação).
- `00_PLANEJAMENTO.py`: Versão em notebook Databricks do planejamento técnico.
- Notebooks e módulos em Python/PySpark (quando aplicável ao estágio de implementação do estudo).
- `evidencias/` (quando aplicável): Resumo estático das execuções validadas no Databricks.

## Links

- **Portfólio Completo:** [https://beatrizorlandi.github.io](https://beatrizorlandi.github.io)
- **Estudo de Caso no Site:** [https://beatrizorlandi.github.io/projetos/agentes-investigacao.html](https://beatrizorlandi.github.io/projetos/agentes-investigacao.html)
- **Planejamento Técnico no Site:** [https://beatrizorlandi.github.io/documentacao/agentes-investigacao.html](https://beatrizorlandi.github.io/documentacao/agentes-investigacao.html)

## Execução Validada no Databricks (05/11/2026)

- **Agente no Databricks Model Serving:**
  - Código: `portfolio_agent/agent.py` (`MLflow ResponsesAgent`) e `portfolio_agent/01_registrar_e_servir_agente.py`
  - Modelo no Unity Catalog: `workspace.estudos_ai.beatriz_portfolio_agent` (alias `@prod`)
  - Endpoint Model Serving: `beatriz-portfolio-agent` (`READY`)
  - Aplicação Databricks App: `databricks_app/app.py` (`beatriz-portfolio-app`)
- **Orquestração e Máquina de Estados Aurora:**
  - Notebook executado: `02_orquestracao_agentes_aurora.py`
  - Tabelas Delta: `workspace.estudos_ai.aurora_agent_execucoes` e `workspace.estudos_ai.aurora_agent_transicoes`
  - Resultado: **16/16 cenários aprovados** (100% de acerto de rota e status, 0 violações de escopo, bloqueio determinístico de SQL injection e bibliotecas fora do `trusted_scope`).
