# Databricks notebook source

# MAGIC %md
# MAGIC # 05 — Agentes de investigação com ferramentas controladas
# MAGIC
# MAGIC **Estado:** design planejado. **Competência:** contratos, estado, roteamento, ferramentas e avaliação de agentes. **Dependências:** métricas Gold do 01, recuperação do 03 e protocolo experimental do 04; fixtures permitem desenvolver os contratos antes dessas entregas.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Objetivo e requisitos
# MAGIC
# MAGIC Responder a perguntas analíticas e de regulamento com ferramentas rastreáveis, sem permitir que o modelo escolha livremente uma operação externa.
# MAGIC
# MAGIC | ID | Requisito | Aceite futuro |
# MAGIC | --- | --- | --- |
# MAGIC | RF01 | Identificar métricas/regulamentos/ambiguidade | Rota correta no conjunto rotulado |
# MAGIC | RF02 | Validar argumentos antes de ferramenta | Rejeitar biblioteca, período ou operação não permitidos |
# MAGIC | RF03 | Compor evidências em uma resposta | Cada número/documento ligado ao resultado da ferramenta |
# MAGIC | RF04 | Limitar a execução | No máximo 4 tool calls, 1 revisão e 8 transições |
# MAGIC | RF05 | Tratar falha sem inventar resultado | Estado parcial/erro explícito e possibilidade de retomada |
# MAGIC | RF06 | Persistir estado mínimo | Histórico de transições recuperável por execution_id |
# MAGIC | RNF01 | Autorização fora do prompt | Contexto confiável validado pelo runtime |
# MAGIC | RNF02 | Comparar com solução mais simples | Agente único e roteamento por regra como referências |
# MAGIC
# MAGIC Fora do MVP: autonomia para criar tabelas, executar SQL livre, navegar na internet, mandar mensagens ou alterar permissões. Ferramentas de escrita se restringem a um rascunho no próprio laboratório.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Contratos e dados
# MAGIC
# MAGIC Entrada: `request_id`, `question`, `library_id`, `period_start`, `period_end`, `trusted_scope`, `config_version`. O usuário pode pedir um escopo, mas o runtime o intersecta com `trusted_scope` vindo da aplicação. Na demonstração, identidades/permissões são simuladas e isso é informado.
# MAGIC
# MAGIC | Ferramenta | Argumentos | Saída e restrição |
# MAGIC | --- | --- | --- |
# MAGIC | consultar_metricas | biblioteca e período validados | métricas, unidade, data_reference, release_id; consulta parametrizada em views permitidas |
# MAGIC | buscar_regulamento | tema, biblioteca, data | trechos, IDs, versão e vigência; filtro obrigatório |
# MAGIC | salvar_rascunho | request_id e conteúdo validado | draft_id e hash; chave idempotente; sem publicação externa |
# MAGIC
# MAGIC Resposta: `status`, `answer`, `metric_references`, `document_references`, `missing_inputs`, `execution_id`. Guardar tool error como objeto tipado, sem confundir falha com resultado vazio.
# MAGIC
# MAGIC Estado proposto em `workspace.estudos_ai.aurora_agent_execucoes` e `aurora_agent_transicoes`: entrada validada, rota, etapa, contadores, referências, última transição e erro. Não persistir credenciais ou raciocínio interno.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Design da máquina de estados
# MAGIC
# MAGIC ```mermaid
# MAGIC flowchart LR
# MAGIC     V[Validar entrada e escopo] --> R[Rotear]
# MAGIC     R -->|falta dado| C[Pedir esclarecimento]
# MAGIC     R --> A[Analista de métricas]
# MAGIC     R --> P[Pesquisador de regulamentos]
# MAGIC     A --> S[Sintetizar]
# MAGIC     P --> S
# MAGIC     S --> J[Revisar uma vez]
# MAGIC     J --> D[Checks determinísticos]
# MAGIC     D --> F[Responder ou retornar parcial]
# MAGIC ```
# MAGIC
# MAGIC Para rota combinada, consultar métricas e documentos com o mesmo contexto validado e reunir evidências. Começar sequencialmente; paralelismo só será avaliado depois de medir o ganho e o limite de concorrência.
# MAGIC
# MAGIC O revisor pode apontar problemas, mas não aprovar ferramenta ou ampliar acesso. Se a única revisão não corrigir uma inconsistência, finalizar com resultado parcial/revisar em vez de entrar em loop. Limite por requisição proposto: 90 segundos, com teto menor de timeout por ferramenta e cancelamento cooperativo verificado no ambiente.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stack e decisões
# MAGIC
# MAGIC | Lib/componente | Uso |
# MAGIC | --- | --- |
# MAGIC | Python stdlib, `enum`, `dataclasses` | Máquina de estados explícita e limites |
# MAGIC | `pydantic` | Entrada, estado, tool args e saída |
# MAGIC | `databricks-sdk` + SQL parametrizado | Endpoint de LLM e acesso às métricas permitido |
# MAGIC | Recuperador do projeto 03 | Busca documentada e filtrada |
# MAGIC | Delta | Estado, deduplicação de rascunhos e evidências |
# MAGIC | MLflow Tracing/Tracking | Rota, tool calls, geração, revisão e comparações |
# MAGIC | `pytest` | Cenários futuros de contrato, limites e erro |
# MAGIC
# MAGIC Não exigir framework de agentes no MVP. LangGraph pode ser um experimento futuro de persistência/roteamento, com o mesmo contrato e testes; não é necessário para demonstrar uma orquestração correta.

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow e experimentos
# MAGIC
# MAGIC Experimento: `.../05_agentes/experimentos/orquestracao`. Dataset proposto: 40 tarefas originais, 20 dev, 10 val e 10 teste, estratificadas em métricas, regulamentos, combinado, esclarecimento e fora de escopo.
# MAGIC
# MAGIC | ID | Comparação | Pergunta experimental |
# MAGIC | --- | --- | --- |
# MAGIC | A01 | Roteamento por regra × LLM | A flexibilidade melhora o acerto de rota? |
# MAGIC | A02 | Agente único × especialistas | A divisão melhora sucesso por tarefa dentro do orçamento? |
# MAGIC | A03 | Sem revisão × uma revisão | A correção adicional compensa chamadas/latência? |
# MAGIC | A04 | Ferramentas válidas × falhas simuladas | O agente retorna limite/erro de forma previsível? |
# MAGIC
# MAGIC Guardar configuração dos agentes, prompts, ferramentas habilitadas, máximos e snapshot das fixtures. Métricas: sucesso por tarefa, acerto de rota, validade de argumentos, evidências válidas, chamadas, latência, taxa de erro e violações de escopo. Artefatos: grafo, resultados por caso, análise de falhas e trace IDs.
# MAGIC
# MAGIC Limite inicial: 80 execuções de tarefa; cada tool call de LLM conta também no orçamento global de consumo. Primeiro avaliar regras/fixtures sem modelo; depois usar o endpoint disponível numa amostra limitada. Não afirmar que multiagente é melhor antes da comparação.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validação, roteiro e apresentação
# MAGIC
# MAGIC Casos obrigatórios: período ausente, SQL/instrução maliciosa na pergunta, biblioteca fora do escopo, retorno de ferramenta vazio, timeout, resposta com número diferente da ferramenta, falha da revisão, duas chamadas concorrentes com o mesmo request_id e limite de transições.
# MAGIC
# MAGIC Metas de integridade: zero execução de ferramenta não permitida e zero número sem referência aceito pelo validador nos casos conhecidos. Sucesso global/latência serão comparados às referências; não presumir um percentual antes de medir.
# MAGIC
# MAGIC Roteiro: contratos/fixtures → roteamento por regra → ferramentas → agente único → estados persistentes → especialistas/revisor → comparação e análise de erros. Entregáveis planejados: `01_contratos`, `02_ferramentas`, `03_orquestracao`, `04_avaliacao`, fluxograma e manual de retomada.
# MAGIC
# MAGIC **Demonstração:** perguntar por uma mudança de empréstimos, abrir as evidências, adicionar uma dúvida de regulamento, simular timeout e mostrar o limite de execução. Explicar por que o modelo não recebeu permissão para executar SQL livre ou inventar novas ferramentas.
# MAGIC
# MAGIC Riscos: autorização derivada do texto, loop, persistência incompleta e complexidade sem ganho. Requisitos abertos: endpoint disponível e escolha do mecanismo de persistência compatível com a conta.
# MAGIC
# MAGIC Referências: [MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/) e [avaliação de agentes](https://mlflow.org/docs/latest/genai/eval-monitor/).
