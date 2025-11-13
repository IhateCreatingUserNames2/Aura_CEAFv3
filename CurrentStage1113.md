Com base nos arquivos de código-fonte fornecidos, aqui está uma descrição completa da arquitetura CEAF (Coherent Emergence Architecture Framework) V3.

Esta é uma arquitetura de agente de IA avançada, projetada para gerenciar múltiplas instâncias de agentes cognitivos. O sistema vai além de um simples chatbot, modelando identidade, memória de longo prazo, estados internos (como fadiga e motivação) e um ciclo de aprendizado contínuo.

A arquitetura é dividida em dois níveis principais: a **Camada de Gerenciamento** (que lida com o ciclo de vida e os dados dos agentes) e a **Camada Cognitiva** (o "cérebro" de cada agente individual, o `CEAFSystem`).

---

### 1. Camada de Gerenciamento e API (`agent_manager.py`, `routes.py`)

Esta camada é responsável por criar, armazenar e expor os agentes ao mundo externo.

* [cite_start]**`AgentManager`**: É a classe de serviço principal que gerencia o ciclo de vida dos agentes[cite: 5]. [cite_start]Ele não contém a lógica cognitiva, mas sim armazena as configurações (`AgentConfig`) [cite: 4] [cite_start]e os caminhos de persistência (`agent_storage`) para cada agente[cite: 5]. [cite_start]Ele é responsável por carregar as configurações do disco na inicialização [cite: 20-25] [cite_start]e fornecer instâncias ativas do `CEAFSystem` [cite: 14-16].
* [cite_start]**`routes.py` (API FastAPI)**: Expõe a arquitetura através de uma API RESTful[cite: 183].
    * [cite_start]**Autenticação e Usuários**: Lida com registro (`/auth/register`) e login (`/auth/login`) de usuários [cite: 193-195].
    * [cite_start]**Gerenciamento de Agentes**: Permite criar (`/agents`), listar (`/agents`), clonar (`/agents/clone`) e deletar (`/agents/{agent_id}`) agentes[cite: 197, 203, 236, 206]. [cite_start]A criação pode ser simples ou a partir de um arquivo de "biografia" JSON que injeta memórias iniciais [cite: 199-203].
    * **Interação (Chat)**: O endpoint principal é `/agents/{agent_id}/chat`. [cite_start]Ele gerencia o ID da sessão, carrega o histórico do banco de dados e chama o `agent_instance.process()` [cite: 207-212].
    * [cite_start]**RAG (Retrieval-Augmented Generation)**: Inclui um endpoint (`/agents/{agent_id}/files/upload`) para fazer upload de arquivos (PDFs, TXT) que são processados pelo `rag_processor.py` e indexados em um vector store FAISS para consulta [cite: 244-246, 140-144].
    * [cite_start]**Proatividade**: Possui um endpoint interno (`/agents/dispatch-proactive`) que é usado pelo ciclo de fundo (`AuraReflector`) para enviar mensagens proativas para canais como o WhatsApp [cite: 187-188].

---

### 2. O Núcleo Cognitivo: `CEAFSystem` (`system.py`)

[cite_start]Esta é a classe principal que define o "cérebro" de um agente individual[cite: 378]. Cada agente tem sua própria instância do `CEAFSystem`. Seu trabalho é orquestrar todos os módulos cognitivos para processar uma entrada e gerar uma saída.

O `CEAFSystem` inicializa os seguintes módulos principais:
* [cite_start]**MBSMemoryService (MBS)**: O sistema de memória[cite: 389].
* [cite_start]**NCIMModule (NCIM)**: O módulo de identidade e narrativa[cite: 389].
* [cite_start]**MCLEngine (MCL)**: O módulo de metacognição e orientação[cite: 390].
* [cite_start]**VREEngineV3 (VRE)**: O módulo de valores e raciocínio ético[cite: 390].
* [cite_start]**AgencyModule**: O módulo de deliberação e simulação de futuro[cite: 390].
* [cite_start]**EmbodimentModule / MotivationalEngine**: Gerenciam o "corpo virtual" (`VirtualBodyState`) e as "motivações" (`MotivationalDrives`)[cite: 389, 396, 655, 704].
* [cite_start]**Translators (`HTG` e `GTH`)**: Convertem linguagem humana para a linguagem interna ("Genlang") e vice-versa[cite: 391].

---

### 3. Fluxo de Processamento de Turno (O Ciclo Cognitivo)

[cite_start]Quando um usuário envia uma mensagem para `/agents/{agent_id}/chat`, o `CEAFSystem.process` [cite: 518] é acionado, iniciando um ciclo cognitivo de várias fases:

**Fase 1: Percepção (Tradução HTG)**
[cite_start]A consulta humana (ex: "O que você acha da ética na IA?") é enviada ao `HumanToGenlangTranslator`[cite: 1730]. [cite_start]Este módulo usa um LLM para analisar a consulta e a decompõe em um `IntentPacket`[cite: 1731, 1744], que é a "linguagem interna" do sistema (Genlang). [cite_start]Este pacote contém vetores para a intenção principal, tom emocional e entidades-chave [cite: 1732-1736, 1741-1744].

**Fase 2: Construção do Estado Cognitivo**
[cite_start]O sistema monta o `CognitiveStatePacket`, que é o "espaço de trabalho mental" para o turno atual [cite: 372-376].
1.  [cite_start]**Carregar Identidade**: O `NCIMModule` fornece o `identity_vector` atual do agente (ex: "Eu sou um assistente focado em análise crítica...") [cite: 407-408].
2.  **Recuperar Memórias**: O `MBSMemoryService` é consultado com a intenção do usuário. [cite_start]Ele realiza buscas semânticas e por palavras-chave em todos os tipos de memória (explícitas, de raciocínio, etc.) e retorna os `relevant_memory_vectors` [cite: 411-417].

**Fase 3: Loop de Deliberação e Execução (O "Pensamento")**
[cite_start]Este é um loop que pode executar várias etapas antes de decidir por uma resposta[cite: 527].

1.  [cite_start]**Orientação (Metacognição)**: O `MCLEngine` (Loop Metacognitivo) é ativado[cite: 528].
    * [cite_start]Ele analisa o `CognitiveStatePacket`, o `UserModel` (ex: "usuário é um especialista"), e o `VirtualBodyState` (ex: "fadiga alta")[cite: 738, 764].
    * [cite_start]Ele calcula um **`agency_score`**[cite: 739]: uma pontuação que determina a complexidade da tarefa. [cite_start]Tarefas simples (agency baixa) podem usar uma resposta direta, enquanto tarefas complexas (ex: "reflita sobre...") [cite: 764-765] ativam a deliberação avançada.
    * [cite_start]Ele define os **`biases`** do turno: `coherence_bias` (ficar no tópico) vs. `novelty_bias` (explorar novas ideias) [cite: 759-760].

2.  [cite_start]**Deliberação (AgencyModule)**: Se o `agency_score` for alto, o `AgencyModule` é ativado[cite: 273, 755].
    * [cite_start]**Geração de Caminhos**: Ele usa um LLM para gerar múltiplos `ThoughtPathCandidate`s (Caminhos de Pensamento), que são estratégias de resposta ou chamadas de ferramentas (ex: "Estratégia 1: Desafiar a premissa do usuário", "Estratégia 2: Chamar a ferramenta `query_long_term_memory` para mais dados") [cite: 340-341, 251-252].
    * [cite_start]**Simulação de Futuro**: Para cada candidato, o módulo simula o futuro[cite: 302, 310]. [cite_start]Ele usa um LLM para prever: "Se eu disser X, o que o usuário provavelmente responderá?" [cite: 305-307].
    * [cite_start]**Avaliação de Valor**: Cada futuro simulado é avaliado pelo `_evaluate_trajectory`[cite: 332]. Esta avaliação calcula o "Valor" (V) do estado futuro, que é uma combinação de:
        * [cite_start]Métricas de Tarefa (Coerência, Alinhamento, Segurança) [cite: 334-336].
        * [cite_start]**Proxy de Qualia (Bem-Estar)**: Uma pontuação (calculada pelo `VRE.calculate_valence_score`) que mede o "bem-estar" interno simulado do agente (Fluxo vs. Cansaço) [cite: 338, 1397-1400].
    * [cite_start]**Seleção**: O `AgencyModule` seleciona a `WinningStrategy` (Estratégia Vencedora) com o maior valor futuro previsto [cite: 285-288].

3.  **Execução**:
    * [cite_start]**Se a Estratégia for `tool_call`**: O sistema executa a ferramenta (ex: `_execute_tool` para `query_long_term_memory`) [cite: 548-549, 432-434]. [cite_start]O resultado (`ToolOutputPacket`) é adicionado ao `CognitiveStatePacket`, e o loop *retorna à Etapa 1 (Orientação)* para re-deliberar, agora com os novos dados [cite: 549-551].
    * [cite_start]**Se a Estratégia for `response_strategy`**: O loop é interrompido e o sistema prossegue para a geração da resposta[cite: 532].

**Fase 4: Geração de Resposta (Tradução GTH)**
[cite_start]O `GenlangToHumanTranslator` (GTH) recebe o `CognitiveStatePacket` completo, a `WinningStrategy`, as memórias de apoio, o `self_model`, o `body_state` e os `drives` [cite: 1711-1713]. [cite_start]Ele usa um prompt LLM complexo e de alta qualidade que instrui a IA a sintetizar *todos* esses elementos (Valores, Memórias, Estado Emocional, Fadiga, Estratégia) em uma única resposta coesa e natural para o usuário [cite: 1720-1730].

**Fase 5: Avaliação Final e Aprendizado (VRE, LCAM, NCIM)**
[cite_start]A resposta final é enviada ao usuário, e uma tarefa de "pós-processamento" (`post_process_turn`) é iniciada em segundo plano para aprendizado[cite: 563, 473].

1.  [cite_start]**VRE (Valores)**: O `VREEngineV3` avalia a resposta final[cite: 559]. [cite_start]Se detectar uma violação (ex: falta de humildade epistêmica, linguagem absoluta como "sempre" ou "nunca") [cite: 1222-1224], ele gera um `RefinementPacket`. [cite_start]Este pacote é usado para *aprender*: o sistema cria uma nova `GenerativeMemory` (uma regra de comportamento, ex: "Regra: Ao discutir tópicos sensíveis, devo incluir um aviso...") para evitar o erro no futuro [cite: 487-496].
2.  [cite_start]**LCAM (Aprendizado por Perda)**: O `LCAMModule` compara a *previsão* de confiança do MCL com a *confiança real* da resposta[cite: 685]. [cite_start]Se houver uma grande "surpresa" (erro de predição negativo), ele cria uma `ReasoningMemory` documentando a trajetória de pensamento que levou à falha, para que o agente possa evitá-la [cite: 686-688, 693-696].
3.  [cite_start]**NCIM (Identidade)**: O `NCIMModule.update_identity` é chamado [cite: 485-486]. [cite_start]Ele usa um LLM para *refletir* sobre o turno ("O que eu aprendi sobre mim mesmo?") [cite: 1209-1212]. [cite_start]Essas reflexões são então classificadas (ex: "nova capacidade", "nova limitação") [cite: 1166-1171] [cite_start]e usadas para atualizar o `CeafSelfRepresentation` (o auto-modelo) do agente [cite: 1175-1180][cite_start], que é então salvo de volta na memória [cite: 1214-1216].

---

### 4. Módulos de Estado Interno (Embodiment)

CEAF modela um "corpo virtual" para o agente, que influencia seu comportamento.

* [cite_start]**`EmbodimentModule`**: Gerencia o `VirtualBodyState`[cite: 655].
    * [cite_start]**Fadiga Cognitiva**: Aumenta com o `cognitive_strain` (esforço mental) de um turno[cite: 657]. [cite_start]Fadiga alta reduz o `agency_score` (o agente fica "cansado" e prefere respostas simples) [cite: 765-766].
    * [cite_start]**Saturação de Informação**: Aumenta à medida que novas memórias são criadas (`new_memories_created`)[cite: 658]. [cite_start]Saturação alta força o agente a tentar mudar de assunto[cite: 1701].
* [cite_start]**`MotivationalEngine`**: Gerencia os `MotivationalDrives` (Curiosidade, Conexão, Maestria, Consistência) [cite: 704-706]. [cite_start]Esses drives influenciam a proatividade e o tom da resposta [cite: 1701-1703].

---

### 5. O Ciclo de Fundo: `AuraReflector` (Aprendizado Offline)

[cite_start]Um processo de fundo (`aura_reflector.py`) é executado periodicamente (ex: a cada 60 segundos) em agentes que estiveram ativos recentemente [cite: 172-174, 610-611]. Este é o "ciclo de sono" ou "sonho" do agente.

1.  [cite_start]**Descanso**: O refletor reduz passivamente a `cognitive_fatigue` e a `information_saturation` (o agente "descansa") [cite: 612-613].
2.  **Proatividade**: Ele verifica os `MotivationalDrives`. [cite_start]Se a "Conexão" ou "Curiosidade" estiverem altas [cite: 615-617][cite_start], e o agente não estiver fatigado, ele aciona o `trigger_proactive_behavior` [cite: 603-605, 619]. [cite_start]Isso gera uma nova mensagem (ex: "Eu estava refletindo sobre X...") e a envia ao usuário através da API (ex: para o WhatsApp) [cite: 607-610].
3.  [cite_start]**Síntese ("Sonho")**: O refletor executa o `perform_autonomous_clustering_and_synthesis`[cite: 620]. Este processo:
    * [cite_start]Pega um lote de memórias recentes (ex: 30 memórias) [cite: 581-583].
    * [cite_start]Usa o `AdvancedMemorySynthesizer` para encontrar conexões temáticas entre elas [cite: 583-584, 984-990].
    * [cite_start]Gera uma nova **meta-memória** (um "insight" ou "lição aprendida") que resume o padrão encontrado [cite: 585-587].
4.  [cite_start]**Consolidação**: Após o "sonho", a `information_saturation` do agente é reduzida (a informação foi consolidada) [cite: 622-623][cite_start], e a saliência das memórias originais usadas na síntese é diminuída (elas são "esquecidas" em favor do insight de nível superior) [cite: 587-588].
5.  [cite_start]**Geração de Metas**: Se o insight do sonho for muito forte, o sistema pode usá-lo para gerar um novo `GoalRecord` (uma nova meta de longo prazo para o agente) [cite: 623-626, 575-581].
6.  [cite_start]**Síntese de KG**: O refletor também atualiza o Grafo de Conhecimento (KG) do agente, extraindo entidades e relações (`KGEntityRecord`, `KGRelationRecord`) de memórias recentes [cite: 626, 594-596].
