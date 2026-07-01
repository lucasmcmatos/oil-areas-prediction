# CLAUDE.md — Projeto OBSAT: Predição de Áreas com Possível Petróleo

> Este arquivo é o handoff de um planejamento feito em conversa com o Claude
> (app), incluindo pesquisa científica e prototipagem/validação de uma
> abordagem. O objetivo aqui é o Claude Code continuar a implementação real
> no ambiente local do Lucas, seguindo as decisões já tomadas e validadas
> abaixo — mas está aberto a ajustes de estrutura, já que a organização final
> do projeto ainda será definida localmente.

## 0. Status atual da implementação (atualizado em 2026-07-01)

A implementação local (com Claude Code) já está em andamento, seguindo um
plano incremental combinado com o Lucas: cada etapa é implementada, testada
e entregue para revisão — **o Lucas commita manualmente**, o Claude Code não
faz `git commit`/`push`. Isso significa que o estado do working tree pode
estar à frente do último commit em `main`; confira `git log` e `git status`
ao retomar para saber exatamente onde parou.

**Concluído nesta sessão:**

- [x] **Etapa 0 — Housekeeping**: `.gitignore` criado, `pyproject.toml`
  limpo (removidos tensorflow/xgboost/matplotlib/seaborn/jupyter do scaffold
  inicial; adicionados scikit-learn/numpy/pandas/joblib/fastapi/uvicorn +
  pytest/httpx como dev deps), estrutura de pastas criada
  (`data/`, `src/`, `models/`, `api/`, `dashboard/`, `tests/`).
- [x] **Etapa 1 — Gerador de dataset sintético**: `src/generate_dataset.py`
  implementado conforme seção 4, rodado e validado — `data/synthetic_dataset.csv`
  gerado com 12.000 amostras, prevalência de positivos **2.43%** (bate com o
  ~2.4% do protótipo).
- [x] **Etapa 2 — Treino e avaliação**: `src/train_model.py` implementado —
  resultado real: **ROC-AUC 0.874**, MAE de calibração vs. `true_probability`
  **0.108**, feature importances `pressure_hpa≈0.49, methane_ppm≈0.37,
  longitude≈0.08, latitude≈0.06` — consistente com o protótipo (seção 5).
  Modelo salvo em `models/oil_probability_model.joblib`, métricas em
  `models/metrics.json`.
- [x] **Etapa 3 — API FastAPI**: `api/main.py` com `/health`, `/predict`,
  `/predict/batch`. Testado ponta a ponta com uvicorn real: 99.61% perto do
  campo Tupi/Lula, 8.79% em ponto de fundo.
- [x] **Etapa 4 — Dashboard + acesso mobile**: `dashboard/index.html`
  implementado (Leaflet, tema "telemetria de satélite" da seção 6). Depois
  ajustado a pedido do Lucas para uso em campo pelo celular junto com o
  satélite:
  - A API agora **também serve o dashboard** como estático na raiz
    (`app.mount("/", StaticFiles(directory="dashboard", html=True))` em
    `api/main.py`) — um único processo/porta, sem servidor HTTP separado.
  - O JS do dashboard usa `API_BASE = window.location.origin` (não mais
    `http://127.0.0.1:8000` fixo) — funciona ao abrir de qualquer IP,
    inclusive o IP da rede local a partir do celular.
  - CSS com `@media (max-width: 768px)`: em telas estreitas o mapa fica em
    cima e o painel de controle embaixo (rolável), com botões/inputs maiores
    para toque. Validado com screenshot em viewport desktop (1400px) e
    mobile (390px, tamanho de iPhone).
  - Para acessar do celular: `uv run uvicorn api.main:app --host 0.0.0.0
    --port 8000` na máquina, depois `http://<ip-local-da-máquina>:8000/` no
    celular (mesma rede Wi-Fi). Isso **substitui** o fluxo de dois servidores
    (`uvicorn` + `python -m http.server`) descrito originalmente na seção 6 —
    seção 6 abaixo já foi atualizada para refletir isso.

**Decisões tomadas durante a implementação (que fecham perguntas em aberto
da seção 10 original):**
- Código, comentários e mensagens de commit: **inglês**.
- Artefatos gerados (`data/synthetic_dataset.csv`,
  `models/oil_probability_model.joblib`, `models/metrics.json`): **commitados
  no git** (reprodutibilidade imediata para a banca da OBSAT, sem precisar
  re-treinar antes de demonstrar).
- Testes automatizados: **incluídos** como etapa própria (ver pendências).

**Ainda falta (retomar em casa):**

- [ ] **Etapa 5 — Testes automatizados**: `tests/test_generate_dataset.py` e
  `tests/test_api.py` (pytest) ainda não implementados.
- [ ] **Etapa 6 — README**: `README.md` ainda é só o placeholder padrão do
  `uv init`; falta documentar contexto, como rodar, e resumo científico.
- [ ] Validação interativa completa do dashboard num navegador de verdade
  (cliques em "Analisar ponto"/"Varrer área", não só carregamento da
  página) — só foi validado visualmente via screenshot automatizado, sem
  simulação confiável de clique neste ambiente. Vale um teste manual rápido.
- [ ] Itens da seção 9 original que continuam abertos (relatório da OBSAT
  com citações formais, decisão sobre abordagem híbrida, ajuste de features
  quando o payload real do satélite for definido).

## 1. Contexto do projeto

- Projeto para a **OBSAT** (Olimpíada Brasileira de Satélites), que **será
  avaliado por avaliadores da competição** — por isso, todo o embasamento
  (científico, técnico e de dados) precisa estar documentado e defensável,
  não apenas funcional.
- Objetivo: modelo que recebe leituras de **metano (CH₄)** e **pressão**
  (features prioritárias) mais **posição (latitude/longitude)**, e estima a
  **probabilidade (%) de haver petróleo** naquele ponto.
- **Não há acesso ao payload real** que será usado em produção (sensor do
  satélite/CubeSat ainda não definido/disponível). Por isso o modelo precisa
  ser treinado com dados obtidos de outra forma, mantida cientificamente
  defensável.
- Escopo geográfico: **bacias de Campos e Santos** (litoral Sudeste do
  Brasil, incluindo pré-sal).
- Prazo: **apertado, menos de 1 semana** até a entrega/apresentação.
- Entregável esperado: **modelo + dashboard de visualização**.
- Autor: Lucas, estudante de Engenharia da Computação (UFMA), com background
  em ML aplicado a ciências (ionosfera, saúde, etc.) — prefere comunicação
  direta e honesta, sem otimismo artificial, e deixar claro o que é estimativa
  vs. conhecimento sólido.

## 2. Fundamentação científica (usar isso no relatório da OBSAT)

O conceito do projeto é real e tem nome na geologia do petróleo:
**hydrocarbon microseepage** (microsseepage de hidrocarbonetos).

- Reservatórios de petróleo vazam continuamente pequenas quantidades de
  hidrocarbonetos leves (metano, etano, propano) até a superfície, através
  de falhas e microporos na rocha selante (o selo nunca é perfeito).
- É consenso geológico que **todas as bacias petrolíferas exibem algum tipo
  de vazamento próximo à superfície** (micro ou macrosseepage).
- Estudos de campo mostram que a concentração de metano tende a **decair
  com a distância** de uma falha/trap ativo (relação usada na geração do
  dataset sintético — ver seção 4).
- Zonas de **sobrepressão (overpressure)** no reservatório estão associadas
  a acumulação de hidrocarbonetos — por isso pressão é uma feature
  secundária relevante, não arbitrária.
- Dado estatístico forte para citar no relatório: em estudos de survey de
  microsseepage, **~80% dos poços perfurados em áreas com anomalia positiva
  de microsseepage resultaram em descoberta comercial**, contra **~14% em
  áreas sem anomalia associada** (fonte: literatura de exploração por
  microsseepage, ex. trabalhos de D. Schumacher).
- Fontes de pesquisa relevantes encontradas (buscar e citar formalmente no
  relatório, com autor/ano corretos):
  - "Integrating Hydrocarbon Microseepage Data with Seismic Data Doubles
    Exploration Success" (survey de microsseepage vs. taxa de sucesso).
  - Capítulo "Gas geochemistry surveys for petroleum" (ScienceDirect) —
    base conceitual de surveys geoquímicos de gás para petróleo.
  - Estudos sobre o campo de seepage marinho de Coal Oil Point (Califórnia)
    — dados reais de concentração/composição de bolhas de metano/CO₂ em
    seepage ativo (referência de ordem de grandeza).

**Importante**: esses princípios (decaimento espacial do sinal, associação
com sobrepressão) são reais. Os **valores numéricos exatos** usados no
gerador sintético (amplitude do sinal, escala de decaimento em km) são
**simplificações didáticas**, calibradas para o problema ser aprendível por
um modelo simples — isso deve ficar **explícito e documentado** no relatório
da OBSAT como limitação consciente, não escondido.

## 3. Decisão sobre a origem dos dados (já discutida e fechada)

Não existe dataset público pronto que combine exatamente metano + pressão +
posição + rótulo de presença de petróleo por ponto. Opções avaliadas:

| Opção | Veredito |
|---|---|
| Datasets de produção de poços reais (ex. **Volve field**, Equinor, Mar do Norte) | Real, mas: sem medição de metano atmosférico; só tem exemplos "positivos" (poços já confirmados); dezenas de GB/TB via Azure; Mar do Norte, não Brasil. Inviável no prazo. |
| Datasets de sensoriamento remoto de metano (TROPOMI/Sentinel-5P) | Cobrem emissões atmosféricas em geral (inclusive industriais), não rotulados para petróleo especificamente. |
| **Dataset sintético fisicamente informado** (✅ decisão tomada) | Rápido, controlável, permite gerar casos positivos E negativos (essencial pra classificação), e é defensável se a geração seguir relações reais da literatura (seção 2) e as limitações forem documentadas. |

**Decisão: Opção 1 — dataset sintético fisicamente informado.** Já foi
prototipado e validado nesta conversa (ver seção 4), pronto para ser
implementado localmente.

## 4. Especificação do gerador de dataset sintético (validado)

Isso já foi implementado e testado num sandbox de protótipo. Os parâmetros
abaixo já foram calibrados (uma primeira tentativa deu sinal fraco demais,
foi ajustada). Pode ser usado como referência direta de implementação.

### 4.1 Ancoragem em campos reais

Usar coordenadas aproximadas de campos de petróleo reais nas bacias de
Campos e Santos como "âncoras" de sinal positivo (uso ilustrativo/educacional,
não para exploração real):

```
Tupi/Lula (Santos, pré-sal)   -25.20, -43.00
Búzios (Santos, pré-sal)      -25.30, -43.50
Sapinhoá (Santos, pré-sal)    -25.00, -43.30
Marlim (Campos)               -22.50, -40.30
Roncador (Campos)             -22.05, -40.10
Barracuda (Campos)            -22.35, -40.10
Jubarte (Espírito Santo)      -20.60, -39.80
Albacora (Campos)             -22.20, -40.30
```

### 4.2 Bounding box de amostragem

Latitude: -27.5 a -19.0 · Longitude: -45.5 a -37.5 (litoral SE do Brasil,
cobrindo as bacias de Campos e Santos e margem continental próxima).

### 4.3 Geração das features (por ponto amostrado aleatoriamente na bbox)

1. Calcular `distance_km` = distância haversine até o campo-âncora mais próximo.
2. **Metano** (ppm):
   ```
   background = 1.9 ppm          (nível atmosférico de fundo real)
   anomaly    = 0.35 * exp(-distance_km / 35.0)
   methane_ppm = background + anomaly + ruído_gaussiano(0, 0.03)
   ```
3. **Pressão** (hPa):
   ```
   background = 1013.0 hPa        (pressão padrão ao nível do mar)
   anomaly    = 4.0 * exp(-distance_km / 45.0)
   pressure_hpa = background + anomaly + ruído_gaussiano(0, 0.5)
   ```
4. **Probabilidade "verdadeira" (função geradora, usada só para validação, não
   para treino)**:
   ```
   methane_z  = anomaly_metano / 0.35     (normalizado 0–1)
   pressure_z = anomaly_pressao / 4.0     (normalizado 0–1)

   logit = 7.0*methane_z + 3.0*pressure_z - 5.5 + ruído_gaussiano(0, 0.25)
   true_probability = sigmoid(logit)
   ```
   Os pesos (7.0 metano vs 3.0 pressão) refletem a prioridade pedida:
   metano como sinal principal, pressão como secundário.
5. **Rótulo observável** (o que um cenário real teria disponível — poço
   seco vs. produtor, não uma probabilidade):
   ```
   label = Bernoulli(true_probability)
   ```

### 4.4 Ponto de atenção crítico (já vivido no protótipo)

Na primeira tentativa, o ruído estocástico da função geradora estava alto
demais em relação ao sinal real → o modelo treinado só chegou a **AUC 0.59**
(quase aleatório). Depois de reduzir o ruído do logit (de σ=0.4 para σ=0.25),
aumentar as escalas de decaimento espacial (de 18km/25km para 35km/45km, o
que aumenta a área com sinal aprendível) e reforçar o bias negativo (de -3.0
para -5.5, reduzindo falsos positivos aleatórios longe dos campos), o
resultado melhorou para **AUC 0.87**, com prevalência de positivos caindo de
6.6% para ~2.4% (mais realista — a maior parte do litoral não tem petróleo).

**Se ao implementar localmente o modelo sair com desempenho ruim, o
primeiro lugar a olhar é esse equilíbrio ruído-vs-sinal na função geradora**,
não necessariamente o modelo em si.

### 4.5 Colunas finais do CSV

```
latitude, longitude, methane_ppm, pressure_hpa,
distance_to_nearest_field_km  (guardar para análise, NÃO usar como feature de treino — teria efeito de "vazamento" da resposta),
true_probability                (guardar só para avaliação/calibração, NÃO usar no treino),
label                           (0/1 — este é o alvo real de treino)
```

`N_SAMPLES` sugerido: 12.000 (rápido de treinar, suficiente para os efeitos
aparecerem com clareza). `random_state`/seed fixo (ex. 42) para
reprodutibilidade — importante para o relatório da OBSAT.

## 5. Decisão sobre o modelo

**RandomForestClassifier (scikit-learn)**, não TensorFlow/rede neural, pelos
seguintes motivos (documentar no relatório como decisão justificada, não
"caminho mais fácil"):

- Problema é tabular, com poucas features (4) e relações não-lineares
  simples — árvores capturam isso bem sem overfitting.
- `predict_proba` nativo já dá probabilidades calibradas — exatamente a
  saída pedida pelo projeto.
- Treina em segundos em CPU (relevante: a GPU do laboratório é uma GeForce
  G210, compute capability 1.2, **incompatível com TensorFlow moderno** —
  já confirmado, não vale a pena tentar configurar CUDA nela).
- Dá `feature_importances_` — importante para o relatório mostrar que o
  modelo aprendeu a priorizar metano e pressão (validado no protótipo:
  metano ~37%, pressão ~49%, longitude ~8%, latitude ~6%).
- Mais fácil de explicar/defender numa banca de olimpíada do que uma rede
  neural "caixa-preta".

Configuração validada no protótipo:
```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    class_weight="balanced",   # dataset desbalanceado (poucos positivos, como na realidade)
    random_state=42,
    n_jobs=-1,
)
```

### Métricas de avaliação (recomendadas)

- **ROC-AUC** (resultado do protótipo: 0.87)
- **Erro médio absoluto vs. `true_probability`** (só possível porque o
  dataset é sintético e guardamos a probabilidade real — é uma forma de
  validar calibração que não seria possível com dados reais; deixar isso
  claro no relatório como vantagem extra de usar dado sintético). Resultado
  do protótipo: ~0.11.
- Considerar também `classification_report` (precision/recall) dado o
  desbalanceamento de classes.

## 6. Arquitetura da entrega (modelo + API + dashboard)

**[Implementado — ver seção 0 para o status exato]**. Decisão original:
dashboard em HTML/JS puro consumindo uma API, sem framework de front-end
pesado. Durante a implementação local isso evoluiu para **um único processo
FastAPI servindo tanto a API quanto o dashboard**, para permitir acesso via
celular em campo junto com o satélite (ver seção 0 para o porquê).

- **API** (`api/main.py`): FastAPI servindo o modelo treinado (`joblib`),
  com endpoints:
  - `GET /health` — status da API e se o modelo está carregado.
  - `POST /predict` — recebe `{latitude, longitude, methane_ppm,
    pressure_hpa}`, retorna `{oil_probability_percent, classification}`.
  - `POST /predict/batch` — mesma coisa para lista de pontos (usado para
    varrer uma grade no mapa).
  - CORS liberado (`allow_origins=["*"]`).
  - `app.mount("/", StaticFiles(directory="dashboard", html=True))` —
    serve `dashboard/index.html` na raiz, montado **depois** das rotas
    acima para não sobrepor `/health`, `/predict`, etc.
- **Dashboard** (`dashboard/index.html`): HTML/CSS/JS único, usando
  **Leaflet.js** (via CDN) para o mapa, tile layer escuro (CartoDB dark),
  centrado no litoral SE do Brasil. Funcionalidades:
  - Sliders para simular leitura de metano/pressão.
  - Clique no mapa define lat/lon do ponto de leitura.
  - Botão "Analyze point" → chama `/predict`, plota marcador colorido por
    probabilidade (teal <30%, âmbar 30–70%, vermelho >70%).
  - Botão "Scan visible area" → gera grade de pontos dentro do viewport
    atual e chama `/predict/batch`, útil para visualizar o "mapa de calor"
    de probabilidade ao redor de uma região.
  - Marcadores de referência (cinza) nos campos reais usados na geração do
    dataset, para dar contexto visual.
  - `API_BASE = window.location.origin` — funciona a partir de qualquer
    host/IP que sirva a página (localhost, IP da LAN para o celular, ou uma
    futura URL pública), sem hardcode.
  - `@media (max-width: 768px)` — em telas estreitas o layout empilha
    (mapa em cima, painel de controle embaixo, rolável) com alvos de toque
    maiores.
  - Linguagem visual: tema escuro tipo "console de telemetria de satélite"
    (paleta: fundo quase preto `#0a0e13`, âmbar `#ff9f1c`, teal `#3adbc1`),
    tipografia Space Grotesk (display) + IBM Plex Mono (dados) + Inter
    (corpo) — condizente com o contexto OBSAT/payload de satélite.

**Como rodar** (um único comando, serve API + dashboard juntos):
```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- No próprio computador: `http://127.0.0.1:8000/`.
- No celular (mesma rede Wi-Fi): descobrir o IP local da máquina
  (`ip a` / `hostname -I`) e acessar `http://<esse-ip>:8000/`.

Comportamento validado com a API real rodando: 99.61% de probabilidade perto
do campo Tupi/Lula (sinal elevado), 8.79% num ponto de fundo sem anomalia —
mesma ordem de grandeza do protótipo original (99.6% / 6.98%).

## 7. Ambiente já configurado na máquina do laboratório

- Linux Mint, `uv` instalado e funcionando, VS Code instalado, git com SSH
  configurado.
- GPU: GeForce G210 (GT218) — **não usar para TensorFlow/CUDA**, é
  compute capability 1.2, abaixo do mínimo suportado (3.5+). Seguir com
  CPU (RandomForest não precisa de GPU de qualquer forma).
- Dependências já usadas no protótipo (adicionar via `uv add`):
  ```
  numpy pandas scikit-learn joblib fastapi "uvicorn[standard]"
  ```
- Repositório: nome sugerido/definido `atmospheric-oil-detection`
  (ou `oil-areas-prediction`, já usado no protótipo local do Lucas).

## 8. Estrutura de projeto (implementada)

```
oil-areas-prediction/
├── data/
│   └── synthetic_dataset.csv       # gerado por src/generate_dataset.py — commitado
├── src/
│   ├── generate_dataset.py         # gerador fisicamente informado (seção 4)
│   └── train_model.py              # treino + avaliação (seção 5)
├── models/
│   ├── oil_probability_model.joblib  # commitado
│   └── metrics.json                  # commitado
├── api/
│   └── main.py                     # FastAPI + serve o dashboard (seção 6)
├── dashboard/
│   └── index.html                  # dashboard HTML/JS/Leaflet (seção 6)
├── tests/                          # pytest — ainda vazio, ver seção 0
├── pyproject.toml
└── .gitignore
```
Isto já é a estrutura real do projeto, não mais só uma referência de
protótipo — ver seção 0 para o que está implementado em cada pasta.

## 9. O que ainda falta decidir/fazer (em aberto)

Ver seção 0 para a lista atualizada e detalhada do que falta (testes,
README, validação interativa do dashboard). Itens que **não** são sobre
código/implementação, ainda em aberto:

- [ ] Escrever a seção de metodologia do relatório da OBSAT citando
  formalmente as fontes da seção 2 (autor, ano, DOI/link).
- [ ] Decidir se vale a pena evoluir para a abordagem híbrida (dado real de
  poços tipo Volve + metano sintético) se sobrar tempo — discutido como
  "upgrade" possível, não obrigatório.
- [ ] Decidir se o payload final do satélite (quando definido) vai exigir
  ajuste nas features ou apenas re-treino com o mesmo pipeline.

## 10. Tom e prioridades do Lucas (para o Claude Code manter consistência)

- Prefere avaliação honesta e direta, sem otimismo artificial — inclusive
  sobre limitações do dado sintético.
- Projeto precisa ser **defensável cientificamente** perante avaliadores,
  não só funcional.
- **Idioma confirmado**: código, comentários e mensagens de commit em
  **inglês** (decisão fechada durante a implementação — não é mais uma
  pergunta em aberto). Comunicação em conversa continua em português.
- Fluxo de trabalho: implementação **incremental**, uma etapa por vez, com
  sugestão de mensagem de commit ao final de cada uma — **o Lucas é quem
  commita**, não o Claude Code.
