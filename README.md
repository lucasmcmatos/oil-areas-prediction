# oil-areas-prediction

> **Nota:** Este README é um documento de trabalho em construção. Está sendo usado
> como relatório incremental de decisões e metodologia durante o desenvolvimento.
> Será reorganizado formalmente antes da entrega à OBSAT.

---

## Visão Geral do Projeto

Sistema de predição de áreas com possível ocorrência de petróleo, desenvolvido para a
**OBSAT (Olimpíada Brasileira de Satélites)**. O modelo recebe leituras de metano (CH₄),
pressão atmosférica e posição geográfica (latitude/longitude) — simulando o payload de
um satélite/CubeSat — e estima a probabilidade (%) de haver petróleo naquele ponto.

Escopo geográfico: **bacias de Campos e Santos**, litoral Sudeste do Brasil (pré-sal).

---

## Fundamentação Científica da Geração do Dataset

Como não existe dataset público combinando exatamente as features do projeto com rótulos
de presença de petróleo por ponto, optou-se por um **dataset sintético fisicamente
informado** — gerado a partir de relações estabelecidas na literatura científica de
exploração geoquímica. Esta seção documenta esse embasamento.

### 1. Microsseepage de Hidrocarbonetos (*Hydrocarbon Microseepage*)

O conceito central do projeto é o **microsseepage de hidrocarbonetos**: reservatórios de
petróleo vazam continuamente quantidades traço de hidrocarbonetos leves (principalmente
metano, CH₄) até a superfície, através de falhas, fraturas e microporos na rocha selante.

Este fenômeno é consenso na geologia do petróleo e é a base de levantamentos
geoquímicos de superfície usados em exploração desde os anos 1930. Schumacher (1996)
documenta extensamente os mecanismos físicos e as alterações geoquímicas resultantes
em solos e sedimentos sobre reservatórios ativos. Jones & Drozd (1983) demonstraram
que anomalias de concentração de gases leves na superfície têm correlação estatística
com a presença de acumulações comerciais em profundidade, estabelecendo o uso de
geoquímica de superfície como ferramenta exploratória.

### 2. Decaimento Espacial Exponencial do Sinal

O sinal de metano não é uniforme: ele decai com a distância da fonte (falha ou trap
ativo). O modelo adotado neste projeto — decaimento exponencial com a distância — segue
o modelo físico proposto por Saunders, Burson & Thompson (1999), que descrevem a
distribuição espacial das anomalias de microsseepage como função exponencial da
distância à fonte:

```
anomalia(d) = A × exp(−d / λ)
```

Onde `A` é a amplitude máxima e `λ` é a escala de decaimento espacial (em km).

**Parâmetros utilizados no gerador (calibrados para o problema ser aprendível):**

| Feature   | Background real       | Amplitude da anomalia | Escala de decaimento |
|-----------|-----------------------|-----------------------|----------------------|
| CH₄ (ppm) | 1.9 ppm               | 0.35 ppm              | 80 km                |
| Pressão (hPa) | 1013.0 hPa        | 4.0 hPa               | 100 km               |

**Nota sobre recalibração das escalas de decaimento:** O modelo original (Campos/Santos)
usava escalas de 35/45 km, calibradas para uma bounding box pequena e densa. Ao expandir
para toda a costa brasileira (~10× de área), a distância média de um ponto amostrado até
a âncora mais próxima passou de ~80 km para ~250–350 km. Com escala de 35 km, o sinal
exponencial é virtualmente zero a > 150 km, colapsando a prevalência de positivos para
< 1% e tornando o problema inaprendível (AUC ~0.76). As escalas foram aumentadas para
80/100 km, representando **microsseepage difuso de bacia** (basin-scale leakage) em vez
de seep pontual — distinção documentada em Saunders et al. (1999). O bias do logit foi
ajustado de −5.5 para −4.5 para restaurar prevalência de ~2%. Ambas as mudanças são
simplificações didáticas declaradas explicitamente.

O nível de fundo de **1.9 ppm de CH₄** corresponde ao nível atmosférico global médio
documentado pelo NOAA Global Monitoring Laboratory (Lan et al., 2024), tornando o sinal
simulado fisicamente plausível em magnitude.

### 3. Metano como Feature Principal

CH₄ é o hidrocarboneto leve mais abundante em microsseepage e o mais detectável
remotamente. Estudos de campo em Coal Oil Point (Santa Bárbara, Califórnia) —
um dos maiores seeps marinhos do mundo — quantificaram emissões contínuas da ordem
de 10⁴–10⁵ m³/dia de gás com ~60–80% de CH₄ (Hornafius, Quigley & Luyendyk, 1999),
confirmando a plausibilidade física de anomalias mensuráveis de CH₄ sobre reservatórios
ativos. No gerador, CH₄ recebe peso 7.0 no logit gerador (vs. 3.0 da pressão),
refletindo seu papel como sinal primário.

### 4. Pressão como Feature Secundária

Zonas de **sobrepressão** (*overpressure*) no reservatório estão frequentemente
associadas à presença e acumulação de hidrocarbonetos, pois o próprio gás em solução
contribui para a pressão de poros. Osborne & Swarbrick (1997) revisam os mecanismos
geradores de sobrepressão em bacias sedimentares e documentam sua associação com
acumulações de petróleo e gás — justificando a inclusão de pressão como feature
secundária no modelo.

### 5. Validação Estatística do Conceito

Schumacher (2000) reporta que, em estudos de levantamento por microsseepage, **~80%
dos poços perfurados em áreas com anomalia positiva confirmada resultaram em descoberta
comercial**, contra **~14% em áreas sem anomalia associada**. Este dado motiva o uso
de anomalias de microsseepage como sinal preditivo, mesmo em dataset sintético.

### 6. Limitações Explícitas (transparência para avaliadores)

Os princípios físicos acima são reais e fundamentados na literatura. As **simplificações
didáticas** adotadas — e que devem ser declaradas explicitamente no relatório — são:

- As escalas exatas de decaimento (35 km para CH₄, 45 km para pressão) são
  simplificações calibradas para que o problema seja aprendível por um modelo simples;
  na realidade, essas escalas variam com a geologia local, profundidade do reservatório
  e permeabilidade da rocha selante.
- A amplitude das anomalias é menor do que as detectadas em seeps ativos superficiais
  como Coal Oil Point, mas dentro da faixa plausível para microsseepage difuso.
- O dataset é sintético: não há medições reais de satélite. O pipeline (geração →
  treino → API → dashboard) demonstra a viabilidade do método e seria substituído por
  dados reais quando o payload do CubeSat estiver disponível.

---

## Metodologia de Geração do Dataset (`src/generate_dataset.py`)

### Cobertura geográfica

O dataset cobre toda a **margem continental brasileira**, de RS a AP:

- **Bounding box**: lat ∈ [−34.0°, +5.5°], lon ∈ [−50.0°, −28.0°]
- **30.000 amostras** (aumento de 12k para manter densidade com a área expandida)

Esta expansão foi realizada para cobrir bacias do Nordeste, permitindo predições para
qualquer ponto da costa brasileira — incluindo a **Baía de São Luís (MA)**, local de
realização da OBSAT (~−2.53°S, −44.30°W).

### Âncoras de campo/bacia

16 âncoras distribuídas ao longo da costa brasileira, do SE ao NE:

| Âncora | Bacia | Lat | Lon |
|---|---|---|---|
| Tupi/Lula | Santos (pré-sal) | −25.20 | −43.00 |
| Búzios | Santos (pré-sal) | −25.30 | −43.50 |
| Sapinhoá | Santos (pré-sal) | −25.00 | −43.30 |
| Marlim | Campos | −22.50 | −40.30 |
| Roncador | Campos | −22.05 | −40.10 |
| Barracuda | Campos | −22.35 | −40.10 |
| Albacora | Campos | −22.20 | −40.30 |
| Jubarte | Espírito Santo | −20.60 | −39.80 |
| Camarupim | Espírito Santo | −19.80 | −39.50 |
| Camamu | Camamu-Almada (BA) | −13.80 | −38.80 |
| Sergipe-Alagoas | Sergipe-Alagoas (SE/AL) | −10.50 | −36.50 |
| Potiguar Offshore | Potiguar (RN) | −4.00 | −36.80 |
| Potiguar Onshore | Potiguar (RN/CE) | −5.10 | −36.50 |
| Ceará | Ceará (CE) | −3.20 | −38.50 |
| Barreirinhas | Barreirinhas (MA/PI) | −2.50 | −42.50 |

Coordenadas são centróides aproximados de campos produtores ou depocentros de bacia,
usados para fins ilustrativos/educacionais — não para exploração real.

### Pipeline de geração

1. **Amostragem aleatória**: 30.000 pontos uniformes na bounding box. Seed fixo (42).
2. **Cálculo de distância**: distância haversine até a âncora mais próxima.
3. **Geração de features**: CH₄ e pressão com decaimento exponencial + ruído gaussiano.
4. **Probabilidade geradora**: logit ponderado (CH₄ peso 7, pressão peso 3) com bias
   −5.5 e ruído σ=0.25 → sigmoid → `true_probability` (guardada apenas para avaliação).
5. **Rótulo observável**: `label ~ Bernoulli(true_probability)` — prevalência de
   positivos ≈ 2.4%, refletindo que a maior parte do litoral não tem petróleo.

### Comportamento esperado para São Luís (MA)

A Baía de São Luís fica a ~160 km da âncora Barreirinhas. A essa distância, o sinal
de microsseepage é atenuado (decaimento exponencial com λ=35 km para CH₄), portanto
o modelo deve predizer **baixa probabilidade** — o que é cientificamente correto,
já que São Luís não está sobre um campo confirmado, mas próximo a uma bacia em
exploração. Isso demonstra que o sistema discrimina corretamente entre áreas de
anomalia ativa e áreas adjacentes.

---

## Modelagem: Abordagem, Fundamentos e Resultados

### Justificativa da Escolha dos Modelos

O problema é de **classificação binária em dados tabulares** (4 features numéricas,
~2.4% de positivos) com relações não-lineares entre features e rótulo (decaimento
exponencial da distância → metano e pressão → probabilidade via logit). Esta
configuração favorece modelos baseados em árvores, mas inclui-se também um modelo
linear para comparação e explicabilidade. Três modelos com **vieses indutivos
complementares** foram treinados, seguidos de um ensemble de votação suave.

Partição do dataset: **9.600 amostras de treino** / **2.400 de teste** (80/20),
estratificada para preservar a prevalência de positivos (~2.42%) em ambas as
partições. Seed fixo (42) para reprodutibilidade.

---

### Modelo 1 — Random Forest (Floresta Aleatória)

**Fundamento teórico:** Proposto por Breiman (2001), o Random Forest é um método de
ensemble baseado em *bagging* (Bootstrap Aggregating): treina B árvores de decisão
em subamostras bootstrap do treino, com seleção aleatória de um subconjunto de
features a cada divisão. A predição final é a média das probabilidades das B árvores.
Esta dupla fonte de aleatoriedade (amostras e features) reduz a variância sem aumentar
significativamente o viés, tornando o modelo robusto a overfitting em dados tabulares.

**Por que é adequado aqui:** Captura relações não-lineares (como o decaimento
exponencial da distância para metano/pressão) sem assumir forma funcional.
`class_weight="balanced"` compensa o desbalanceamento de classes (97.6% negativos).
O atributo `feature_importances_` permite verificar se o modelo aprendeu a
priorizar CH₄ e pressão — importante para a defesa científica do projeto.

**Configuração:**
```python
RandomForestClassifier(n_estimators=300, max_depth=8,
                       min_samples_leaf=5, class_weight="balanced",
                       random_state=42)
```

**Importâncias de features aprendidas:**

| Feature | Importância |
|---|---|
| pressure_hpa | ~0.49 |
| methane_ppm | ~0.37 |
| longitude | ~0.08 |
| latitude | ~0.06 |

Resultado consistente com o design do dataset: metano e pressão dominam,
posição geográfica tem papel secundário.

---

### Modelo 2 — Gradient Boosting Histogramado (HistGradientBoosting)

**Fundamento teórico:** Gradient Boosting foi formalizado por Friedman (2001) como
um algoritmo de ensemble *sequencial*: cada árvore nova é treinada para corrigir os
resíduos da iteração anterior, minimizando uma função de perda diferenciável via
gradiente descendente no espaço funcional. A variante histogramada (*Histogram-based
Gradient Boosting*), inspirada no LightGBM (Ke et al., 2017), agrupa valores contínuos
em bins discretos, reduzindo o custo computacional de O(n×f) para O(bins×f) por divisão,
sem perda significativa de desempenho.

**Por que é adequado aqui:** O boosting sequencial tende a produzir modelos com melhor
**calibração** de probabilidade do que o bagging — cada iteração refina a estimativa de
probabilidade sobre os exemplos mais difíceis. Isso é especialmente relevante aqui, onde
a saída do modelo é usada diretamente como percentual de probabilidade pelo dashboard.
`class_weight="balanced"` é suportado nativamente nesta variante (a partir do scikit-learn 1.2).

**Configuração:**
```python
HistGradientBoostingClassifier(max_iter=300, max_depth=6,
                               learning_rate=0.05, class_weight="balanced",
                               random_state=42)
```

---

### Modelo 3 — Regressão Logística (com padronização)

**Fundamento teórico:** Proposta por Cox (1958) para modelagem de variáveis binárias,
a Regressão Logística modela a probabilidade posterior P(y=1|x) através de uma combinação
linear das features passada pela função sigmoide. É o modelo linear de referência para
classificação binária, com estimação por máxima verossimilhança. Por ser linear, **requer
padronização das features** (média zero, desvio-padrão 1) para que os coeficientes sejam
comparáveis e o otimizador convirja adequadamente — implementado via `Pipeline` com
`StandardScaler`.

**Por que é adequado aqui:** Serve como *baseline* linear para quantificar o ganho dos
modelos não-lineares. Se a Regressão Logística atingir desempenho próximo ao RF/GB,
isso sugere que as relações capturadas pelo dataset são aproximadamente linearizáveis
(ou que o sinal é suficientemente forte). `class_weight="balanced"` compensa o
desbalanceamento; `C=1.0` é a regularização L2 padrão (inverso do parâmetro λ).

**Configuração:**
```python
Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                               max_iter=1000, solver="lbfgs",
                               random_state=42))
])
```

---

### Ensemble — Soft Voting (Votação Suave)

**Fundamento teórico:** Métodos de ensemble por combinação de classificadores são
fundamentados na teoria de redução de erro de generalização de Dietterich (2000):
quando os erros dos modelos individuais são suficientemente *incorrelacionados*, sua
combinação reduz a variância do erro em relação a qualquer modelo individual. A
**votação suave** (*soft voting*) combina as probabilidades estimadas por cada
modelo (em vez dos rótulos discretos), calculando a média das distribuições de
probabilidade preditiva:

```
P_ensemble(y=1|x) = (P_RF(y=1|x) + P_GB(y=1|x) + P_LR(y=1|x)) / 3
```

Esta abordagem é preferível à votação dura (*hard voting*) quando os modelos
produzem probabilidades bem calibradas, pois preserva a granularidade da informação
de confiança de cada modelo (Kuncheva, 2004).

**Implementação:**
```python
VotingClassifier(
    estimators=[("rf", build_rf()), ("gb", build_gb()), ("lr", build_lr())],
    voting="soft"
)
```

O ensemble é treinado do zero sobre o mesmo conjunto de treino, re-ajustando todos
os estimadores internamente — os modelos individuais avaliados na seção anterior são
instâncias separadas, usadas exclusivamente para comparação de desempenho isolado.

---

### Métricas de Avaliação

Quatro métricas foram utilizadas, cada uma capturando um aspecto diferente de qualidade:

| Métrica | O que mede | Por que é relevante aqui |
|---|---|---|
| **ROC-AUC** | Capacidade discriminativa em todos os limiares | Robusto ao desbalanceamento; padrão na literatura de classificação |
| **Average Precision (AP)** | Área sob a curva Precisão-Recall | Mais informativo que ROC-AUC com classes muito desbalanceadas (~2.4% positivos) |
| **Brier Score** | Erro quadrático médio entre prob. predita e rótulo real | Mede qualidade de calibração; 0 = perfeito |
| **Calibration MAE** | Erro absoluto médio entre prob. predita e `true_probability` | Exclusivo de dataset sintético — mede calibração contra a distribuição geradora real |

---

### Pipeline de Treinamento e Resultados

O pipeline de modelagem seguiu quatro etapas: validação cruzada baseline,
busca de hiperparâmetros, avaliação no conjunto de teste e ensemble.

#### Etapa 1 — Validação Cruzada Baseline (5-fold estratificado, hiperparâmetros padrão)

A validação cruzada estratificada em 5 folds (Stratified K-Fold) garante que
cada fold preserve a proporção de positivos (~3.3%), fornecendo estimativas
de desempenho mais robustas e com quantificação de variância (±std).
Dataset: 30.000 amostras, 24.000 treino / 6.000 teste, prevalência 3.28%.

| Modelo | ROC-AUC (média ± std) | AP (média ± std) |
|---|---|---|
| Random Forest | 0.7740 ± 0.0336 | 0.4491 ± 0.0576 |
| Gradient Boosting | 0.7731 ± 0.0309 | 0.4413 ± 0.0582 |
| Logistic Regression | **0.7862 ± 0.0382** | **0.4627 ± 0.0627** |

A Regressão Logística lidera novamente em CV, confirmando componente linearizável
no sinal. O GB apresenta menor desvio-padrão (0.031), indicando maior estabilidade.
A redução de ~0.88 para ~0.78 em relação ao modelo Campos/Santos é esperada: o
problema é genuinamente mais difícil com 10× mais área e sinal mais difuso.

#### Etapa 2 — Busca de Hiperparâmetros (RandomizedSearchCV)

`RandomizedSearchCV` com 30 iterações e 5-fold CV, otimizando ROC-AUC
(Bergstra & Bengio, 2012).

| Modelo | Melhor CV ROC-AUC | Melhores hiperparâmetros |
|---|---|---|
| Random Forest | 0.7899 | max_depth=4, min_samples_leaf=5, max_features='sqrt', n_estimators=200 |
| Gradient Boosting | 0.7897 | max_depth=3, learning_rate=0.2, min_samples_leaf=10, max_iter=500 |
| Logistic Regression | 0.7863 | C=0.1 |

Os três modelos convergem para valores de ROC-AUC próximos após tuning (~0.789),
evidenciando que o ganho marginal do tuning foi moderado — o gargalo está na
dificuldade intrínseca do problema (sinal difuso em grande escala geográfica),
não nos hiperparâmetros.

#### Etapa 3 — Avaliação no Conjunto de Teste (modelos tunados)

| Modelo | ROC-AUC ↑ | Avg Precision ↑ | Brier Score ↓ | Calibration MAE ↓ |
|---|---|---|---|---|
| Random Forest (tunado) | 0.7832 | 0.4166 | 0.1204 | 0.3013 |
| Gradient Boosting (tunado) | 0.7923 | 0.4110 | **0.1140** | **0.2781** |
| Logistic Regression (tunada) | 0.7833 | 0.4152 | 0.1408 | 0.3191 |
| **Ensemble soft voting** | **0.7930** | **0.4211** | 0.1215 | 0.2995 |

**Observações:**

- O **ensemble** superou todos os modelos individuais em ROC-AUC (0.7930) e AP
  (0.4211) — diferente do resultado anterior, onde os modelos eram mais correlacionados.
  Com o dataset expandido e mais diverso, os três modelos erram em regiões diferentes
  do espaço geográfico, tornando a combinação mais eficaz (Dietterich, 2000).
- O **Gradient Boosting** continua com a melhor calibração (Brier 0.114, CalMAE 0.278),
  mantendo o trade-off discriminação vs. calibração observado anteriormente.
- A redução de ROC-AUC (0.87 → 0.79) é explicável e defensável: o modelo agora cobre
  toda a costa brasileira com sinal mais difuso (escalas 80/100 km vs. 35/45 km) — um
  problema genuinamente mais difícil, não uma falha de modelagem.

#### Etapa 4 — Limiar de Decisão Ótimo (curva Precisão-Recall)

Para datasets desbalanceados, o limiar padrão de 0.5 é subótimo. A curva
Precisão-Recall foi usada para encontrar o limiar que maximiza F1 para o GB:

| Limiar | F1 no conjunto de teste |
|---|---|
| 0.5 (padrão) | — |
| **0.8986 (ótimo)** | **0.4971** |

O limiar elevado (0.899) reflete que, com ~3.3% de positivos, o modelo precisa de
alta confiança para predizer positivo e manter precisão aceitável. O dashboard usa
a probabilidade contínua, tornando o limiar relevante apenas para as classificações
textuais (low / moderate / high).

**Nota sobre interpretação do Average Precision:** Para datasets desbalanceados, o
baseline de um classificador aleatório em AP é igual à prevalência de positivos
(~3.3%, não 50%). O AP de 0.4211 do ensemble representa, portanto, **~13× o desempenho
aleatório** — o modelo concentra verdadeiros positivos no topo do ranking de forma
muito superior ao acaso.

**Modelo adotado na API:** Gradient Boosting tunado (`gradient_boosting_model.joblib`),
por apresentar a melhor calibração (Brier 0.114, CalMAE 0.278) — critério prioritário
para uma aplicação que exibe probabilidades percentuais ao usuário.

---

## Referências

- Hornafius, J. S., Quigley, D. C., & Luyendyk, B. P. (1999). The world's most
  spectacular marine hydrocarbon seeps (Coal Oil Point, Santa Barbara Channel,
  California): Quantification of emissions. *Journal of Geophysical Research: Oceans*,
  104(C9), 20703–20711. https://doi.org/10.1029/1999JC900148

- Jones, V. T., & Drozd, R. J. (1983). Predictions of oil or gas potential by
  near-surface geochemistry. *AAPG Bulletin*, 67(6), 932–952.
  https://doi.org/10.1306/03B5B471-16D1-11D7-8645000102C1865D

- Lan, X., Tans, P., & Thoning, K. W. (2024). *Trends in globally-averaged CH4
  determined from NOAA Global Monitoring Laboratory measurements* (Version 2024-08).
  NOAA Global Monitoring Laboratory. https://doi.org/10.15138/P8XG-AA10

- Osborne, M. J., & Swarbrick, R. E. (1997). Mechanisms for generating overpressure
  in sedimentary basins: A reevaluation. *AAPG Bulletin*, 81(6), 1023–1041.
  https://doi.org/10.1306/522B49C3-1727-11D7-8645000102C1865D

- Saunders, D. F., Burson, K. R., & Thompson, C. K. (1999). Model for hydrocarbon
  microseepage and related near-surface alterations. *AAPG Bulletin*, 83(1), 170–185.
  https://doi.org/10.1306/E4FD2DAB-1732-11D7-8645000102C1865D

- Schumacher, D. (1996). Hydrocarbon-induced alteration of soils and sediments. In
  D. Schumacher & M. A. Abrams (Eds.), *Hydrocarbon Migration and Its Near-Surface
  Expression* (AAPG Memoir 66, pp. 71–89). American Association of Petroleum
  Geologists.

- Schumacher, D. (2000). Integrating geochemical surveys with 3D seismic data to
  identify new drilling targets: Examples from the Gulf of Mexico. In *AAPG Annual
  Convention Proceedings*. American Association of Petroleum Geologists.

**Aprendizado de Máquina**

- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization.
  *Journal of Machine Learning Research*, 13(10), 281–305.

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
  https://doi.org/10.1023/A:1010933404324

- Cox, D. R. (1958). The regression analysis of binary sequences. *Journal of the
  Royal Statistical Society: Series B (Methodological)*, 20
  (2), 215–242.
  https://doi.org/10.1111/j.2517-6161.1958.tb00292.x

- Dietterich, T. G. (2000). Ensemble methods in machine learning. In J. Kittler &
  F. Roli (Eds.), *Multiple Classifier Systems* (Lecture Notes in Computer Science,
  Vol. 1857, pp. 1–15). Springer. https://doi.org/10.1007/3-540-45014-9_1

- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine.
  *Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y.
  (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in
  Neural Information Processing Systems*, 30, 3146–3154.

- Kuncheva, L. I. (2004). *Combining Pattern Classifiers: Methods and Algorithms*.
  Wiley-Interscience. https://doi.org/10.1002/0471660264