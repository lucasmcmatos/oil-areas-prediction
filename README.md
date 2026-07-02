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
| CH₄ (ppm) | 1.9 ppm               | 0.35 ppm              | 35 km                |
| Pressão (hPa) | 1013.0 hPa        | 4.0 hPa               | 45 km                |

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

1. **Âncoras de campo real**: coordenadas de 8 campos de petróleo nas bacias de Campos
   e Santos (Tupi/Lula, Búzios, Sapinhoá, Marlim, Roncador, Barracuda, Jubarte,
   Albacora) usadas como centros de sinal positivo.
2. **Amostragem aleatória**: 12.000 pontos uniformes na bounding box
   lat ∈ [−27.5°, −19.0°], lon ∈ [−45.5°, −37.5°]. Seed fixo (42) para
   reprodutibilidade.
3. **Cálculo de distância**: distância haversine (arco de grande círculo) até o campo
   mais próximo.
4. **Geração de features**: CH₄ e pressão com decaimento exponencial + ruído gaussiano.
5. **Probabilidade geradora**: logit ponderado (CH₄ peso 7, pressão peso 3) com bias
   −5.5 e ruído σ=0.25 → sigmoid → `true_probability` (guardada apenas para avaliação).
6. **Rótulo observável**: `label ~ Bernoulli(true_probability)` — prevalência de
   positivos ≈ 2.4%, refletindo que a maior parte do litoral não tem petróleo.

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

### Resultados Comparativos

Todos os modelos avaliados no mesmo conjunto de teste (2.400 amostras, 2.42% positivos):

| Modelo | ROC-AUC ↑ | Avg Precision ↑ | Brier Score ↓ | Calibration MAE ↓ |
|---|---|---|---|---|
| Random Forest | **0.8740** | 0.5102 | 0.0408 | 0.1085 |
| Gradient Boosting | 0.8516 | **0.5556** | **0.0259** | **0.0357** |
| Logistic Regression | 0.8580 | 0.5564 | 0.0739 | 0.1934 |
| **Ensemble (soft voting)** | 0.8554 | 0.5492 | 0.0386 | 0.1115 |

**Observações:**

- **Random Forest** tem a maior capacidade discriminativa (ROC-AUC 0.874), consistente
  com o resultado do protótipo original (AUC 0.87).
- **Gradient Boosting** se destaca na **calibração**: Brier Score de 0.026 e
  Calibration MAE de 0.036 — muito superior aos demais. Para uma aplicação que exibe
  probabilidades percentuais ao usuário, este é o indicador mais crítico.
- **Regressão Logística** é surpreendentemente competitiva em AP (0.556) e ROC-AUC
  (0.858), indicando que o sinal do dataset tem componente aproximadamente linearizável.
  Sua calibração ruim (MAE 0.193) reflete a limitação do modelo linear ao capturar a
  forma sigmoidal do decaimento.
- **Ensemble soft voting** equilibra os modelos mas não domina nenhuma métrica
  individualmente — comportamento esperado quando os modelos têm erros parcialmente
  correlacionados. Sua vantagem principal é a **robustez**: ao combinar três vieses
  indutivos distintos, reduz a probabilidade de falhas sistemáticas em regiões do
  espaço de features não bem cobertas por nenhum modelo individual.

O **Gradient Boosting foi adotado como modelo final da API** por apresentar a melhor
calibração — critério prioritário para uma aplicação que exibe probabilidades percentuais
ao usuário. O ensemble foi descartado como opção final por não superar o GB individualmente
em nenhuma das métricas de calibração.

**Nota sobre interpretação do Average Precision:** Para datasets desbalanceados, o
baseline de um classificador aleatório em AP é igual à prevalência de positivos
(~2.4%, não 50%). O AP de 0.556 do GB representa, portanto, **23× o desempenho
aleatório** — o modelo consegue concentrar verdadeiros positivos no topo do ranking
de probabilidade de forma muito superior ao acaso.

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