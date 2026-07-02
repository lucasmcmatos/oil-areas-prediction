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