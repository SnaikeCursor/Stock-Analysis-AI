# Erkenntnisse: Swiss SPI Extra Aktienanalyse

**Projekt:** Vorhersage der Aktienperformance von Schweizer Small/Mid-Caps (SPI Extra)
**Zeitraum Analyse:** Q1 2024 (Training), 2025 (Forward-Test)
**Stand:** März 2026

---

## Ansatz 1: Drei-Klassen-Klassifikation (Baseline)

### Setup

- **Universum:** SPI Extra, nach Liquiditätsfilter (~136 Aktien)
- **Modell:** Random Forest Classifier (gewählt gegenüber XGBoost wegen besserem Hold-out F1)
- **Klassifikation:** Quartalsrendite Q1 2024, Perzentil-basiert — Top 25% = Winners, Bottom 25% = Losers, Mitte = Steady
- **Features:** 28 (18 technische + 10 fundamentale), Stichtag 31.12.2023 (kein Lookahead)
- **Validation:** Stratified 75/25 Hold-out Split + 5-Fold CV

### Ergebnisse

**In-Sample (Hold-out Q1 2024):**

| Metrik | CV (5-Fold) | Hold-out (25%) |
|---|---|---|
| Accuracy | 56.6% ± 9.8% | 44.1% |
| F1 macro | 51.7% ± 12.3% | 39.2% |

**Out-of-Sample Forward-Test (2025):**

| Metrik | Wert |
|---|---|
| Accuracy | 48.5% (vs. 33.3% Zufall) |
| F1 macro | 42.1% |
| Korrekt klassifiziert | 66 / 136 |

**Strategie-Renditen (2025, exkl. Kosten):**

| Strategie | Return | Sharpe | Max Drawdown |
|---|---|---|---|
| Long Winners | +28.5% | 1.40 | -19.8% |
| Long/Short | +10.9% | 1.55 | -3.6% |
| Benchmark (Equal Weight) | +9.7% | 0.83 | -14.1% |

**Trefferquoten:**
- Winners: 42.1% (8/19 korrekt)
- Steady: 56.1% (46/82 korrekt)
- Losers: 34.3% (12/35 korrekt)

### Was funktioniert hat

- **Kein Overfitting:** OOS-Performance war leicht besser als In-Sample — das Modell hat nicht überangepasst.
- **Benchmark geschlagen:** Long Winners (+28.5%) schlägt Equal Weight (+9.7%) deutlich.
- **Long/Short risikoadjustiert stark:** Sharpe 1.55 bei nur -3.6% Max Drawdown.
- **Gruppensortierung stimmt:** Vorhergesagte Winners hatten den höchsten mittleren Return (+33.5%), Losers den niedrigsten (+5.3%).

### Was nicht funktioniert hat

- **Accuracy nur 48.5%:** Besser als Zufall (33.3%), aber nicht zuverlässig.
- **Winner-Erkennung schwach:** Nur 8 von 19 vorhergesagten Winners waren tatsächlich Winners.
- **Median vs. Mean:** Median-Return der vorhergesagten Winners nur +4.5% — der hohe Durchschnitt (+33.5%) wird von wenigen Extremgewinnern getrieben. Fragile Strategie.
- **Modell-Bias:** Das Modell klassifiziert 82 von 136 Aktien als Steady — es ist "zu vorsichtig" und vermeidet extreme Vorhersagen.

### Wichtigste Features (SHAP-Analyse)

| Feature | Stärkstes Signal für |
|---|---|
| debt_equity | Losers (hohe Verschuldung → schlechte Performance) |
| profit_margin | Losers (niedrige Marge → Underperformance) |
| atr_14_pct | Steady/Winners (Volatilität als Trennsignal) |
| hvol_60d | Steady (historische Volatilität) |
| revenue_growth | Steady (Umsatzwachstum) |
| roe | Winners (hohe Eigenkapitalrendite) |
| volume_ratio_20_60 | Losers (ungewöhnliches Volumen) |

---

## Ansatz 2: Multi-Period Enhanced (mehrere Quartale)

### Setup

- **Idee:** Statt nur Q1 2024 werden 7 Quartale (Q2-2023 bis Q4-2024) gestapelt → ~950 Beobachtungen statt 135
- **Erweiterungen:** Rank-Features, Sektor-Dummies, korrelierte Features entfernt, LightGBM, Ensemble, Binary Classification, Walk-Forward CV
- **Hypothese:** Mehr Trainingsdaten → besseres Modell

### Ergebnisse

Das Enhanced-Modell hat **keine Verbesserung** gegenüber dem Baseline gebracht.

### Analyse der Ursachen

**1. Pseudo-Replikation statt echte neue Daten**
7 Quartale × 136 Aktien = 950 Zeilen, aber es sind dieselben 136 Aktien. Fundamentale Features (debt_equity, profit_margin, market_cap) ändern sich zwischen Quartalen kaum. Das Modell sieht 7× fast die gleiche Aktie — kein echter Informationsgewinn.

**2. Regime-Instabilität**
Was einen Winner in Q2-2023 ausmacht (z.B. Post-COVID-Recovery), gilt nicht in Q4-2024 (z.B. Zinsumfeld). Stacking über verschiedene Regime erzwingt ein Durchschnittsmodell, das für kein Regime optimal ist.

**3. Label-Noise durch relative Klassifikation**
Die Labels sind Perzentil-basiert *innerhalb* jedes Quartals. In einem starken Marktquartal ist ein "Loser" bei -2%, in einem schwachen bei -30%. Gleiche Features → völlig unterschiedliche Labels → Modell verwirrt.

**4. Datenqualitätsproblem Q2-2023**
Der früheste Trainingszeitraum (Q2-2023, Feature-Cutoff 31.03.2023) hatte nur ~196 Handelstage History. Das SMA(200)-Feature war für alle Aktien NaN. Nicht kritisch (1 von 7 Perioden), aber ein Hinweis auf Datengrenzen.

**5. Fundamentale Grenze: 136 Aktien sind zu wenig**
Für ein 3-Klassen-Problem mit 28 Features braucht man typischerweise Hunderte bis Tausende *unabhängige* Beobachtungen. Mehr Zeitperioden der gleichen 136 Aktien lösen dieses Problem nicht.

---

## Übergreifende Erkenntnisse

### Technische Lektionen

1. **`_MIN_BARS` Pauschalfilter war destruktiv:** Der ursprüngliche Code hatte `_MIN_BARS = 200`, was *alle* technischen Features auf NaN setzte, weil die OHLCV-Daten erst ab Juli 2023 begannen (nur ~131 Tage vor Cutoff). Fix: `YF_START` auf 2022-07-01 erweitert und `_MIN_BARS` auf 63 gesenkt.

2. **Jupyter Kernel-Cache:** Nach Änderungen an `config.py` oder `src/*.py` muss der Kernel in *jedem* Notebook neu gestartet werden — Python-Module werden beim Import gecached.

3. **NumPy-Versionskonflikt:** `shap` importiert `cv2` (OpenCV), das gegen NumPy 1.x kompiliert war. Fix: `pip install --upgrade opencv-python` (4.8 → 4.13).

4. **Parquet-Cache:** Beim Ändern von `YF_START` müssen die gecachten Parquet-Dateien gelöscht werden, sonst werden die alten Daten wiederverwendet.

### Methodische Lektionen

1. **Mehr Daten ≠ besseres Modell**, wenn die "neuen" Daten keine neue Information enthalten (gleiche Aktien, ähnliche Features über Quartale).

2. **Cross-Sectional vs. Time-Series:** Das Modell lernt Querschnitts-Muster (welche Aktie innerhalb eines Quartals besser performt), nicht Zeitreihen-Muster. Das ist prinzipiell korrekt, aber 136 Aktien sind zu wenig Querschnitt.

3. **Drei Klassen sind problematisch:** Die "Steady"-Klasse (mittlere 50%) dominiert und das Modell lernt, im Zweifel "Steady" vorherzusagen. Zwei Klassen (Winner vs. Rest) oder kontinuierliche Regression wären schärfer.

4. **Perzentil-Labels sind nicht stabil:** Ein "Winner" in Q1 2024 hat andere absolute Returns als ein "Winner" in Q3-2023. Das verwirrt Modelle, die über mehrere Perioden trainiert werden.

5. **Signal ist real, aber schwach:** Die Baseline schlägt den Benchmark konsistent, was auf ein echtes (wenn auch schwaches) Signal hindeutet. Die Herausforderung ist nicht "gibt es ein Signal?", sondern "wie macht man es robust genug für den Einsatz?".

---

## Nächste Schritte: Regression statt Klassifikation

Basierend auf den Erkenntnissen planen wir einen Paradigmenwechsel:

- **Kontinuierliche Rendite-Vorhersage** statt 3-Klassen-Klassifikation
- **1-Monats-Horizont** für mehr Trainingsbeobachtungen
- **Ranking-basierte Evaluation** (Information Coefficient / Spearman) statt Accuracy
- **Portfolio aus Ranking** statt aus Klassen-Labels
- Siehe `REGRESSION_PLAN.md` für Details

---

## Projektstruktur (Referenz)

```
Stock Analysis AI/
├── config.py                        # Zentrale Konfiguration
├── data/                            # OHLCV + Fundamentals Cache
├── notebooks/
│   ├── 01_data_collection.ipynb     # Daten laden & Qualität
│   ├── 02_classification.ipynb      # Q1 2024 Returns & Gruppen
│   ├── 03_feature_engineering.ipynb  # Feature-Matrix
│   ├── 04_feature_analysis.ipynb    # Diskriminanz-Analyse, SHAP
│   ├── 05_model_training.ipynb      # RF/XGB Training (Baseline + Enhanced)
│   └── 06_forward_test.ipynb        # 2025 Forward-Test
├── src/
│   ├── universe.py                  # SPI Extra Ticker
│   ├── data_loader.py               # yfinance + Cache
│   ├── classifier.py                # Performance-Klassifikation
│   ├── features.py                  # Feature Engineering
│   ├── analysis.py                  # Statistische Analyse
│   ├── model.py                     # ML-Modell (Klassifikation)
│   └── backtest.py                  # Forward-Test
└── ERGEBNIS_ZUSAMMENFASSUNG.md      # Ergebnis-Übersicht Ansatz 1
```
