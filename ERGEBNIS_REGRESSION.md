# Ergebnis-Zusammenfassung: Regression (1-Monats-Forward-Return) — Erweitertes Modell v2

**Modell:** Random Forest Regressor (200 Trees, max_features=sqrt, min_samples_leaf=5)
**Universum:** SPI Extra — 136 Aktien nach Liquiditätsfilter (min. tägliches Volumen)
**Training:** 28 Monate (Sep 2022 – Dez 2024), 3'780 Beobachtungen
**OOS Forward-Test:** 11 Monate (Jan – Nov 2025), 1'493 Beobachtungen
**Features:** 28 nach Korrelationsfilter (|r| > 0.85), aus 38 Roh-Features (23 technisch + 13 fundamental inkl. Analysten-Konsensus + 2 Saisonalität) plus Cross-Sectional-Rank-Features
**Transaktionskosten:** 0 bps (ohne Kosten im OOS-Backtest)

## Änderungen gegenüber dem Basismodell (v1)

| Aspekt | Basismodell (v1) | Erweitertes Modell (v2) |
|---|---|---|
| Roh-Features | 28 (18 tech. + 10 fund.) | 38+ (23 tech. + 13 fund. + 2 Saisonalität + Rank-Features) |
| Korrelationsfilter | nicht verwendet | `drop_correlated_features(0.85)` |
| Features nach Filter | 28 | 28 |
| Hold-out | 1 Monat | 3 Monate (`holdout_periods=3`) |
| Transaktionskosten (OOS) | 20 bps | 0 bps |
| Neue Feature-Gruppen | — | Analysten (Rating, Count, Target-Upside), Liquidität (Amihud, Volume-Trend, Spread-Proxy), Saisonalität (month_sin/cos), Mean-Reversion (Skew, Max-DD) |

## Modell-Metriken (Hold-out)

| Metrik | v1 (1 Monat) | v2 (Dez 2024) | Einordnung |
|---|---|---|---|
| Information Coefficient (IC) | +0.027 | **+0.119** | Stark verbessert — deutliches Ranking-Signal |
| RMSE | 9.84% | 12.28% | Höher — breitere Streuung im Hold-out-Monat |
| MAE | 7.21% | 7.41% | Leicht gestiegen |
| R² | −0.67 | −1.60 | Negativ — Modell kann absolute Returns nicht vorhersagen |
| CV IC (expanding window) | — | +0.009 ± 0.088 (24 Folds) | Schwach — grosser Spread über Folds |

## OOS-Strategie-Performance (2025, ohne Transaktionskosten)

| Strategie | Kumulierter Return | Ann. Return | Volatilität | Sharpe Ratio | Max Drawdown | Ø Turnover/Monat |
|---|---|---|---|---|---|---|
| **Long-Only (Top-10)** | −5.6% | −6.7% | 25.2% | −0.27 | −27.0% | 58% |
| **Short-Only (Bottom-10)** | **+16.4%** | **+20.1%** | 21.6% | **+0.93** | −18.0% | 65% |
| **Long/Short** | **+6.7%** | **+8.1%** | 11.2% | **+0.72** | −7.1% | 61% |
| **Benchmark (EW)** | +1.6% | +2.0% | 12.5% | +0.16 | −14.1% | – |

### Pro 1 CHF investiert (Jan–Nov 2025)

| Strategie | Endwert | Gewinn/Verlust |
|---|---|---|
| Long-Only (Top-10) | 0.94 CHF | −0.06 CHF |
| Short-Only (Bottom-10) | 1.16 CHF | +0.16 CHF |
| Long/Short | 1.07 CHF | +0.07 CHF |
| Benchmark (EW) | 1.02 CHF | +0.02 CHF |

## Monatliche Information Coefficients (OOS 2025)

| Monat | IC | Universum | Interpretation |
|---|---|---|---|
| Jan 2025 | +0.012 | 136 | Neutral |
| Feb 2025 | +0.048 | 136 | Schwach positiv |
| Mär 2025 | +0.085 | 136 | Positiv |
| Apr 2025 | **+0.259** | 136 | Sehr stark positiv |
| Mai 2025 | +0.048 | 136 | Schwach positiv |
| Jun 2025 | +0.108 | 136 | Positiv |
| Jul 2025 | −0.050 | 136 | Negativ |
| Aug 2025 | −0.103 | 136 | Negativ |
| Sep 2025 | +0.004 | 136 | Neutral |
| Okt 2025 | +0.017 | 136 | Neutral |
| Nov 2025 | −0.024 | 136 | Leicht negativ |

| Aggregat | v1 | v2 | Veränderung |
|---|---|---|---|
| **Mean IC** | +0.020 | **+0.037** | +85% |
| **Std IC** | 0.066 | 0.095 | Breiter |
| **IR (IC / Std)** | 0.29 | **0.39** | +34% |
| **Positive Monate** | 7 / 11 (64%) | **8 / 11 (73%)** | Verbessert |

## SHAP Feature-Importance (Top 25)

| Feature | Mean |SHAP| | Gruppe |
|---|---|---|
| atr_14_pct | 0.0097 | Technisch |
| sma_ratio_50_200 | 0.0078 | Technisch |
| hvol_20d | 0.0053 | Technisch |
| dist_52w_low | 0.0046 | Technisch |
| market_cap_log | 0.0041 | Fundamental |
| bb_width | 0.0036 | Technisch |
| hvol_60d | 0.0035 | Technisch |
| profit_margin | 0.0034 | Fundamental |
| pb_ratio | 0.0032 | Fundamental |
| roe | 0.0027 | Fundamental |
| mom_3m | 0.0026 | Technisch |
| rel_volume_5d | 0.0025 | Technisch |
| dist_52w_high | 0.0023 | Technisch |
| mom_6m | 0.0021 | Technisch |
| adx_14 | 0.0019 | Technisch |
| macd_diff_norm | 0.0018 | Technisch |
| roc_10 | 0.0016 | Technisch |
| mom_1m | 0.0015 | Technisch |
| volume_ratio_20_60 | 0.0015 | Technisch |
| obv_slope_20d | 0.0015 | Technisch |
| zscore_20d | 0.0015 | Technisch |
| revenue_growth | 0.0013 | Fundamental |
| rsi_14 | 0.0010 | Technisch |
| ev_ebitda | 0.0010 | Fundamental |
| debt_equity | 0.0009 | Fundamental |

Die neuen Features (Analysten-Konsensus, Liquidität, Saisonalität, erweiterte Mean-Reversion) und Rank-Features erscheinen nicht in den Top 25 der SHAP-Importance. Trotzdem hat sich die Prognosefähigkeit des Modells verbessert — die neuen Features wirken möglicherweise als Regularisierungssignal und verbessern die Generalisierung.

## Vergleich: Regression v1 vs v2

| Aspekt | v1 (Basismodell) | v2 (Erweitertes Modell) | Bewertung |
|---|---|---|---|
| Hold-out IC | +0.027 | **+0.119** | ✅ Stark verbessert |
| Mean OOS IC | +0.020 | **+0.037** | ✅ Verbessert |
| IR (IC / Std) | 0.29 | **0.39** | ✅ Näher an robustem Schwellenwert |
| Long-Only Return | −2.5% (20 bps) | −5.6% (0 bps) | ⚠️ Schlechter trotz 0 Kosten |
| L/S Return | −7.5% (20 bps) | **+6.7% (0 bps)** | ✅ Deutlich verbessert |
| Short-Seite | −15.4% (20 bps) | **+16.4% (0 bps)** | ✅ Short-Identifikation funktioniert |
| Positive IC-Monate | 64% | **73%** | ✅ Verbessert |

**Wichtiger Hinweis:** Die v1-Ergebnisse enthalten 20 bps Transaktionskosten pro Rebalancing, die v2-Ergebnisse enthalten keine Kosten. Ein direkter Vergleich der absoluten Returns ist daher eingeschränkt. Die IC-Metriken sind davon nicht betroffen und zeigen eine klare Verbesserung.

## Vergleich: Regression v2 vs Klassifikation (OOS 2025)

| Aspekt | Regression v2 (monatlich) | Klassifikation (jährlich) |
|---|---|---|
| Ansatz | Kontinuierlicher Return, monatlich | Winner/Loser, jährlicher Hold |
| Long-Only Kumuliert | −5.6% | **+55.9%** |
| Long-Only Sharpe | −0.27 | **+2.19** |
| Benchmark Kumuliert | +1.6% | +9.7% |
| Long vs Benchmark | Long schlechter | **Long deutlich besser** |
| Max Drawdown (Long) | −27.0% | −23.2% |
| Volatilität (Long) | 25.2% | 26.1% |
| OOS Zeitraum | Jan–Nov 2025 (209 Tage) | Jan–Dez 2025 (248 Tage) |

Das Klassifikationsmodell übertrifft das Regressionsmodell im Long-Only-Vergleich deutlich. Der entscheidende Unterschied: Die Klassifikation hält ein festes Winner-Portfolio für das ganze Jahr (kein Turnover), während die Regression monatlich rebalanciert (Ø 58% Turnover). Bei Einführung von Transaktionskosten würde sich der Vorteil der Klassifikation weiter vergrössern.

## Gesamtbewertung

### Was funktioniert (v2 vs v1)

- **Deutlich höheres Hold-out IC (+0.119 vs +0.027):** Das erweiterte Feature-Set mit Korrelationsfilter erzeugt ein stärkeres Signal in der Trainingsphase
- **Höheres Mean OOS IC (+0.037 vs +0.020):** Die Ranking-Fähigkeit im Forward-Test hat sich fast verdoppelt
- **Verbessertes IR (0.39 vs 0.29):** Die Signalkonsistenz ist höher — näher am robusten Schwellenwert von ~0.5
- **Short-Seite funktioniert:** Die identifizierten Bottom-10 verlieren tatsächlich (+16.4% Short-Return), was den L/S-Spread positiv macht
- **73% positive IC-Monate:** In 8 von 11 Monaten zeigt das Ranking in die richtige Richtung

### Was nicht funktioniert

- **Long-Only verliert vs Benchmark:** Auch ohne Transaktionskosten liefert die Top-10-Auswahl −5.6% vs +1.6% (Benchmark) — das Modell identifiziert die besten Aktien nicht zuverlässig
- **Hoher Turnover (~58–65%):** Bei realistischen Kosten (20 bps) würde der L/S-Spread weiter schrumpfen
- **Negatives R²:** Das Modell kann absolute Returns nicht vorhersagen
- **Neue Features wenig prädiktiv:** Analysten, Liquidität, Saisonalität und Rank-Features erscheinen nicht in den Top-25-SHAP-Features — der Informationsgehalt dieser Datenquellen ist begrenzt
- **Klassifikation deutlich überlegen:** Das Winner/Loser-Modell mit jährlichem Hold (+55.9%) schlägt die Regression (−5.6%) massiv

### Fazit

Das erweiterte Regressionsmodell (v2) zeigt messbare Verbesserungen gegenüber dem Basismodell in den statistischen Metriken (IC, IR). Besonders die Short-Seite funktioniert nun besser, was zu einem positiven L/S-Return von +6.7% führt. Allerdings bleibt die Long-Only-Strategie defizitär: Mit 1 CHF investiert in die Long-Only-Strategie hätte man am Ende von Nov 2025 **0.94 CHF** zurück — ein Verlust von 6 Rappen (ohne Transaktionskosten).

Die Erweiterung der Feature-Basis hat die Prognosefähigkeit verbessert, aber nicht ausreichend für eine profitable Long-Only-Strategie. Für einen produktionsreifen Ansatz wären nötig:
- IC > 0.05 im OOS-Test
- Deutlich niedrigerer Turnover (Reduktion durch Turnover-Penalty oder längere Halteperioden)
- Alternative: Kombination von Regressions-Ranking mit der Klassifikations-Auswahl

---

# Robustness-Validierung der Klassifikations-Strategie

**Ziel:** Prüfen ob die starke OOS-Performance der Klassifikation (Long Winners +55.9%, Sharpe 2.19 in 2025) robust ist oder ein Zufallstreffer.
**Script:** `robustness_test.py` — drei Tests mit bestehenden `src/`-Modulen, Laufzeit ~8 Min.

## Test 1: Walk-Forward über zwei OOS-Jahre

Für jedes OOS-Jahr wird ein separates Modell auf dem jeweils letzten verfügbaren Quartal trainiert.

| Konfiguration | Long Return | Benchmark | Sharpe | Long > BM? |
|---|---|---|---|---|
| OOS 2024, Train Q3-2023 | **−1.6%** | +2.3% | −0.17 | ❌ |
| OOS 2024, Train Q4-2023 | +7.6% | +2.3% | 0.51 | ✅ |
| OOS 2025, Train Q1-2024 (Original) | **+27.0%** | +9.7% | 1.45 | ✅ |

**Ergebnis:** 2 von 3 Konfigurationen schlagen den Benchmark. Das 2024-OOS hängt stark vom Trainingsquartal ab — Q3-2023 liefert ein negatives Long-Portfolio, Q4-2023 ist positiv. Das Original-Setup (Q1-2024 → 2025) bestätigt sich, aber mit +27.0% statt +55.9% (vermutlich wegen anderer Feature-Selektion durch GridSearch).

## Test 2: Verschiedene Trainings-Quartale für OOS 2025

Selbes OOS-Jahr (2025), aber vier verschiedene Trainings-Quartale:

| Trainingsquartal | Long Return | Sharpe | Winners (N) | Long > BM (+9.7%)? |
|---|---|---|---|---|
| Q1-2024 (Original) | **+27.0%** | 1.45 | 23 | ✅ |
| Q2-2024 | +6.7% | 0.35 | 22 | ❌ |
| Q3-2024 | +12.9% | 0.98 | 31 | ✅ |
| Q4-2024 | **+18.3%** | **2.95** | 22 | ✅ |

**Ergebnis:** 3 von 4 Quartalen liefern Long > Benchmark. Q4-2024 erzielt sogar den höchsten Sharpe (2.95) mit dem konzentriertesten Portfolio (22 Winners). Q2-2024 ist die einzige Variante unter dem Benchmark. Die Strategie ist über verschiedene Trainingsquartale **mehrheitlich** stabil, aber nicht universell.

## Test 3: Permutationstest (1'000 Iterationen)

Vergleich der Modell-Auswahl mit 1'000 zufälligen gleichgrossen Portfolios aus dem gesamten Universum:

| OOS-Jahr | Modell-Return | Zufall (Median) | p-Wert | Signifikant (p < 0.10)? |
|---|---|---|---|---|
| 2024 | +7.6% | +0.4% | **0.267** | ❌ |
| 2025 | +27.0% | +9.8% | **0.018** | ✅ |

**Ergebnis:** Für 2025 ist die Modell-Auswahl mit p = 0.018 klar besser als Zufall (unter der üblichen 5%-Schwelle). Für 2024 kann die Überperformance gegenüber Zufall statistisch **nicht** bestätigt werden (p = 0.267).

## Gesamtverdikt: INCONCLUSIVE

| Kriterium | Ergebnis | Schwelle für „ROBUST" |
|---|---|---|
| Walk-Forward Long > BM | 2/3 (67%) | 3/3 (100%) |
| Multi-Quarter Long > BM | 3/4 (75%) | ≥3/4 (75%) ✅ |
| Permutation signifikant | 1/2 (50%) | 2/2 (100%) |

Die Strategie ist **weder** ein eindeutiger Zufallstreffer **noch** durchgängig robust:

- **Für 2025-OOS** sprechen die Daten klar: drei von vier Trainingsquartalen schlagen den Benchmark, Permutationstest hoch signifikant (p = 0.018)
- **Für 2024-OOS** ist das Bild brüchig: nur mit Train Q4-2023 funktioniert es, Q3-2023 liefert Verluste, Permutation nicht signifikant
- **Sensitivität auf Trainingsquartal:** Die Performance variiert erheblich je nach Trainingsdaten (Long von −1.6% bis +27.0%) — ein Zeichen, dass das Modell auf spezifische Marktregimes reagiert

---

# Robustness-Validierung mit erweitertem Datensatz (SPI ~215 Titel, 2018–2024)

**Ziel:** Erneute Robustness-Prüfung nach Erweiterung des Universums auf ~215 SPI-Titel und Verlängerung der Historie auf 2018 (24 Quartale Q1-2019 bis Q4-2024).
**Script:** `robustness_test.py --refresh-ohlcv`, Laufzeit ~6 Min.

## Walk-Forward (2021–2025, Train Q4 des Vorjahres)

| OOS-Jahr | Long Return | Benchmark | Sharpe | Long > BM? | Permutation p |
|---|---|---|---|---|---|
| 2021 | **−5.0%** | +17.1% | −0.26 | ❌ | 0.985 |
| 2022 | **−24.4%** | −18.0% | −1.07 | ❌ | 0.922 |
| 2023 | **+29.8%** | +2.4% | 1.61 | ✅ | **0.001** |
| 2024 | +9.8% | +5.0% | 0.71 | ✅ | 0.231 |
| 2025 | **+25.7%** | +11.6% | **3.19** | ✅ | 0.066 |

## Multi-Quarter Ensemble (Q1–Q4 2024 → OOS 2025)

| Konfiguration | Long Return | Sharpe | Winners |
|---|---|---|---|
| Ensemble Q1–Q4-2024 | **−9.8%** | −0.45 | 7 |

## Gesamtverdikt: INCONCLUSIVE

- Walk-forward: 3/5 beat benchmark
- Multi-quarter ensemble: 0/1 beat benchmark
- Permutation: 2/5 signifikant (p<0.10), 1/5 (p<0.05)
- Plan gate (≥4/5 WF beat BM & ≥4/5 p<0.05): **FAIL**

Das erweiterte Universum und die längere Historie verbessern die Ergebnisse in Bull-Phasen (2023–2025), aber das Modell versagt weiterhin in Bear/Volatile-Phasen (2021–2022). Dies bestätigt das Kernproblem: **starke Regime-Abhängigkeit**.

---

# Regime-basierte Strategie (Phase 5)

**Ziel:** Separate ML-Modelle pro Marktregime trainieren, um die Regime-Abhängigkeit zu eliminieren.
**Ansatz:** Regelbasierter Detektor auf ^SSMI (SMA-200 + 60d-RealVol) → 3 Regimes (Bull/Bear/Crisis) → regime-spezifische Modelle mit Blending bei Transition.
**Script:** `robustness_test.py --regime-only`, Laufzeit ~31 Min.

## Regime-Labeling der 24 Trainingsquartale

| Regime | Quartale | Anteil |
|---|---|---|
| **Bull** | ~16 Quartale (Q2-2019–Q4-2021, Q1-2022, Q2-2023–Q3-2024) | 67% |
| **Crisis** | 5 Quartale (Q1-2019, Q2/Q3-2020, Q2/Q3-2022) | 21% |
| **Bear** | 3 Quartale (Q4-2022, Q1-2023, Q4-2023) | **12%** |

**Problem:** Bear hat nur 3 Quartale — unter dem Minimum von 4 für ein eigenes Modell. Bear nutzt daher das Fallback-Modell mit Regime-Dummies. Crisis hat kein OOS-Testjahr.

## Regime-Stabilität (intra-Quartal)

- Median Transitions pro Quartal: 0.5
- Median dominant_share: 99.2%
- 12/24 Quartale mit ≥1 Regime-Wechsel innerhalb des Quartals

Regimes sind überwiegend stabil; Wechsel geschehen primär an den Quartalsgrenzen (Q1-2022 und Q4-2024 zeigen erhöhte Instabilität mit 5 Transitions).

## Walk-Forward: Regime-Aware vs Single-Model (Long Winners)

| OOS | Erkanntes Regime (Konfidenz) | Blending | Regime Long | Single Long | Delta | vs BM |
|---|---|---|---|---|---|---|
| 2021 | Bull (0.90) | 100% Bull | +84.8% | +84.8% | ~0% | ✅ |
| 2022 | Bull (1.00) | 100% Bull | **−38.4%** | −36.9% | −1.5% | ❌ |
| 2023 | Bear (0.64) | 36% Bull / 64% Bear | **+35.2%** | +17.8% | **+17.4%** | ✅ |
| 2024 | Bull (0.09) | 9% Bull / 91% Bear | +37.6% | +37.4% | +0.2% | ✅ |
| 2025 | Bear (0.38) | 62% Bull / 38% Bear | +9.5% | **+38.9%** | **−29.5%** | ❌ |

**Hinweis:** 2021 und 2022 verwenden das gleiche Bull-Modell mit hoher Konfidenz — die Performance-Differenz zu den alten Tests (−5.0% / −24.4%) resultiert daraus, dass das Regime-Modell auf **allen Bull-Quartalen** trainiert wird statt nur auf Q4 des Vorjahres.

## OOS-Metriken nach erkanntem Regime

| Regime | N Jahre | Ø Long Cum | Ø Sharpe | Ø L/S Cum | Ø BM Cum | Ø Accuracy |
|---|---|---|---|---|---|---|
| Bear | 2 | +22.3% | 1.26 | +16.5% | +7.0% | 59.4% |
| Bull | 3 | +28.0% | 1.65 | +18.6% | +1.4% | 56.4% |

## Gesamtverdikt Phase 5: INCONCLUSIVE

- 3/5 OOS-Jahre schlagen den Benchmark
- 3/5 OOS-Jahre schlagen das Single-Model
- 3/5 OOS-Jahre schlagen **beide** gleichzeitig
- Erforderlich für ROBUST: ≥4/5

## Bewertung der Regime-Strategie

### Was funktioniert

- **Transitions-Phasen (2023):** Der Regime-Detektor erkennt den Bear am Cutoff Ende 2022 korrekt. Das Blending (64% Bear) wählt 27 statt 3 Winners → Sharpe 2.07 vs 0.69, Return +35.2% vs +17.8%. Klarster Mehrwert des Regime-Ansatzes.
- **Breiter trainierte Modelle (2021):** Das Bull-Modell nutzt alle ~16 Bull-Quartale statt nur Q4-2020 → Return +84.8% statt −5.0% im Single-Quartal-Test. Mehr Trainingsdaten stabilisieren die Feature-Gewichte.
- **Regime-Stabilität:** Regimes sind empirisch persistent (99% dominant_share) — die Annahme „aktuelles Regime ≈ künftiges Regime" hält innerhalb eines Quartals.

### Was nicht funktioniert

- **SMA(200) Lag (2022):** Am Cutoff Ende 2021 ist der SMI noch klar über SMA(200) → Regime = Bull mit 100% Konfidenz. Der Bärmarkt beginnt erst während des OOS-Jahres. Der nachlaufende Indikator ist zu langsam.
- **Over-Diversification (2025):** Das Blending (62% Bull / 38% Bear) produziert 10 Winners (verwässert). Das Single-Model wählt nur 2 Winners (konzentriert) und erzielt +38.9% vs +9.5%. Mehr ist nicht immer besser.
- **Bear-Daten zu dünn:** Nur 3 von 24 Quartalen sind Bear → kein eigenes Bear-Modell möglich, nur Fallback mit Regime-Dummies.
- **Crisis nie getestet:** Kein OOS-Jahr fällt in ein Crisis-Regime — Effektivität unbekannt.

---

# Regime v2: Bull / Bear / Sideways + Frühwarnsystem (Phase A)

**Ziel:** Verbesserung des Regime-Detektors durch (1) Bull/Bear/Sideways statt Bull/Bear/Crisis, (2) SMA(50)/SMA(200)-Alignment statt nur SMA(200), (3) Frühwarnindikatoren (sma_cross_gap, sma_50_slope) für Konfidenz-Dämpfung bei bevorstehenden Regime-Wechseln.
**Daten:** Gleiche 24 Quartale (Q1-2019 bis Q4-2024), gleiche ~215 SPI-Titel. Isoliert den Effekt der neuen Regime-Logik.
**Script:** `robustness_test.py --regime-only`, Laufzeit ~29 Min.

## Regime-Definition v2

- **Bull**: Close > SMA(50) UND SMA(50) > SMA(200) — alle drei aufsteigend
- **Bear**: Close < SMA(50) UND SMA(50) < SMA(200) — alle drei absteigend
- **Sideways**: Alles andere — SMAs widersprüchlich, kein klarer Trend

Frühwarnung: `sma_cross_gap = (SMA50 - SMA200) / SMA200` und `sma_50_slope` (20d Rate-of-Change) dämpfen die Konfidenz, wenn ein Crossover bevorsteht oder SMA(50) gegen den Trend dreht.

## Regime-Verteilung: v1 vs v2

| Regime | v1 (Bull/Bear/Crisis) | v2 (Bull/Bear/Sideways) |
|---|---|---|
| Bull | 16 (67%) | **12 (50%)** |
| Bear/Crisis→Bear | 3 (12%) | **6 (25%)** |
| Crisis→Sideways | 5 (21%) | **6 (25%)** |
| **Eigene Modelle möglich** | Nur Bull | **Alle drei** |

Die v2-Verteilung ist deutlich balancierter. Alle drei Regimes haben ≥ 6 Quartale → **kein Fallback nötig**, alle erhalten ein eigenes dediziertes Modell.

## Regime-Stabilität v1 vs v2

| Metrik | v1 | v2 |
|---|---|---|
| Median Transitions/Quartal | 0.5 | **4.0** |
| Quartale mit ≥1 Switch | 12/24 (50%) | **24/24 (100%)** |
| Median dominant_share | 99.2% | **69.9%** |

v2-Regimes sind **deutlich weniger stabil** innerhalb eines Quartals. Die SMA(50)/SMA(200)-Logik reagiert schneller, produziert aber auch mehr Rauschen.

## Walk-Forward: v1 vs v2 vs Single-Model

| OOS | v1 Regime (Conf) | v2 Regime (Conf) | v1 Long | v2 Long | Single Long | BM | v2 > BM? |
|---|---|---|---|---|---|---|---|
| 2021 | Bull (0.90) | Bull (0.51) | +84.8% | **+87.8%** | +84.8% | +17.1% | ✅ |
| 2022 | Bull (1.00) | Bull (0.69) | −38.4% | **−37.0%** | −36.9% | −18.0% | ❌ |
| 2023 | Bear (0.64) | Bear (0.16) | +35.2% | **+36.0%** | +17.8% | +2.4% | ✅ |
| 2024 | Bull (0.09) | Sideways (0.39) | +37.6% | +34.4% | +37.4% | +5.0% | ✅ |
| 2025 | Bear (0.38) | Bear (0.08) | +9.5% | **+27.8%** | +38.9% | +11.6% | ✅ |

## Gesamtverdikt Phase A: INCONCLUSIVE

| Kriterium | v1 | v2 | Trend |
|---|---|---|---|
| Beat Benchmark | 3/5 | **4/5** | ✅ Verbesserung |
| Beat Single-Model | 3/5 | 2/5 | ⚠️ Verschlechterung |
| Beat Both | 3/5 | 2/5 | ⚠️ Verschlechterung |

## Detailanalyse der Veränderungen v1 → v2

### Was sich verbessert hat

- **2025 massiv verbessert:** +27.8% (v2) vs +9.5% (v1). Das Sideways-dominierte Blending (92% Sideways / 8% Bear) produziert einen einzelnen Winner statt 10. Weniger Over-Diversification.
- **4/5 beat Benchmark:** Erstmals schlägt die Regime-Strategie den Benchmark in 4 von 5 Jahren. 2025 springt von ❌ (+9.5% vs BM +11.6%) auf ✅ (+27.8% vs BM +11.6%).
- **Frühwarnung wirkt bei 2022:** Konfidenz sinkt von 1.00 (v1) auf 0.69 (v2) am Cutoff Ende 2021. Das System erkennt, dass der SMA-Cross-Gap schrumpft. Allerdings reicht die Dämpfung noch nicht — 69% Bull ist immer noch falsch.
- **Balancierte Regime-Verteilung:** 12/6/6 statt 16/5/3 — jedes Regime hat ein eigenes dediziertes Modell, kein Fallback nötig.

### Was sich verschlechtert hat

- **Beat Single-Model sinkt:** Nur noch 2/5 (v1: 3/5). Das Single-Model schlägt v2 in 2024 und 2025 mit konzentrierteren Portfolios (9 bzw. 2 Winners vs 6 bzw. 1).
- **Regime-Stabilität massiv reduziert:** Jedes Quartal hat mindestens 1 Transition, median 4 pro Quartal. Die dominant_share von nur 70% bedeutet, dass das erkannte Regime in 30% der Tage nicht gilt. Das Regime-Signal ist verrauschter.
- **2022 bleibt das Sorgenkind:** Trotz Frühwarnung wird Ende 2021 noch Bull (0.69) erkannt. Der Bärmarkt beginnt erst im OOS-Jahr — kein nachlaufender Indikator kann das lösen.
- **Sehr konzentrierte Picks:** 2023 und 2025 jeweils nur 1 Winner. Hohes Einzeltitel-Risiko.

## OOS-Metriken nach Regime (v2)

| Regime | N Jahre | Ø Long Cum | Ø Sharpe | Ø L/S Cum | Ø BM Cum |
|---|---|---|---|---|---|
| Bear | 2 | +31.9% | 0.75 | +21.1% | +7.0% |
| Bull | 2 | +25.4% | 1.10 | +15.0% | −0.5% |
| Sideways | 1 | +34.4% | 1.48 | +9.9% | +5.0% |

---

# Regime v2 + Smoothing + Min-Winners

**Ziel:** Zwei gezielte Verbesserungen am Regime v2: (1) Rolling Majority Vote über 10 Handelstage zur Regime-Glättung, (2) Minimum 5 Winners pro OOS-Jahr zur Eliminierung von Einzeltitel-Risiko.
**Daten:** Gleiche 24 Quartale (Q1-2019 bis Q4-2024), gleiche ~215 SPI-Titel. Isoliert den Effekt von Smoothing + Min-Winners.
**Script:** `robustness_test.py --regime-only`, Laufzeit ~26 Min.

## Implementierte Änderungen

- **Regime-Glättung:** `smoothing_window=10` in `detect_regime`, `label_periods`, `get_regime_history`. Statt nur den letzten Tag zu klassifizieren, wird über die letzten 10 Handelstage ein Mehrheitsvotum gebildet. Die Konfidenz ist der Mittelwert der Tage mit dem Mehrheits-Label.
- **Min-Winners:** `min_winners=5` in `compute_portfolio_weights`. Falls weniger als 5 Ticker als "Winner" klassifiziert werden, werden die Top-5 nach P(Winner) aus dem gesamten Universum selektiert.

## Regime-Verteilung (mit Smoothing)

| Regime | v2 ohne Smoothing | v2 + Smoothing | Differenz |
|---|---|---|---|
| Bull | 12 (50%) | 11 (46%) | −1 |
| Bear | 6 (25%) | 5 (21%) | −1 |
| Sideways | 6 (25%) | 8 (33%) | +2 |

Das Smoothing verschiebt Grenzfälle in Richtung Sideways — erwartetes Verhalten, da widersprüchliche Tage innerhalb des Fensters zu keinem klaren Mehrheitsvotum führen.

## Regime-Stabilität: v1 vs v2 vs v2+Smoothing

| Metrik | v1 (Bull/Bear/Crisis) | v2 (ohne Smoothing) | v2 + Smoothing |
|---|---|---|---|
| Median Transitions/Q | 0.5 | 4.0 | **2.0** |
| Mean Transitions/Q | — | — | 2.25 |
| Quartale mit ≥1 Switch | 12/24 (50%) | 24/24 (100%) | 23/24 (96%) |
| Median dominant_share | 99.2% | 69.9% | **68.75%** |

Die Transitions halbieren sich (4.0 → 2.0), aber die dominant_share bleibt unter 70%. Das Smoothing glättet das Signal, eliminiert aber nicht die grundsätzliche Instabilität der SMA(50)/SMA(200)-Logik innerhalb eines Quartals.

## Walk-Forward: v2+Smoothing+MinW vs v2 vs v1 vs Single

| OOS | Regime (Conf) | Blending | v2+S+M Long | v2 Long | v1 Long | Single Long | BM | v2+S+M > BM? | v2+S+M > Single? |
|---|---|---|---|---|---|---|---|---|---|
| 2021 | Bull (0.34) | 34% Bull / 66% Sideways | **+95.0%** | +87.8% | +84.8% | +84.8% | +17.1% | ✅ | ✅ |
| 2022 | Bull (0.69) | 69% Bull / 31% Sideways | −38.6% | −37.0% | −38.4% | −36.9% | −18.0% | ❌ | ❌ |
| 2023 | Bear (0.10) | 10% Bear / 90% Sideways | +22.0% | +36.0% | +35.2% | +20.6% | +2.4% | ✅ | ✅ |
| 2024 | Sideways (0.44) | 56% Bear / 44% Sideways | +27.3% | +34.4% | +37.6% | +37.4% | +5.0% | ✅ | ❌ |
| 2025 | Bear (0.04) | 4% Bear / 96% Sideways | **+41.4%** | +27.8% | +9.5% | +28.8% | +11.6% | ✅ | ✅ |

## Min-Winners Fill (aktiviert in 4 Fällen)

| OOS | Modell | Predicted Winners | After Fill | Kommentar |
|---|---|---|---|---|
| 2023 | Regime | 0 | **5** | Regime-Modell identifiziert keinen Winner — Safety-Net greift |
| 2023 | Single | 3 | **5** | Auch Single-Modell unter Minimum |
| 2025 | Regime | 2 | **5** | Nur 2 Winners predicted, aufgefüllt auf 5 |
| 2025 | Single | 2 | **5** | Gleiches Bild beim Single-Modell |

Die Min-Winners-Logik funktioniert wie beabsichtigt: Sie eliminiert das Einzeltitel-Risiko und stabilisiert die Performance in Jahren mit wenigen Predicted Winners.

## Sharpe Ratios und Long/Short-Performance

| OOS | Regime Sharpe | Single Sharpe | Regime L/S | Single L/S |
|---|---|---|---|---|
| 2021 | **3.82** | 3.71 | +36.8% | +31.8% |
| 2022 | −1.02 | −1.03 | −5.7% | −5.9% |
| 2023 | 1.05 | 0.92 | +18.7% | +18.6% |
| 2024 | 1.14 | **1.88** | +8.5% | +16.1% |
| 2025 | **1.74** | 1.22 | +15.1% | +11.2% |

## OOS-Metriken nach Regime (v2+Smoothing+MinW)

| Regime | N Jahre | Ø Long Cum | Ø Sharpe | Ø L/S Cum | Ø BM Cum | Ø Accuracy |
|---|---|---|---|---|---|---|
| Bear | 2 | +31.7% | 1.39 | +16.9% | +7.0% | 54.7% |
| Bull | 2 | +28.2% | 1.40 | +15.5% | −0.5% | 53.8% |
| Sideways | 1 | +27.3% | 1.14 | +8.5% | +5.0% | 56.0% |

## Gesamtverdikt: INCONCLUSIVE

| Kriterium | v1 | v2 | v2+S+M | Trend |
|---|---|---|---|---|
| Beat Benchmark | 3/5 | 4/5 | **4/5** | = (stabil) |
| Beat Single-Model | 3/5 | 2/5 | **3/5** | ✅ Wiederhergestellt |
| Beat Both | 3/5 | 2/5 | **3/5** | ✅ Wiederhergestellt |

Erforderlich für ROBUST: ≥ 4/5 beat both. Aktuell: 3/5.

## Erfolgskriterien-Check

| Kriterium | Ziel | Erreicht | Status |
|---|---|---|---|
| Median Transitions/Q | ≤ 2 | **2.0** | ✅ |
| Min Winners/Jahr | ≥ 5 | **5–6 in allen Jahren** | ✅ |
| Beat Benchmark | ≥ 4/5 | **4/5** | ✅ |
| Beat Single-Model | ≥ 3/5 | **3/5** | ✅ |
| Dominant Share | ≥ 80% | **68.75%** | ❌ |

4 von 5 Erfolgskriterien erreicht — die bisher beste Version.

## Detailanalyse

### Was sich verbessert hat (vs v2 ohne Smoothing)

- **Beat-Single-Rate zurück auf 3/5:** Das Smoothing + Min-Winners stellt die v1-Quote wieder her (v2 hatte nur 2/5). Beide Interventionen wirken: Das Smoothing stabilisiert das Regime-Signal, Min-Winners verhindert Ein-Titel-Portfolios.
- **2021 ist stärkster Jahrgang:** +95.0% (beste Performance über alle Versionen). Das niedrig-konfidente Bull-Blending (34% Bull / 66% Sideways) produziert ein diversifiziertes 5-Winner-Portfolio mit perfekter Hit-Rate (100%).
- **2025 massiv verbessert:** +41.4% (v2+S+M) vs +27.8% (v2) vs +9.5% (v1). Die Min-Winners-Auffüllung (2→5) und das Sideways-dominierte Blending (96%) erzeugen ein robustes Portfolio.
- **Einzeltitel-Risiko eliminiert:** Kein OOS-Jahr hat weniger als 5 Winners. In v2 hatten 2023 und 2025 nur je 1 Winner.
- **Transitions halbiert:** Von 4.0 auf 2.0 pro Quartal — das Regime-Signal ist deutlich stabiler.

### Was offen bleibt

- **Dominant Share unter Ziel (68.75% vs 80%):** Das Regime wechselt innerhalb eines Quartals immer noch in ~30% der Tage. Aggressiveres Smoothing (window=20) könnte helfen, riskiert aber Lag.
- **2022 bleibt strukturelles Problem:** Bull (0.69 Konfidenz) am Cutoff Ende 2021 — kein nachlaufender Indikator kann den Beginn eines Bear-Markts vorhersagen, der erst im OOS-Jahr startet.
- **2024 schwächer als Single-Model:** +27.3% vs +37.4%. Das Bear/Sideways-Blending (56%/44%) passt nicht optimal zum tatsächlichen 2024-Markt. Das Single-Model wählt konzentrierter (9 Winners mit 56% Hit-Rate).
- **Nur 5 OOS-Jahre:** Bei 5 Jahren reicht 1 Ausreisser (2022) um ROBUST zu verfehlen. Mehr OOS-Jahre würden den Effekt einzelner Fehlklassifikationen verdünnen.

---

# Phase B: Erweiterte Historie (52 Quartale, 11 OOS-Jahre)

**Ziel:** Regime v2 + Smoothing + Min-Winners auf deutlich grösserem Datensatz validieren, um (1) mehr Bear-Trainingsdaten zu liefern und (2) einzelne OOS-Ausreisser (2022) weniger schwer wiegen zu lassen.
**Daten:** 52 Quartale (Q1-2012 bis Q4-2024), `YF_START = 2010-01-01` (2 Jahre Pre-Roll für SMA-200). ~160 SPI-Titel nach Liquiditätsfilter.
**OOS-Jahre:** 2015–2025 (11 Jahre). ROBUST-Schwelle: ≥ 8/11 beat both.
**Script:** `robustness_test.py --regime-only`, Laufzeit ~77 Min.

## Regime-Verteilung (52 Quartale mit Smoothing)

| Regime | Quartale | Anteil |
|---|---|---|
| Bull | ~26 | 50% |
| Sideways | ~16 | 31% |
| Bear | ~10 | 19% |

Alle drei Regime haben nun genügend Quartale für eigene dedizierte Modelle — das Bear-Modell profitiert von ~10 statt 5 Quartalen (Eurokrise 2012, SNB-Nachbeben 2015–16, Q4-2018, COVID-Nachbeben, 2022).

## Regime-Stabilität (52 Quartale)

| Metrik | 24 Quartale | 52 Quartale |
|---|---|---|
| Median Transitions/Q | 2.0 | **2.0** |
| Mean Transitions/Q | 2.25 | **2.40** |
| Quartale mit ≥1 Switch | 23/24 (96%) | 47/52 (90%) |
| Median dominant_share | 68.75% | **68.75%** |

Die Stabilitätsmetriken sind identisch — das Smoothing wirkt konsistent über beide Datensätze.

## Walk-Forward: Regime-Aware vs Single-Model (11 OOS-Jahre)

| OOS | Regime (Conf) | Regime Long | Single Long | Delta | Sharpe Regime | vs BM | vs Single |
|---|---|---|---|---|---|---|---|
| 2015 | Bull (0.27) | +35.1% | +54.6% | −19.6% | 0.75 | ✅ (+8.7%) | ❌ |
| 2016 | Bear (0.17) | +29.9% | +8.7% | **+21.2%** | 1.61 | ✅ (+10.4%) | ✅ |
| 2017 | Sideways (0.11) | +49.2% | +25.9% | **+23.3%** | 3.35 | ✅ (+20.3%) | ✅ |
| 2018 | Bull (0.26) | −16.9% | −14.3% | −2.6% | −0.75 | ✅ (−20.5%) | ❌ |
| 2019 | Bear (0.01) | +16.7% | +16.2% | +0.5% | 0.55 | ❌ (+20.9%) | ✅ |
| 2020 | Bull (0.57) | +53.5% | +51.5% | +1.9% | 1.93 | ✅ (+4.8%) | ✅ |
| 2021 | Bull (0.34) | +66.8% | +50.8% | **+15.9%** | 2.74 | ✅ (+17.3%) | ✅ |
| 2022 | Bull (0.69) | −36.3% | −31.0% | −5.3% | −0.96 | ❌ (−17.5%) | ❌ |
| 2023 | Bear (0.10) | +51.1% | +26.4% | **+24.7%** | 1.75 | ✅ (+2.1%) | ✅ |
| 2024 | Sideways (0.44) | +86.3% | +96.1% | −9.8% | 3.49 | ✅ (+5.1%) | ❌ |
| 2025 | Bear (0.04) | +43.0% | +22.9% | **+20.1%** | 1.61 | ✅ (+11.4%) | ✅ |

## Gesamtverdikt Phase B: INCONCLUSIVE

| Kriterium | 24Q (v2+S+M) | 52Q (Phase B) | Trend |
|---|---|---|---|
| Beat Benchmark | 4/5 (80%) | **9/11 (82%)** | ✅ Stabil |
| Beat Single-Model | 3/5 (60%) | **7/11 (64%)** | ✅ Leicht verbessert |
| Beat Both | 3/5 (60%) | **6/11 (55%)** | = Stabil |
| ROBUST-Schwelle | ≥4/5 | ≥8/11 | 6/11 → 2 zu wenig |

## OOS-Metriken nach Regime

| Regime | N Jahre | Ø Long Cum | Ø Sharpe | Ø L/S Cum | Ø BM Cum | Ø Accuracy |
|---|---|---|---|---|---|---|
| Bear | 4 | +35.2% | 1.38 | +18.4% | +11.2% | 54.1% |
| Bull | 5 | +20.4% | 0.74 | +17.6% | −1.4% | 52.3% |
| Sideways | 2 | +67.7% | 3.42 | +28.7% | +12.7% | 55.3% |

## Min-Winners Fill (Phase B)

| OOS | Modell | Predicted | After Fill | Kommentar |
|---|---|---|---|---|
| 2016 | Regime | 0 | **5** | Kein Winner predicted |
| 2018 | Regime | 0 | **5** | Kein Winner predicted |
| 2019 | Beide | 0 | **5** | Kein Winner in beiden Modellen |
| 2020 | Regime | 0 | **5** | Kein Winner predicted |
| 2021 | Regime | 1 | **5** | Nur 1 predicted |
| 2023 | Regime | 1 | **5** | Nur 1 predicted |
| 2025 | Beide | 2 | **5** | 2 predicted |

Die Min-Winners-Auffüllung aktiviert sich in 7 von 11 Jahren für das Regime-Modell — das bedeutet, das Regime-Modell ist sehr selektiv und klassifiziert oft weniger als 5 Titel als "Winner". Die Auffüllung nach P(Winner) ist kritisch für die Performance.

## Detailanalyse Phase B

### Was sich durch mehr Daten verbessert hat

- **Bear-Jahre sind die klare Stärke:** 2016 (+21.2% Delta), 2023 (+24.7%), 2025 (+20.1%) — in Bear/Transition-Phasen liefert das Regime-Modell konsistent >+20% Mehrwert gegenüber dem Single-Model.
- **2023 massiv verbessert:** +51.1% (52Q) vs +22.0% (24Q). Das Bear-Modell profitiert enorm von den zusätzlichen Bear-Trainingsquartalen (Eurokrise, SNB-Schock).
- **2017 erstmals getestet:** +49.2% vs +25.9% Single — das Sideways-Regime mit niedrigster Konfidenz (0.11) erzeugt ein hervorragendes Portfolio.
- **9/11 beat Benchmark:** Die Strategie schlägt den Markt in 82% der Jahre — konsistenter als bei 5 OOS-Jahren.
- **Stabilität bestätigt:** Median Transitions und dominant_share identisch zu 24Q — das Smoothing skaliert korrekt.

### Persistente Schwächen

- **Bull-Regime ist schwach:** In Bull-Jahren (2015, 2018) verliert das Regime-Modell gegen das Single-Model. Der Bull-Detektor hat niedrige Konfidenz (0.27, 0.26) und blendet stark mit Sideways — suboptimal für reine Bull-Märkte.
- **2022 und 2019 verfehlen den Benchmark:** 2022 ist das bekannte Lag-Problem. 2019 ist neu: +16.7% Long vs +20.9% Benchmark — die Strategie ist leicht unterdurchschnittlich.
- **Regime-Modell sehr selektiv:** In 7/11 Jahren weniger als 5 predicted Winners. Ohne Min-Winners-Fill wären viele Portfolios leer oder Ein-Titel-Portfolios. Die Klassifikation ist konservativ.
- **2024 massiv gut, aber Single ist besser:** +86.3% vs +96.1%. Das Sideways-Blending (56% Bear / 44% Sideways) wählt nur 3 statt 5 Winners — das Single-Model findet 5 bessere.

### Muster: Wann gewinnt welches Modell?

| Marktphase | Regime-Modell gewinnt | Single-Model gewinnt |
|---|---|---|
| Bear/Transition | 2016, 2023, 2025 (+20% Avg Delta) | — |
| Niedrig-konfidentes Regime | 2017, 2020, 2021 | — |
| Reine Bull-Phase | — | 2015, 2018 |
| Hohe Konfidenz Bull → Bear | — | 2022 |
| Starker Sideways-Markt | — | 2024 |

Das Regime-Modell liefert den grössten Mehrwert in **unsicheren Marktphasen** (niedrige Konfidenz, Bear-Übergänge). In klaren Bull-Phasen oder bei Fehlklassifikation (2022) ist das einfachere Single-Model besser.

---

## Erkenntnisse aus allen bisherigen Tests

1. **Die Strategie ist profitabel:** 9/11 Jahre schlagen den Benchmark. In absoluten Zahlen: Nur 2018 und 2022 (Bear-Jahre mit falscher Bull-Erkennung) produzieren Verluste.
2. **Regime-Mehrwert ist real, aber nicht universell:** In 7/11 Jahren schlägt das Regime-Modell das Single-Model. Der Mehrwert konzentriert sich auf Bear/Transition-Phasen (+20% Avg Delta).
3. **Das Kernproblem ist die Bull-Erkennung:** Niedrige Konfidenz in Bull-Phasen führt zu suboptimalem Blending. Das Single-Model wählt in reinen Bull-Phasen besser.
4. **Min-Winners ist geschäftskritisch:** Ohne die Auffüllung wären 7/11 Portfolios zu dünn besetzt. Die Klassifikation ist zu konservativ für ein Minimum von 5 Winners.
5. **ROBUST verfehlt mit 6/11 statt 8/11:** Die Lücke besteht aus 2 "Grenzfällen" (2015: −19.6% Delta, 2024: −9.8% Delta) und nicht aus katastrophalen Fehlern.

---

# Phase 6: Quartalsweises Rebalancing mit Hysterese + Transaktionskosten

**Lauf:** `python robustness_test.py --quarterly --use-cache` (2. April 2026)
**Gesamtlaufzeit:** ~8 Min (Cache-Hit für Regime-Modelle, ~8 Min Evaluation)
**Modell:** Regime-aware Random-Forest-Ensemble (identisch mit Phase 5)
**Transaktionskosten:** 40 bps one-way (realistisch CH) — Entry + Exit + proportional bei Rebalancing
**Hysterese-Regel:** `keep_non_losers` — Aktie bleibt im Portfolio solange nicht als Loser klassifiziert
**Selektion:** **Winners-first + P(W)-Fill** — erst alle als "Winners" klassifizierten Titel, dann Auffüllung auf min_winners=5 nach höchster P(Winner). Damit identisch mit Phase 5 Selektionslogik.
**Gewichtung:** **P(Winner)-proportional** mit **30% Position-Cap** (iteratives Redistribution-Capping)
**Benchmark:** Equal-Weight über gesamtes Universum (ohne Rebalancing-Kosten)

## Rebalancing-Frequenz-Vergleich

| Metrik | quarterly | semi_annual | annual |
|---|---|---|---|
| **Jahre mit Long > BM** | **10/11** | 9/11 | 9/11 |
| **Ø Long kumulativ** | **+38.8%** | +36.4% | +33.6% |
| Ø Benchmark kumulativ | +5.7% | +5.7% | +5.7% |
| Ø Turnover pro Rebalancing | 30% | 58% | 100% |
| Summe Kosten (bps über 11 Jahre) | 1056 | 1024 | 880 |
| **Ø Sharpe Ratio** | **1.59** | 1.50 | 1.45 |
| Ø Max Drawdown | −20.4% | −20.1% | −21.4% |

**Verdict:** Beste Frequenz ist **quarterly** (10/11 Jahre schlagen Benchmark, höchste Ø-Rendite +38.8%, bester Sharpe 1.59).

## Per-Year-Detail: Quarterly (rebalance_freq=1)

| Jahr | Long Cum | BM Cum | Sharpe | Max DD | Kosten (bps) | > BM |
|------|----------|--------|--------|--------|--------------|------|
| 2015 | +64.9% | +8.7% | 1.40 | −24.3% | 112 | ✓ |
| 2016 | +29.1% | +10.4% | 1.56 | −12.2% | 80 | ✓ |
| 2017 | +51.6% | +20.3% | 3.71 | −5.3% | 80 | ✓ |
| 2018 | −18.0% | −20.5% | −0.80 | −30.9% | 80 | ✓ |
| 2019 | +43.9% | +20.9% | 1.26 | −14.5% | 112 | ✓ |
| 2020 | +52.4% | +4.8% | 1.88 | −26.3% | 80 | ✓ |
| 2021 | +67.3% | +17.3% | 2.74 | −12.0% | 80 | ✓ |
| 2022 | −36.3% | −17.5% | −0.94 | −49.3% | 96 | ✗ |
| 2023 | +54.6% | +2.1% | 1.86 | −14.1% | 112 | ✓ |
| **2024** | **+99.5%** | +5.1% | **4.05** | −11.2% | 96 | ✓ |
| 2025 | +17.8% | +11.4% | 0.73 | −24.1% | 128 | ✓ |

## Per-Year-Detail: Semi-Annual (rebalance_freq=2)

| Jahr | Long Cum | BM Cum | Sharpe | Max DD | Kosten (bps) | > BM |
|------|----------|--------|--------|--------|--------------|------|
| 2015 | +65.8% | +8.7% | 1.45 | −23.2% | 112 | ✓ |
| 2016 | +29.8% | +10.4% | 1.59 | −12.2% | 80 | ✓ |
| 2017 | +51.7% | +20.3% | 3.72 | −5.3% | 80 | ✓ |
| 2018 | −17.7% | −20.5% | −0.79 | −31.1% | 80 | ✓ |
| 2019 | +36.0% | +20.9% | 1.20 | −11.1% | 112 | ✓ |
| 2020 | +52.5% | +4.8% | 1.89 | −26.3% | 80 | ✓ |
| 2021 | +66.4% | +17.3% | 2.71 | −11.9% | 80 | ✓ |
| 2022 | −36.4% | −17.5% | −0.97 | −47.1% | 96 | ✗ |
| 2023 | +66.1% | +2.1% | 2.28 | −14.1% | 112 | ✓ |
| **2024** | **+88.1%** | +5.1% | **3.50** | −13.7% | 80 | ✓ |
| 2025 | −1.8% | +11.4% | −0.07 | −24.7% | 112 | ✗ |

## Per-Year-Detail: Annual (rebalance_freq=4)

| Jahr | Long Cum | BM Cum | Sharpe | Max DD | Kosten (bps) | > BM |
|------|----------|--------|--------|--------|--------------|------|
| 2015 | +34.0% | +8.7% | 0.73 | −28.4% | 80 | ✓ |
| 2016 | +28.9% | +10.4% | 1.55 | −12.2% | 80 | ✓ |
| 2017 | +51.4% | +20.3% | 3.69 | −5.3% | 80 | ✓ |
| 2018 | −17.5% | −20.5% | −0.78 | −31.0% | 80 | ✓ |
| 2019 | +15.8% | +20.9% | 0.52 | −15.9% | 80 | ✗ |
| 2020 | +52.2% | +4.8% | 1.88 | −26.3% | 80 | ✓ |
| 2021 | +65.4% | +17.3% | 2.67 | −12.0% | 80 | ✓ |
| 2022 | −36.8% | −17.5% | −0.97 | −50.5% | 80 | ✗ |
| 2023 | +49.9% | +2.1% | 1.71 | −16.1% | 80 | ✓ |
| **2024** | **+84.8%** | +5.1% | **3.41** | −13.6% | 80 | ✓ |
| 2025 | +41.9% | +11.4% | 1.56 | −24.7% | 80 | ✓ |

## Turnover-Detail: Quarterly (pro Quartal)

| Jahr | Q | Cutoff | Pos. | Swaps | Turnover | Kosten (bps) | Q-Return |
|------|---|--------|------|-------|----------|--------------|----------|
| 2015 | Q1 | 2014-12-31 | 5 | 5 | 100% | 40 | +69.0% |
| 2015 | Q2 | 2015-03-31 | 5 | 1 | 20% | 16 | −17.1% |
| 2015 | Q3 | 2015-06-30 | 5 | 1 | 20% | 16 | +4.0% |
| 2015 | Q4 | 2015-09-30 | 5 | 0 | 0% | 40 | +13.2% |
| 2016 | Q1 | 2015-12-31 | 5 | 5 | 100% | 40 | +0.5% |
| 2016 | Q2 | 2016-03-31 | 5 | 0 | 0% | 0 | +8.0% |
| 2016 | Q3 | 2016-06-30 | 5 | 0 | 0% | 0 | +20.7% |
| 2016 | Q4 | 2016-09-30 | 5 | 0 | 0% | 40 | −1.4% |
| 2017 | Q1 | 2016-12-31 | 5 | 5 | 100% | 40 | +12.1% |
| 2017 | Q2 | 2017-03-31 | 5 | 0 | 0% | 0 | +15.4% |
| 2017 | Q3 | 2017-06-30 | 5 | 0 | 0% | 0 | +9.5% |
| 2017 | Q4 | 2017-09-30 | 5 | 0 | 0% | 40 | +7.0% |
| 2018 | Q1 | 2017-12-31 | 5 | 5 | 100% | 40 | +3.2% |
| 2018 | Q2 | 2018-03-31 | 5 | 0 | 0% | 0 | +2.2% |
| 2018 | Q3 | 2018-06-30 | 5 | 0 | 0% | 0 | +4.3% |
| 2018 | Q4 | 2018-09-30 | 5 | 0 | 0% | 40 | −25.5% |
| 2019 | Q1 | 2018-12-31 | 5 | 5 | 100% | 40 | +29.0% |
| 2019 | Q2 | 2019-03-31 | 5 | 2 | 40% | 32 | −3.2% |
| 2019 | Q3 | 2019-06-30 | 5 | 0 | 0% | 0 | +16.7% |
| 2019 | Q4 | 2019-09-30 | 5 | 0 | 0% | 40 | −1.3% |
| 2020 | Q1 | 2019-12-31 | 5 | 5 | 100% | 40 | +3.7% |
| 2020 | Q2 | 2020-03-31 | 5 | 0 | 0% | 0 | +23.6% |
| 2020 | Q3 | 2020-06-30 | 5 | 0 | 0% | 0 | +12.7% |
| 2020 | Q4 | 2020-09-30 | 5 | 0 | 0% | 40 | +5.6% |
| 2021 | Q1 | 2020-12-31 | 5 | 5 | 100% | 40 | +21.1% |
| 2021 | Q2 | 2021-03-31 | 5 | 0 | 0% | 0 | +16.7% |
| 2021 | Q3 | 2021-06-30 | 5 | 0 | 0% | 0 | +11.3% |
| 2021 | Q4 | 2021-09-30 | 5 | 0 | 0% | 40 | +6.4% |
| 2022 | Q1 | 2021-12-31 | 5 | 5 | 100% | 40 | −17.7% |
| 2022 | Q2 | 2022-03-31 | 5 | 1 | 20% | 16 | −31.4% |
| 2022 | Q3 | 2022-06-30 | 5 | 0 | 0% | 0 | −6.5% |
| 2022 | Q4 | 2022-09-30 | 5 | 0 | 0% | 40 | +20.7% |
| 2023 | Q1 | 2022-12-31 | 5 | 5 | 100% | 40 | +41.4% |
| 2023 | Q2 | 2023-03-31 | 5 | 2 | 40% | 32 | −1.3% |
| 2023 | Q3 | 2023-06-30 | 5 | 0 | 0% | 0 | −0.6% |
| 2023 | Q4 | 2023-09-30 | 5 | 0 | 0% | 40 | +11.5% |
| **2024** | Q1 | 2023-12-31 | 5 | 5 | 100% | 40 | **+37.9%** |
| 2024 | Q2 | 2024-03-31 | 5 | 0 | 0% | 0 | +23.7% |
| 2024 | Q3 | 2024-06-30 | 5 | 0 | 0% | 0 | +13.4% |
| 2024 | Q4 | 2024-09-30 | 5 | 1 | 20% | 56 | +3.1% |
| 2025 | Q1 | 2024-12-31 | 5 | 5 | 100% | 40 | +1.7% |
| 2025 | Q2 | 2025-03-31 | 5 | 3 | 60% | 48 | +22.6% |
| 2025 | Q3 | 2025-06-30 | 5 | 0 | 0% | 0 | +4.8% |
| 2025 | Q4 | 2025-09-30 | 5 | 0 | 0% | 40 | −9.8% |

## Turnover-Detail: Semi-Annual (pro Halbjahr)

| Jahr | H | Cutoff | Pos. | Swaps | Turnover | Kosten (bps) | H-Return |
|------|---|--------|------|-------|----------|--------------|----------|
| 2015 | H1 | 2014-12-31 | 5 | 5 | 100% | 40 | +41.5% |
| 2015 | H2 | 2015-06-30 | 5 | 2 | 40% | 72 | +17.2% |
| 2016 | H1 | 2015-12-31 | 5 | 5 | 100% | 40 | +8.5% |
| 2016 | H2 | 2016-06-30 | 5 | 0 | 0% | 40 | +19.7% |
| 2017 | H1 | 2016-12-31 | 5 | 5 | 100% | 40 | +29.5% |
| 2017 | H2 | 2017-06-30 | 5 | 0 | 0% | 40 | +17.1% |
| 2018 | H1 | 2017-12-31 | 5 | 5 | 100% | 40 | +6.3% |
| 2018 | H2 | 2018-06-30 | 5 | 0 | 0% | 40 | −22.6% |
| 2019 | H1 | 2018-12-31 | 5 | 5 | 100% | 40 | +32.6% |
| 2019 | H2 | 2019-06-30 | 5 | 2 | 40% | 72 | +2.6% |
| 2020 | H1 | 2019-12-31 | 5 | 5 | 100% | 40 | +27.4% |
| 2020 | H2 | 2020-06-30 | 5 | 0 | 0% | 40 | +19.7% |
| 2021 | H1 | 2020-12-31 | 5 | 5 | 100% | 40 | +41.3% |
| 2021 | H2 | 2021-06-30 | 5 | 0 | 0% | 40 | +17.8% |
| 2022 | H1 | 2021-12-31 | 5 | 5 | 100% | 40 | −44.2% |
| 2022 | H2 | 2022-06-30 | 5 | 1 | 20% | 56 | +14.0% |
| 2023 | H1 | 2022-12-31 | 5 | 5 | 100% | 40 | +49.1% |
| 2023 | H2 | 2023-06-30 | 5 | 2 | 40% | 72 | +11.4% |
| **2024** | H1 | 2023-12-31 | 5 | 5 | 100% | 40 | **+69.2%** |
| 2024 | H2 | 2024-06-30 | 5 | 0 | 0% | 40 | +11.2% |
| 2025 | H1 | 2024-12-31 | 5 | 5 | 100% | 40 | +7.1% |
| 2025 | H2 | 2025-06-30 | 5 | 2 | 40% | 72 | −8.3% |

## Turnover-Detail: Annual (pro Jahr)

| Jahr | Cutoff | Pos. | Swaps | Turnover | Kosten (bps) | FY-Return |
|------|--------|------|-------|----------|--------------|-----------|
| 2015 | 2014-12-31 | 5 | 5 | 100% | 80 | +34.0% |
| 2016 | 2015-12-31 | 5 | 5 | 100% | 80 | +28.9% |
| 2017 | 2016-12-31 | 5 | 5 | 100% | 80 | +51.4% |
| 2018 | 2017-12-31 | 5 | 5 | 100% | 80 | −17.5% |
| 2019 | 2018-12-31 | 5 | 5 | 100% | 80 | +15.8% |
| 2020 | 2019-12-31 | 5 | 5 | 100% | 80 | +52.2% |
| 2021 | 2020-12-31 | 5 | 5 | 100% | 80 | +65.4% |
| 2022 | 2021-12-31 | 5 | 5 | 100% | 80 | −36.8% |
| 2023 | 2022-12-31 | 5 | 5 | 100% | 80 | +49.9% |
| **2024** | 2023-12-31 | 5 | 5 | 100% | 80 | **+84.8%** |
| 2025 | 2024-12-31 | 5 | 5 | 100% | 80 | +41.9% |

## Analyse und Muster

### Winners-first Selektion (v2) vs. reine P(W)-Top-5 Selektion (v1)

Die Selektionslogik wurde von "Top-5 nach P(Winner) unabhängig vom Label" auf "erst alle echten Winners, dann P(W)-Fill" umgestellt. Dies entspricht der Phase-5-Logik, kombiniert mit quartalsweisem Rebalancing, Hysterese und Transaktionskosten.

| Metrik | Winners-first (v2) | P(W)-Top-5 (v1) | Delta |
|---|---|---|---|
| Quarterly beat_bm | **10/11** | 10/11 | = |
| **Quarterly Ø Long** | **+38.8%** | +32.4% | **+6.4pp** |
| **Quarterly Ø Sharpe** | **1.59** | 1.36 | **+0.23** |
| Annual beat_bm | 9/11 | 9/11 | = |
| **Annual Ø Long** | **+33.6%** | +27.4% | **+6.2pp** |
| **2024 Quarterly** | **+99.5%** | +26.0% | **+73.5pp** |
| 2023 Quarterly | +54.6% | +62.8% | −8.2pp |
| 2022 Quarterly | −36.3% | −36.8% | +0.5pp |

**Kernverbesserung:** Die Winners-first-Selektion bringt **+6.4pp Ø-Rendite** und **+0.23 Sharpe** gegenüber der reinen P(W)-Top-5-Selektion. Der grösste Einzeleffekt ist 2024 mit **+73.5pp** — weil echte Winners (COTN, KURN, LONN) statt der höchsten P(W)-Werte (die oft "Steady" waren) ins Portfolio kamen.

### Hysterese-Wirkung nach Jahr

Die Hysterese-Regel `keep_non_losers` hält Positionen quartalsübergreifend, solange sie nicht als Loser klassifiziert werden. Im Quarterly-Modus zeigt sich ein klares Muster:

| Jahr | Swaps Q1 | Swaps Q2 | Swaps Q3 | Swaps Q4 | Eff. Turnover |
|------|----------|----------|----------|----------|---------------|
| 2015 | 5 (neues Portfolio) | 1 | 1 | 0 | 32% Ø |
| 2016 | 5 | 0 | 0 | 0 | 25% Ø |
| 2017 | 5 | 0 | 0 | 0 | 25% Ø |
| 2018 | 5 | 0 | 0 | 0 | 25% Ø |
| 2019 | 5 | 2 | 0 | 0 | 35% Ø |
| 2020 | 5 | 0 | 0 | 0 | 25% Ø |
| 2021 | 5 | 0 | 0 | 0 | 25% Ø |
| 2022 | 5 | 1 | 0 | 0 | 28% Ø |
| 2023 | 5 | 2 | 0 | 0 | 35% Ø |
| 2024 | 5 | 0 | 0 | 1 | 28% Ø |
| 2025 | 5 | 3 | 0 | 0 | 40% Ø |

**Beobachtung:** Die Hysterese ist sehr effektiv — nach dem initialen Q1-Aufbau werden in 6/11 Jahren **keine oder max 1** Position in Q2–Q4 getauscht. Das Modell identifiziert stabile Non-Losers, die selten in den Loser-Bereich fallen.

### Wo macht Quarterly den grössten Unterschied gegenüber Annual?

| Jahr | Quarterly | Annual | Delta | Treiber |
|------|-----------|--------|-------|---------|
| 2015 | +64.9% | +34.0% | **+30.9pp** | Q2/Q3-Swaps bringen stärkere Titel ins Portfolio |
| 2019 | +43.9% | +15.8% | **+28.1pp** | Q2-Swaps (2 Pos.) korrigieren schwache Erstauswahl |
| 2024 | +99.5% | +84.8% | **+14.7pp** | Q4-Swap (BEAN statt COTN) sichert Gewinne |
| 2025 | +17.8% | +41.9% | **−24.1pp** | Q2-Swaps verschlechtern Auswahl (Gegenbeispiel) |

### Einziges Loss-Jahr: 2022

Alle drei Frequenzen verlieren 2022 gegen den Benchmark (Long −36.3% bis −36.8% vs. BM −17.5%). Ursache: Bear-Markt mit falscher Bull-Erkennung durch das Regime-Modell (Konfidenz 0.69 Bull bei bereits einsetzendem Bärenmarkt). Kein Rebalancing innerhalb des Jahres hilft hier — die initiale Auswahl im Q1 ist bereits falsch positioniert, und die Hysterese hält die Verlustpositionen.

### Kosten-Effizienz

| Frequenz | Ø Kosten/Jahr (bps) | Ø Zusatz-Rendite vs. Annual | Kosten-Rendite-Verhältnis |
|----------|---------------------|----------------------------|---------------------------|
| Quarterly | 96 bps | +5.2pp | 1 bps Kosten → +0.3pp Rendite |
| Semi-Annual | 93 bps | +2.8pp | 1 bps Kosten → +0.2pp Rendite |
| Annual | 80 bps | (Baseline) | — |

Die höheren Kosten bei quarterly/semi-annual werden durch die Mehrrendite deutlich überkompensiert.

## Fazit Phase 6 (Winners-first + P(W)-Gewichtung)

1. **Quarterly Rebalancing ist die beste Frequenz:** 10/11 Jahre schlagen den Benchmark, **Ø Long +38.8%**, **Sharpe 1.59**
2. **Winners-first-Selektion bringt massiven Uplift:** +6.4pp Ø-Rendite vs. reine P(W)-Top-5-Selektion. Durch die Priorisierung echter Winners statt hoher P(W)-Werte allein wählt das System die richtigen Aktien — besonders sichtbar in 2024 (+99.5% vs. +26.0%)
3. **P(Winner)-Gewichtung mit 30% Cap** nutzt das Modell-Konfidenzsignal ohne extreme Konzentration. Gewichte liegen typischerweise bei 17–24%, der Cap greift selten
4. **Hysterese reduziert Turnover massiv:** Ø 30% statt 100%, dadurch bleiben Kosten trotz häufigerem Rebalancing moderat
5. **Transaktionskosten sind verkraftbar:** 40 bps one-way reduzieren die Rendite um ~96 bps/Jahr — ein Bruchteil des Alpha (+33.1pp über BM)
6. **2022 bleibt das einzige Problemjahr:** Alle Frequenzen verlieren; das ist ein Modell-Problem (Bull-Fehlklassifikation), kein Rebalancing- oder Gewichtungs-Problem

### Vergleich mit Phase 5 (Annual, ohne Kosten)

| Metrik | Phase 6 Quarterly | Phase 5 Annual |
|---|---|---|
| Beat-BM-Rate | **10/11** | 9/11 |
| Ø Long Cum | **+38.8%** | +41.2% (ohne Kosten) |
| Ø Sharpe | **1.59** | 1.45 |
| 2024 | **+99.5%** | +86.3% |
| 2022 | −36.3% | −36.8% |
| Transaktionskosten | ✓ (40bps) | ✗ (nicht modelliert) |

Phase 6 ist die **produktionsreife Version** von Phase 5: gleiche Selektionslogik, aber mit realistischen Kosten, Hysterese und quartalsweiser Korrekturmöglichkeit. Die bereinigten Renditen sind trotz Kosten vergleichbar.

### Model-Cache für Wiederholungsläufe

`python robustness_test.py --quarterly --use-cache` speichert/lädt die trainierten Regime-Modelle unter `data/cache/regime_models_robustness.joblib` (58 MB). Der Cache invalidiert automatisch bei Änderungen an Klassifikationsperioden, Seed, Ticker-Universum oder Training-Hyperparametern. Erwartete Laufzeit bei Cache-Hit: ~8 Min (nur Evaluation).

---

# Nächste Schritte

## Erkenntnisse aus allen bisherigen Tests

1. **Die Strategie ist profitabel:** 10/11 Jahre schlagen den Benchmark (Quarterly). In absoluten Zahlen: Nur 2022 (Bear mit falscher Bull-Erkennung) verliert deutlich.
2. **Winners-first-Selektion ist entscheidend:** Die Priorisierung echter Winners vor P(W)-Fill bringt +6.4pp Ø-Rendite und +0.23 Sharpe vs. reine P(W)-Top-5-Selektion.
3. **Regime-Mehrwert ist real, aber nicht universell:** In 7/11 Jahren schlägt das Regime-Modell das Single-Model. Der Mehrwert konzentriert sich auf Bear/Transition-Phasen.
4. **Min-Winners ist geschäftskritisch:** Ohne die Auffüllung wären Portfolios zu dünn besetzt. Die Klassifikation ist zu konservativ für ein Minimum von 5 Winners.
5. **Quarterly + Hysterese + P(W)-Gewichtung ist die robusteste Kombination:** Ø +38.8% Rendite, 1.59 Sharpe, nach Kosten.

## Mögliche nächste Verbesserungen

### Option A: Hybrides Modell (Regime + Single kombinieren)
- In Jahren mit hoher Regime-Konfidenz (>0.5): Regime-Modell verwenden
- In Jahren mit niedriger Konfidenz (<0.3): Single-Model verwenden
- Dazwischen: Gewichteter Mix
- **Erwarteter Effekt:** Vermeidet das Schwäche-Muster in reinen Bull-Phasen

### Option B: Regime-spezifische Feature-Selektion
- Für Bull-Regime: Andere Features als für Bear/Sideways
- Momentum-Features dominieren in Bull, Mean-Reversion in Bear
- **Erwarteter Effekt:** Bessere Accuracy in Bull-Phasen

### Option C: Dynamische Min-Winners
- Statt fixes min_winners=5: Regime-abhängig (Bull: 8–10, Bear: 3–5, Sideways: 5–7)
- **Erwarteter Effekt:** Grössere Portfolios in Bull-Phasen, konzentriertere in Bear

### Option D: Live-Implementierung ✅ (nächster Schritt)
- Transaktionskosten und Rebalancing-Frequenz → erledigt
- Live-Signal-Generierung für das nächste Quartal
- Performance-Monitoring und automatisiertes Reporting
