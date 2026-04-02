# Ergebnis-Zusammenfassung: Swiss SPI Extra Forward-Test

**Modell:** Random Forest Classifier, trainiert auf Q1 2024 (SPI Extra, 135 Aktien nach Liquiditätsfilter)
**Forward-Test:** Gesamtjahr 2025 (136 Aktien)
**Features:** 28 (18 technisch + 10 fundamental), Cutoff vor Klassifikationsperiode (kein Lookahead)

## Modell-Performance

| Metrik | In-Sample (Q1 2024, Hold-out 25%) | Out-of-Sample (2025) | Delta |
|---|---|---|---|
| Accuracy | 44.1% | **48.5%** | +4.4pp |
| F1 macro | 0.392 | **0.421** | +0.03 |
| Stichprobe | 34 Aktien | 136 Aktien | — |

Kein Overfitting erkennbar: OOS leicht besser als In-Sample.
Zufalls-Baseline (3 Klassen): 33.3% — Modell liegt ~15pp darüber.

## Strategie-Renditen (2025, exkl. Kosten)

| Strategie | Kumulierter Return | Pro 1 CHF → | Sharpe Ratio | Max Drawdown |
|---|---|---|---|---|
| **Long Winners** | **+28.5%** | **1.29 CHF** | 1.40 | -19.8% |
| **Long/Short** | +10.9% | 1.11 CHF | **1.55** | -3.6% |
| **Benchmark** (Equal Weight) | +9.7% | 1.10 CHF | 0.83 | -14.1% |

- Long Winners schlägt Benchmark um ~19 Prozentpunkte.
- Long/Short hat den besten Sharpe (1.55) und geringsten Drawdown (-3.6%).

## Trefferquoten pro Gruppe

| Vorhergesagte Gruppe | Anzahl | Davon korrekt | Trefferquote |
|---|---|---|---|
| Winners | 19 | 8 | 42.1% |
| Steady | 82 | 46 | 56.1% |
| Losers | 35 | 12 | 34.3% |

Das Modell erkennt Steady-Aktien am besten (56%) und tendiert dazu, zu viele Aktien als Steady zu klassifizieren (82/136).

## Tatsächliche Returns nach vorhergesagter Gruppe

| Vorhergesagte Gruppe | Mittlerer Return 2025 | Median Return |
|---|---|---|
| Winners | **+33.5%** | +4.5% |
| Steady | +8.6% | +11.6% |
| Losers | +5.3% | -5.3% |

Die Sortierung funktioniert (Winners > Steady > Losers), aber der Median der Winners (+4.5%) zeigt, dass der hohe Durchschnitt von wenigen Extremgewinnern getrieben wird.

## Fazit

**Stärken:**
- Strategie schlägt Benchmark, insbesondere risikoadjustiert (Sharpe).
- Kein Overfitting: OOS leicht besser als In-Sample.
- Gruppensortierung geht in die richtige Richtung.

**Schwächen:**
- 48.5% Accuracy bei 3 Klassen ist besser als Zufall, aber nicht überzeugend.
- Winner-Erkennung schwach (42%) — nur 8/19 korrekt.
- Erfolg hängt stark von wenigen Outperformern ab (fragile Strategie).
- Transaktionskosten, Slippage und Liquiditätsprobleme bei Swiss Small Caps nicht berücksichtigt.

**Empfehlung:** Das Modell zeigt ein schwaches, aber reales Signal. Für produktiven Einsatz müsste die Präzision steigen — durch bessere Features, mehr Trainingsdaten (mehrere Quartale), oder einen einfacheren Ansatz (z.B. Top/Bottom-Dezile statt Quartile).

## Phase 6: Quartals-Rebalancing (Robustness, OOS 2015–2025)

**Lauf:** `python robustness_test.py --quarterly` (2. Apr. 2026, Gesamtlaufzeit ~38 min inkl. einmaligem Training der regime-aware Modelle).

**Setup:** Regime-aware Random-Forest-Ensemble (wie Phase 5), 40 bps Transaktionskosten one-way, Hysterese-Regel `keep_non_losers`, Top‑5 Long-Positionen. Drei Evaluierungen mit identischen trainierten Modellen: **quarterly** (rebalance_freq=1), **semi_annual** (2), **annual** (4).

### Rebalancing-Frequenz-Vergleich (gegen Equal-Weight-Benchmark)

| Metrik | quarterly | semi_annual | annual |
|--------|------------|-------------|--------|
| Jahre mit Long > BM (von 11) | 10/11 | 9/11 | 10/11 |
| Ø Long kumulativ | +34.2% | +32.2% | +28.8% |
| Ø Benchmark kumulativ | +5.7% | +5.7% | +5.7% |
| Ø Turnover (gemeldet) | 28% | 56% | 100% |
| Summe Kosten (total_costs, Einheit wie Report) | 992 | 992 | 880 |
| Ø Sharpe (Long) | 1.36 | 1.31 | 1.28 |
| Ø Max Drawdown (Long) | −19.5% | −19.4% | −20.5% |

### Kurzinterpretation

- **Quarterly** liefert die höchste durchschnittliche Long-Rendite und den besten Sharpe; das Report nennt sie als beste Frequenz (10 Jahre schlagen den Benchmark, Ø Long +34.2%).
- **2022** ist bei allen drei Frequenzen schwächer als der Benchmark (Bärenmarkt); das ist das einzige Jahr mit klar negativem Long-Cum unter quarterly/semi_annual.
- **Semi-annual** verliert 2025 gegenüber dem BM (+2.9% vs. +11.4%), obwohl annual und quarterly dort noch schlagen — mehr zwischenjährliche Umschichtung hilft hier nicht zwingend.
- Höhere Rebalancing-Frequenz erhöht modellierte Kosten und Turnover, bleibt aber in diesem Lauf netto vorteilhaft gegenüber seltenerem Rebalancing.

### Regime-Model-Cache (Wiederholungsläufe)

`python robustness_test.py --quarterly --use-cache` lädt oder speichert die trainierten Regime-Modelle unter `data/cache/regime_models_robustness.joblib` (Hash aus `CLASSIFICATION_PERIODS`, Seed, Ticker-Universum und Trainings-Hyperparametern). `--no-cache` erzwingt Neutraining ohne Cache. Die Evaluation über alle OOS-Jahre läuft in jedem Fall vollständig.
