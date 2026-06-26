# Gap memo — OFI predictability & tradability

*Drafted 2026-06-26. Working title: "Order Flow Imbalance on NSE: A
Reconstruction-Audited, Queue-Aware Study of Predictability and Tradability."*

## The four anchor papers

### 1. Cont, Cucuringu & Zhang (2023) — *Cross-Impact of OFI in Equity Markets*, Quant Finance 23(10) (arXiv 2112.13213)
- **What:** Combine OFI from the top *M* levels of the book into a single
  **integrated OFI** variable (first principal component of the multi-level
  OFIs). Study contemporaneous price impact and lagged cross-asset impact.
- **Key results:**
  - *"Once the information from multiple levels is integrated into OFI,
    multi-asset models with cross-impact do not provide additional explanatory
    power for contemporaneous impact compared to a sparse model without
    cross-impact terms."*
  - *"Lagged cross-asset OFIs do improve the forecasting of future returns,"*
    but the effect *"mainly manifests at short-term horizons and decays rapidly
    in time."*
- **Data/venue:** Nasdaq equities (US). [VERIFY exact tickers/period and the
  reported PC1 variance share + loading uniformity from the PDF — abstract
  doesn't give the numbers; this matters for our contrast below.]

### 2. Kolm, Turiel & Westray (2023) — *Deep Order Flow Imbalance*, Math Finance 33(4) (SSRN 3900141)
- **What:** Deep learning (incl. LSTM) forecasting HF returns at **multiple
  horizons** for **115 Nasdaq stocks** from granular book/flow inputs.
- **Key results:**
  - Models trained on **order flow significantly outperform** models trained on
    raw order books.
  - Forecastability links to microstructure characteristics — "information-rich"
    stocks predict better.
  - **Effective horizon ≈ two average price changes** — a stock-specific,
    event-time horizon, not a fixed clock horizon.
- **Venue:** Nasdaq (US).

### 3. Briola, Bartolucci & Aste (2024/2025) — *Deep LOB Forecasting* / LOBFrame (arXiv 2403.09267)
- **What:** Deep LOB mid-price forecasting across heterogeneous Nasdaq stocks;
  releases the open LOBFrame codebase.
- **The Gap-1 quotes (this is the crux of our paper):**
  - *"High forecasting power does not necessarily correspond to actionable
    trading signals."*
  - *"Traditional machine learning metrics fail to adequately assess the quality
    of forecasts in the Limit Order Book context."*
  - They propose an **operational framework** scoring *"the probability of
    accurately forecasting complete transactions"* instead of R²/accuracy.
- **What they did NOT do:** close the gap with a **queue-position-realistic
  execution layer**. They reframe evaluation but stop short of a full
  queue-aware taker simulator.

### 4. Albers, Cucuringu, Howison & Shestopaloff (2025) — *The Market Maker's Dilemma* (arXiv 2502.18625)
- **What:** Models the **negative correlation between fill likelihood and
  post-fill returns** (adverse selection): easy-to-fill orders are less
  profitable.
- **Side / data:** **Maker-side**, and on **Binance BTC perpetual futures** —
  *not* equities, not NSE/Nasdaq/LSE.
- **Queue-aware:** Yes — explicitly models *"the interplay between returns, queue
  sizes, and orders' queue positions."*
- **Main result:** Viable maker strategies are often **contrarian**, counter-
  trading prevailing book imbalance.

## How our angle differs (the defensible wedge)

| Axis | Albers 2025 | Briola 2024 | **Ours** |
|---|---|---|---|
| Side | Maker | Forecast-only | **Taker / aggressor** |
| Queue model | Yes (maker fill) | No | **Yes (taker fill + adverse selection)** |
| Tradability bridge | Partial (maker) | Flagged, open | **Closes it for OFI takers** |
| Asset class / venue | Crypto perps | Nasdaq | **NSE equities** |
| Reconstruction noise | Ignored | Ignored | **Measured (Gap 2)** |

- **vs Briola:** they *flagged* "R² ≠ tradability"; we *close* it for
  taker-side OFI strategies with a queue-realistic simulator.
- **vs Albers:** they did the maker side on crypto; we do the **taker side on
  NSE equities** with reconstruction auditing.
- **vs CCZ/KTW:** we replicate their integrated/multi-horizon OFI on a **new
  market (NSE)** and stress it through execution + reconstruction error.

## What our prototype already shows (LOBSTER AAPL, today)
- Multi-level integrated OFI (PC1) implemented with dual normalisation.
- **On AAPL (wide-tick regime), integrated PC1 underperforms naive L1** (R²
  0.005 vs 0.068) and full-vector OLS barely beats L1 (0.070) — deep levels add
  almost nothing. **Not a normalisation artifact** (scalar-Q ≈ per-level-Q).
- This *contradicts* the CCZ near-uniform/high-share PC1 picture and is the seed
  of a **regime-dependence** result — the hook for the NSE spread/tick binning.
- OOS predictive R² ≈ 0 past t+1 for every family — consistent with Briola's
  "R² ≠ tradability," now shown for integrated/deep OFI, not just L1.

## Open decisions / risks
1. **NSE TBT data + official snapshots — do we have them?** The whole Gap-3 +
   Gap-2 novelty depends on this. Not present in the repo; "Project Karna"
   unconfirmed. **This is the #1 blocker.**
2. **First venue:** workshop (ICAIF/FMI) tolerates a lighter sim; Tier-1 journal
   (QF / JFM / JFEc) needs the full queue model + cross-section.
3. **Verify CCZ's exact PC1 variance share / loading uniformity** so our
   "regime crossover" contrast is precise, not paraphrased.

## Next steps (ranked)
1. Confirm NSE data availability — gates everything.
2. Add a **small-tick LOBSTER ticker** to demonstrate the PC1 regime crossover
   (works today, no NSE needed) → first real figure.
3. Design the **taker-side queue-aware execution simulator** (the contribution).
4. Write the **reconstruction-error budget** experiment on `--validate` output.
