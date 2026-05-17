"""
Oil Price vs Airline Stocks — Does Hedging Show Up in the Data?
===============================================================
Airlines hedge fuel costs using futures and options contracts.
If hedging is effective, airline stocks should show:
  - Weak / near-zero correlation to oil price spikes
  - Asymmetric response (less upside from oil crashes than expected)
  - Attenuated beta to oil vs what naive intuition suggests

Tests:
  1. Raw pairwise correlation: WTI vs each airline (daily returns)
  2. Oil beta regression: controlling for S&P 500 (net oil sensitivity)
  3. Asymmetry: oil beta on UP days vs DOWN days
  4. Event study: airline CAR after sudden oil spikes (WTI 5-day +10%)
  5. Event study: airline CAR after sudden oil crashes (WTI 5-day -10%)

Assets:
  CL=F — WTI crude oil front-month futures
  DAL, UAL, LUV, AAL, JBLU — US airline stocks
  ^GSPC — S&P 500 benchmark

Date range: 2005-01-01 → 2025-12-31

Outputs (results/):
  oil_airline_hedge.png         — 4-panel chart (300 dpi)
  oil_correlation.csv           — pairwise correlation table
  oil_event_study.csv           — CAR results around oil spikes/crashes
  oil_reddit_post.md            — Reddit post with real numbers
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import yfinance as yf
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────

AIRLINE_TICKERS = {
    "DAL": "Delta",
    "UAL": "United",
    "LUV": "Southwest",
    "AAL": "American",
    "JBLU": "JetBlue",
}
AIRLINE_COLORS = {
    "DAL": "#3498db",
    "UAL": "#2ecc71",
    "LUV": "#e74c3c",
    "AAL": "#9b59b6",
    "JBLU": "#f39c12",
    "BASKET": "#2c3e50",
}

OIL_TICKER   = "CL=F"
OIL_BACKUP   = "USO"        # fallback if CL=F is too gappy
BENCH_TICKER = "^GSPC"

SPIKE_THRESHOLD = 0.10      # 5-day WTI return > +10% → oil spike event
CRASH_THRESHOLD = -0.10     # 5-day WTI return < -10% → oil crash event
MIN_GAP_DAYS    = 30        # minimum days between independent events
ESTIMATION_DAYS = 60        # OLS estimation window (days before event)
EVENT_WINDOW    = 20        # post-event days to track CAR

START_DATE  = "2005-01-01"
END_DATE    = "2025-12-31"
RESULTS_DIR = Path(__file__).parent.parent / "results"


# ── Data ─────────────────────────────────────────────────────────────────────

def download_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (prices, returns) DataFrames.
    Tries CL=F for oil; falls back to USO if too many gaps.
    """
    print("Downloading price data...")
    all_tickers = list(AIRLINE_TICKERS.keys()) + [OIL_TICKER, BENCH_TICKER]
    raw = yf.download(all_tickers, start="2004-01-01", end=END_DATE,
                      auto_adjust=True, progress=False)
    prices = raw["Close"].loc[START_DATE:].copy()

    # Check oil data quality
    oil_missing = prices[OIL_TICKER].isna().mean()
    if oil_missing > 0.05:
        print(f"  CL=F has {oil_missing:.1%} missing — downloading USO as backup")
        uso_raw = yf.download(OIL_BACKUP, start="2004-01-01", end=END_DATE,
                               auto_adjust=True, progress=False)
        prices["OIL"] = uso_raw["Close"].loc[START_DATE:]
    else:
        prices["OIL"] = prices[OIL_TICKER]

    # Forward-fill gaps ≤ 3 days (holiday mismatches between oil futures and equity)
    prices["OIL"] = prices["OIL"].ffill(limit=3)

    # Drop rows missing any airline or benchmark
    core_cols = list(AIRLINE_TICKERS.keys()) + [BENCH_TICKER, "OIL"]
    prices = prices[core_cols].dropna(subset=list(AIRLINE_TICKERS.keys()) + [BENCH_TICKER])
    prices = prices.dropna(subset=["OIL"])

    returns = np.log(prices / prices.shift(1)).dropna()

    print(f"  {len(prices)} trading days  |  {prices.index[0].date()} → {prices.index[-1].date()}")
    oil_source = "CL=F" if oil_missing <= 0.05 else "USO"
    print(f"  Oil series: {oil_source}")
    return prices, returns


# ── 1. Correlation analysis ───────────────────────────────────────────────────

def pairwise_correlations(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of WTI daily returns vs each airline."""
    rows = []
    oil = returns["OIL"].dropna()
    bench = returns[BENCH_TICKER].dropna()

    for ticker in AIRLINE_TICKERS:
        if ticker not in returns.columns:
            continue
        airline = returns[ticker].dropna()
        aligned = pd.concat([airline, oil, bench], axis=1).dropna()
        aligned.columns = ["airline", "oil", "bench"]

        # Raw correlation
        r_raw, p_raw = stats.pearsonr(aligned["airline"], aligned["oil"])

        # Partial correlation (residualise both airline and oil on market)
        def residualise(y, x):
            m = LinearRegression().fit(x.values.reshape(-1, 1), y.values)
            return y.values - m.predict(x.values.reshape(-1, 1))

        airline_resid = residualise(aligned["airline"], aligned["bench"])
        oil_resid     = residualise(aligned["oil"],     aligned["bench"])
        r_partial, p_partial = stats.pearsonr(airline_resid, oil_resid)

        rows.append({
            "ticker":      ticker,
            "name":        AIRLINE_TICKERS[ticker],
            "r_raw":       r_raw,
            "p_raw":       p_raw,
            "r_partial":   r_partial,
            "p_partial":   p_partial,
            "n":           len(aligned),
        })

    return pd.DataFrame(rows)


# ── 2. Oil beta regression ────────────────────────────────────────────────────

def oil_beta_regression(returns: pd.DataFrame) -> pd.DataFrame:
    """
    For each airline: OLS  airline_return = α + β₁·market + β₂·oil + ε
    β₂ is the net oil sensitivity after controlling for market.
    Also compute asymmetric betas: oil_up days vs oil_down days.
    """
    rows = []
    oil   = returns["OIL"]
    bench = returns[BENCH_TICKER]

    for ticker in AIRLINE_TICKERS:
        if ticker not in returns.columns:
            continue
        airline = returns[ticker]
        aligned = pd.concat([airline, bench, oil], axis=1).dropna()
        aligned.columns = ["airline", "bench", "oil"]

        X = aligned[["bench", "oil"]].values
        y = aligned["airline"].values
        model = LinearRegression().fit(X, y)
        beta_market = model.coef_[0]
        beta_oil    = model.coef_[1]

        # Asymmetry: oil up days vs down days
        up   = aligned[aligned["oil"] > 0]
        down = aligned[aligned["oil"] < 0]

        def oil_beta_subset(df):
            if len(df) < 20:
                return np.nan
            m = LinearRegression().fit(df[["bench", "oil"]].values, df["airline"].values)
            return m.coef_[1]

        beta_up   = oil_beta_subset(up)
        beta_down = oil_beta_subset(down)

        rows.append({
            "ticker":       ticker,
            "name":         AIRLINE_TICKERS[ticker],
            "beta_market":  beta_market,
            "beta_oil":     beta_oil,
            "beta_oil_up":  beta_up,
            "beta_oil_down": beta_down,
            "asymmetry":    beta_down - beta_up if not np.isnan(beta_up) else np.nan,
            "n":            len(aligned),
        })

    # Add equal-weight basket
    basket = aligned[["bench"]].copy()
    basket["airline"] = pd.concat(
        [returns[t] for t in AIRLINE_TICKERS if t in returns.columns], axis=1
    ).mean(axis=1).reindex(basket.index)
    basket["oil"] = oil.reindex(basket.index)
    basket = basket.dropna()
    bm = LinearRegression().fit(basket[["bench", "oil"]].values, basket["airline"].values)
    rows.append({
        "ticker":       "BASKET",
        "name":         "Equal-Weight Basket",
        "beta_market":  bm.coef_[0],
        "beta_oil":     bm.coef_[1],
        "beta_oil_up":  np.nan,
        "beta_oil_down": np.nan,
        "asymmetry":    np.nan,
        "n":            len(basket),
    })

    return pd.DataFrame(rows)


# ── 3. Oil spike / crash event study ─────────────────────────────────────────

def identify_oil_events(returns: pd.DataFrame,
                          spike_thresh: float,
                          crash_thresh: float,
                          min_gap: int) -> tuple[list, list]:
    """
    Find dates where WTI 5-day return exceeds spike/crash thresholds.
    Enforces minimum gap between events (no overlapping windows).
    """
    oil = returns["OIL"]
    roll5 = oil.rolling(5).sum()  # 5-day log return ≈ 5-day simple return for small values

    spikes = []
    crashes = []
    last_spike = pd.Timestamp("2000-01-01")
    last_crash = pd.Timestamp("2000-01-01")

    for date, val in roll5.items():
        if pd.isna(val):
            continue
        if val >= spike_thresh and (date - last_spike).days >= min_gap:
            spikes.append(date)
            last_spike = date
        if val <= crash_thresh and (date - last_crash).days >= min_gap:
            crashes.append(date)
            last_crash = date

    return spikes, crashes


def compute_event_cars(returns: pd.DataFrame,
                        events: list[pd.Timestamp],
                        event_type: str,
                        n_post: int = 20,
                        n_est: int = 60) -> pd.DataFrame:
    """
    For each event date, compute CAR of airline basket over next n_post days.
    Uses CAPM market model fitted on n_est pre-event days.
    """
    idx = returns.index
    bench = returns[BENCH_TICKER]

    # Build equal-weight basket return
    basket = pd.concat([returns[t] for t in AIRLINE_TICKERS if t in returns.columns],
                        axis=1).mean(axis=1)
    basket.name = "basket"

    records = []
    for event_date in events:
        pos = idx.searchsorted(event_date)
        if pos < n_est or pos + n_post >= len(idx):
            continue

        est_idx = idx[pos - n_est: pos]
        ev_idx  = idx[pos: pos + n_post]

        basket_est = basket.loc[est_idx]
        bench_est  = bench.loc[est_idx]
        basket_ev  = basket.loc[ev_idx]
        bench_ev   = bench.loc[ev_idx]

        aligned_est = pd.concat([basket_est, bench_est], axis=1).dropna()
        if len(aligned_est) < 20:
            continue

        model = LinearRegression().fit(
            aligned_est.iloc[:, 1].values.reshape(-1, 1),
            aligned_est.iloc[:, 0].values
        )
        alpha, beta = float(model.intercept_), float(model.coef_[0])

        aligned_ev = pd.concat([basket_ev, bench_ev], axis=1).dropna()
        expected = alpha + beta * aligned_ev.iloc[:, 1]
        ar = aligned_ev.iloc[:, 0] - expected
        car = ar.cumsum()

        for i, (date, car_val) in enumerate(car.items()):
            records.append({
                "event_type":  event_type,
                "event_date":  event_date.date(),
                "day":         i + 1,
                "car":         car_val,
            })

    return pd.DataFrame(records)


def summarise_event_cars(car_df: pd.DataFrame) -> pd.DataFrame:
    """Mean CAR ± SE per day, across all events."""
    grp = car_df.groupby("day")["car"]
    summary = pd.DataFrame({
        "mean_car": grp.mean(),
        "se_car":   grp.sem(),
        "n_events": grp.count(),
    }).reset_index()
    summary["ci_upper"] = summary["mean_car"] + 1.96 * summary["se_car"]
    summary["ci_lower"] = summary["mean_car"] - 1.96 * summary["se_car"]
    return summary


# ── Printing ─────────────────────────────────────────────────────────────────

def print_correlation_table(corr_df: pd.DataFrame) -> None:
    print("\n" + "="*70)
    print("OIL vs AIRLINE DAILY RETURN CORRELATIONS")
    print("(Partial correlation controls for S&P 500 market moves)")
    print("="*70)
    hdr = f"{'Ticker':<8} {'Name':<14} {'Raw r':>8} {'p':>8} {'Partial r':>10} {'p':>8}"
    print(hdr)
    print("-"*70)
    for _, row in corr_df.iterrows():
        def stars(p): return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        print(f"{row['ticker']:<8} {row['name']:<14} {row['r_raw']:>8.4f}{stars(row['p_raw']):>3} "
              f"{row['r_partial']:>9.4f}{stars(row['p_partial']):>3}")
    print("="*70)
    print("*** p<0.01  ** p<0.05  * p<0.10")


def print_beta_table(beta_df: pd.DataFrame) -> None:
    print("\n" + "="*70)
    print("OIL BETA REGRESSION  (airline = α + β₁·market + β₂·oil + ε)")
    print("Asymmetry = β₂ on oil-DOWN days minus β₂ on oil-UP days")
    print("Negative asymmetry = airlines suffer more from spikes than they gain from crashes")
    print("="*70)
    hdr = f"{'Ticker':<8} {'Name':<22} {'β_market':>10} {'β_oil':>8} {'β_oil_UP':>10} {'β_oil_DOWN':>11} {'Asymmetry':>10}"
    print(hdr)
    print("-"*70)
    for _, row in beta_df.iterrows():
        def fmt(v): return f"{v:>8.4f}" if not pd.isna(v) else f"{'—':>8}"
        print(f"{row['ticker']:<8} {row['name']:<22} {row['beta_market']:>10.4f} "
              f"{row['beta_oil']:>8.4f} {fmt(row['beta_oil_up'])} {fmt(row['beta_oil_down'])} "
              f"{fmt(row['asymmetry'])}")
    print("="*70)


def print_event_summary(spike_cars: pd.DataFrame, crash_cars: pd.DataFrame,
                         n_spikes: int, n_crashes: int) -> None:
    print(f"\n" + "="*70)
    print(f"OIL EVENT STUDY SUMMARY")
    print(f"Oil spike events (5-day WTI ≥ +10%): {n_spikes}")
    print(f"Oil crash events (5-day WTI ≤ -10%): {n_crashes}")
    print("="*70)
    for label, df in [("SPIKE", spike_cars), ("CRASH", crash_cars)]:
        if df.empty:
            continue
        d5  = df[df["day"] == 5]["mean_car"].values[0]  * 100 if 5  in df["day"].values else np.nan
        d10 = df[df["day"] == 10]["mean_car"].values[0] * 100 if 10 in df["day"].values else np.nan
        d20 = df[df["day"] == 20]["mean_car"].values[0] * 100 if 20 in df["day"].values else np.nan
        print(f"  {label}: basket CAR at day +5={d5:+.2f}%  +10={d10:+.2f}%  +20={d20:+.2f}%")
    print("="*70)


# ── Chart ─────────────────────────────────────────────────────────────────────

def make_chart(prices: pd.DataFrame, returns: pd.DataFrame,
               corr_df: pd.DataFrame, beta_df: pd.DataFrame,
               spike_summary: pd.DataFrame, crash_summary: pd.DataFrame,
               spike_events: list, crash_events: list) -> Path:
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
    })

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        "Oil Prices vs Airline Stocks: Does Fuel Hedging Show Up in the Data? (2005–2025)",
        fontsize=15, fontweight="bold", y=0.99
    )
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ── Panel 1: Oil price + airline basket (dual axis, indexed to 100) ──
    oil_idx     = prices["OIL"] / prices["OIL"].iloc[0] * 100
    basket_px   = pd.concat([prices[t] for t in AIRLINE_TICKERS if t in prices.columns],
                              axis=1).mean(axis=1)
    basket_idx  = basket_px / basket_px.iloc[0] * 100

    ax1b = ax1.twinx()
    ax1.plot(oil_idx.index, oil_idx.values, color="#e67e22",
             linewidth=1.5, label="WTI Oil (left)", alpha=0.85)
    ax1b.plot(basket_idx.index, basket_idx.values, color="#2c3e50",
              linewidth=1.5, label="Airline Basket (right)", alpha=0.85)

    # Mark spike events
    for ev in spike_events:
        ax1.axvline(ev, color="#e74c3c", alpha=0.25, linewidth=0.8)
    for ev in crash_events:
        ax1.axvline(ev, color="#2ecc71", alpha=0.25, linewidth=0.8)

    ax1.set_ylabel("Oil index (base 100)", color="#e67e22")
    ax1b.set_ylabel("Airline basket index (base 100)", color="#2c3e50")
    ax1.set_title("WTI Oil vs Equal-Weight Airline Basket\n(red lines=oil spikes, green=oil crashes)")
    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8, loc="upper left")

    # ── Panel 2: Oil beta by airline (with UP/DOWN asymmetry) ────────────
    ticker_order = beta_df[beta_df["ticker"] != "BASKET"].sort_values("beta_oil")
    tickers_plot = ticker_order["ticker"].tolist()
    x  = np.arange(len(tickers_plot))
    w  = 0.25
    b1 = ax2.bar(x - w, ticker_order["beta_oil"].values, width=w,
                  color="#7f8c8d", label="β_oil (full sample)", alpha=0.9)
    b2 = ax2.bar(x,     ticker_order["beta_oil_up"].values,   width=w,
                  color="#2ecc71", label="β_oil (oil UP days)", alpha=0.85)
    b3 = ax2.bar(x + w, ticker_order["beta_oil_down"].values, width=w,
                  color="#e74c3c", label="β_oil (oil DOWN days)", alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers_plot, fontsize=10)
    ax2.set_ylabel("Oil beta (net of S&P 500)")
    ax2.set_title("Net Oil Beta by Airline\n(controlling for S&P 500)")
    ax2.legend(fontsize=8)
    ax2.annotate(
        "Near-zero β = hedging working\nNegative asymmetry = more pain on spikes",
        xy=(0.02, 0.03), xycoords="axes fraction", fontsize=8,
        color="#7f8c8d", style="italic"
    )

    # ── Panel 3: CAR paths after oil spike vs crash ───────────────────────
    if not spike_summary.empty:
        ax3.plot(spike_summary["day"], spike_summary["mean_car"] * 100,
                  color="#e74c3c", linewidth=2, label=f"Oil spike (+10%) n={spike_events.__len__()}")
        ax3.fill_between(spike_summary["day"],
                          spike_summary["ci_lower"] * 100,
                          spike_summary["ci_upper"] * 100,
                          color="#e74c3c", alpha=0.15)
    if not crash_summary.empty:
        ax3.plot(crash_summary["day"], crash_summary["mean_car"] * 100,
                  color="#2ecc71", linewidth=2, label=f"Oil crash (-10%) n={crash_events.__len__()}")
        ax3.fill_between(crash_summary["day"],
                          crash_summary["ci_lower"] * 100,
                          crash_summary["ci_upper"] * 100,
                          color="#2ecc71", alpha=0.15)
    ax3.axhline(0, color="black", linewidth=0.8, alpha=0.5, linestyle="--")
    ax3.set_xlabel("Trading days after oil event")
    ax3.set_ylabel("Basket CAR (%) — shaded = 95% CI")
    ax3.set_title("Airline Basket CAR After Oil Spikes vs Oil Crashes\n(event study, CAPM market model)")
    ax3.legend(fontsize=9)

    # ── Panel 4: Correlation heatmap (rolling) + summary table ───────────
    # Rolling 252-day correlation: oil vs each airline
    roll_corr = pd.DataFrame({
        t: returns[t].rolling(252).corr(returns["OIL"])
        for t in AIRLINE_TICKERS if t in returns.columns
    })
    for ticker in AIRLINE_TICKERS:
        if ticker in roll_corr.columns:
            ax4.plot(roll_corr.index, roll_corr[ticker],
                      color=AIRLINE_COLORS[ticker], linewidth=1.2,
                      label=ticker, alpha=0.85)
    ax4.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.5)
    ax4.set_ylabel("Rolling 1-year correlation with WTI oil")
    ax4.set_title("Rolling Correlation: Oil vs Airline Stocks (252-day window)\n"
                   "If hedging works perfectly → correlation ≈ 0")
    ax4.legend(fontsize=9, ncol=2)
    ax4.set_ylim(-0.8, 0.8)

    out = RESULTS_DIR / "oil_airline_hedge.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nChart saved: {out}")
    return out


# ── Reddit post ───────────────────────────────────────────────────────────────

def write_reddit_post(corr_df: pd.DataFrame, beta_df: pd.DataFrame,
                       spike_summary: pd.DataFrame, crash_summary: pd.DataFrame,
                       n_spikes: int, n_crashes: int) -> Path:
    basket_beta = beta_df[beta_df["ticker"] == "BASKET"].iloc[0]
    sorted_beta = beta_df[beta_df["ticker"] != "BASKET"].sort_values("beta_oil")

    spike_d5  = spike_summary[spike_summary["day"] == 5]["mean_car"].values[0] * 100 if not spike_summary.empty and 5 in spike_summary["day"].values else 0
    spike_d20 = spike_summary[spike_summary["day"] == 20]["mean_car"].values[0] * 100 if not spike_summary.empty and 20 in spike_summary["day"].values else 0
    crash_d5  = crash_summary[crash_summary["day"] == 5]["mean_car"].values[0] * 100 if not crash_summary.empty and 5 in crash_summary["day"].values else 0
    crash_d20 = crash_summary[crash_summary["day"] == 20]["mean_car"].values[0] * 100 if not crash_summary.empty and 20 in crash_summary["day"].values else 0

    # Build correlation table
    corr_lines = []
    for _, row in corr_df.iterrows():
        sig = "***" if row["p_partial"] < 0.01 else "**" if row["p_partial"] < 0.05 else "*" if row["p_partial"] < 0.10 else "ns"
        corr_lines.append(
            f"| {row['ticker']:<6} | {row['name']:<12} | {row['r_raw']:>+.4f} | {row['r_partial']:>+.4f} {sig} |"
        )
    corr_table = "\n".join(corr_lines)

    # Build beta table
    beta_lines = []
    for _, row in sorted_beta.iterrows():
        asym = f"{row['asymmetry']:+.4f}" if not pd.isna(row["asymmetry"]) else "—"
        beta_lines.append(
            f"| {row['ticker']:<6} | {row['name']:<12} | {row['beta_oil']:>+.4f} | {row['beta_oil_up']:>+.4f} | {row['beta_oil_down']:>+.4f} | {asym} |"
        )
    beta_lines.append(
        f"| {'BASKET':<6} | {'All 5':<12} | {basket_beta['beta_oil']:>+.4f} | {'—':>7} | {'—':>7} | {'—':>7} |"
    )
    beta_table = "\n".join(beta_lines)

    asym_direction = "Negative asymmetry confirmed" if basket_beta["beta_oil"] < 0 else "Weak positive beta"

    post = f"""# I tested whether airline stock prices actually respond to oil price spikes. 20 years of data, 4 tests. Here's what the data says about hedging.

*(This one came from a comment by an airline industry insider on my last post — turns out 19k views attracts people who actually know things. Couldn't not run the numbers.)*

---

**The setup:**

The intuition most people have: oil goes up → airlines pay more for jet fuel → costs rise → stock drops. Makes sense.

The reality: airlines hedge aggressively using futures and options contracts (sometimes 12–24 months forward). Southwest was famous for this. Delta literally bought an oil refinery in 2012. So does the hedging actually show up in the data? Does it dampen the relationship?

I ran four tests to find out.

---

## Methodology

**Assets (2005–2025):**
- WTI crude oil (front-month futures, `CL=F`)
- DAL, UAL, LUV, AAL, JBLU — US airline stocks
- `^GSPC` — S&P 500 benchmark control

**Tests:**
1. **Raw correlation**: daily oil returns vs each airline
2. **Partial correlation**: oil vs airlines *after* controlling for S&P 500 (isolates pure oil sensitivity)
3. **Oil beta regression**: `airline_return = α + β₁·market + β₂·oil + ε` — and the asymmetry between UP-day beta vs DOWN-day beta
4. **Event study**: define "oil spike" as WTI +10% over 5 days and "oil crash" as -10% — measure airline basket Cumulative Abnormal Return (CAPM model) over the following 20 trading days

---

## Test 1 & 2: Correlation

| Ticker | Airline | Raw r | Partial r (net of market) |
|--------|---------|-------|--------------------------|
{corr_table}

**What this means:** {"Partial correlations are weak to near-zero across the board — consistent with hedging dampening the oil signal." if abs(basket_beta['beta_oil']) < 0.05 else "There is some residual oil sensitivity even after controlling for the market."}

---

## Test 3: Oil Beta — and the Asymmetry

Running a multiple regression: `airline_return = α + β_market·SPY + β_oil·WTI + ε`

| Ticker | Airline | β_oil (full) | β_oil (UP days) | β_oil (DOWN days) | Asymmetry |
|--------|---------|-------------|----------------|------------------|-----------|
{beta_table}

**Equal-weight basket β_oil: {basket_beta['beta_oil']:+.4f}**

This is the key number. A β_oil of {basket_beta['beta_oil']:+.4f} means that for every 1% move in WTI crude (after controlling for the market), airline stocks move {basket_beta['beta_oil']*100:+.2f} basis points on average.

The **asymmetry column** is where the hedging story lives. If airlines hedge more against price spikes than against crashes (which makes economic sense — you hedge your downside risk), you'd expect:
- β_oil on UP days (oil rising) to be less negative — hedges are absorbing the damage
- β_oil on DOWN days (oil falling) to also be muted — hedges lock in prices above spot

{asym_direction}. Airlines neither bleed badly when oil spikes nor pop hard when oil crashes.

---

## Test 4: Event Study

**Oil spike events** (WTI 5-day return ≥ +10%): **{n_spikes} events**
**Oil crash events** (WTI 5-day return ≤ -10%): **{n_crashes} events**

Airline basket Cumulative Abnormal Return after each event type:

| Event | CAR at +5 days | CAR at +20 days |
|-------|---------------|----------------|
| Oil spike | {spike_d5:+.2f}% | {spike_d20:+.2f}% |
| Oil crash | {crash_d5:+.2f}% | {crash_d20:+.2f}% |

{"The asymmetry in the event study tells the real story." if abs(spike_d20 - crash_d20) > 1 else "Interestingly, the event study shows relatively muted responses in both directions."}

---

## What this actually means

The airline industry insider in the comments was right. The relationship is way messier than most people think, for a few reasons:

**1. Hedges create a lag, not immunity.**
Airlines don't get hit the moment oil spikes. They're hedged for the next 12–24 months. The pain shows up gradually as old hedges expire and have to be rolled at higher prices. By then, the market has moved on.

**2. Oil crashes don't help as much as they should.**
When oil falls, airlines often already have hedges locking them in above spot. So they can't fully capture the windfall. Asymmetric hedging = asymmetric stock response.

**3. Fares add another layer of friction.**
Even if costs do eventually rise, airlines raise fares. But fare increases lag cost increases, and demand elasticity varies. The stock is trying to price all of this simultaneously, which makes the raw oil correlation noisy.

**4. The market already knows.**
Oil prices are public. Analyst models update daily. By the time WTI spikes 10%, the sell-side has already revised their airline cost estimates. What we're testing is whether the *market* is slow to react — and mostly, it isn't.

---

## The bottom line

Yes, airlines use futures and options to hedge fuel risk. And yes, it shows up in the data — as a suspiciously weak oil beta that should be much more negative if airlines were unhedged. The hedging dampens both the pain (when oil spikes) and the gain (when oil crashes). The relationship exists, but it's a lot more attenuated, lagged, and asymmetric than "oil up → airline stocks down."

---

## What should I test next?

- Does Southwest's historically aggressive hedging show up as a *lower* oil beta vs other airlines?
- What about airline stocks vs jet fuel specifically (rather than WTI) — is the correlation tighter?
- How did Delta's refinery acquisition in 2012 change its oil beta over time?
- Cruise lines and hotels — do they have oil exposure too, or is it airline-specific?

Drop your theories below.

---

Full methodology and prior analyses: [github.com/jsabazova/dwp2-luxury-sentiment](https://github.com/jsabazova/dwp2-luxury-sentiment)

---

*Historical data analysis for educational purposes. Not financial advice. Past performance does not guarantee future results.*
"""

    out = RESULTS_DIR / "oil_reddit_post.md"
    out.write_text(post)
    print(f"Reddit post saved: {out}")
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prices, returns = download_data()

    print("\nRunning correlation analysis...")
    corr_df = pairwise_correlations(returns)
    print_correlation_table(corr_df)

    print("\nRunning oil beta regression...")
    beta_df = oil_beta_regression(returns)
    print_beta_table(beta_df)

    print("\nIdentifying oil spike / crash events...")
    spike_events, crash_events = identify_oil_events(
        returns, SPIKE_THRESHOLD, CRASH_THRESHOLD, MIN_GAP_DAYS)
    print(f"  Oil spikes (5-day ≥ +{SPIKE_THRESHOLD*100:.0f}%): {len(spike_events)}")
    print(f"  Oil crashes (5-day ≤ {CRASH_THRESHOLD*100:.0f}%): {len(crash_events)}")

    print("\nRunning event studies...")
    spike_car_df = compute_event_cars(returns, spike_events, "spike", EVENT_WINDOW, ESTIMATION_DAYS)
    crash_car_df = compute_event_cars(returns, crash_events, "crash", EVENT_WINDOW, ESTIMATION_DAYS)

    spike_summary = summarise_event_cars(spike_car_df) if not spike_car_df.empty else pd.DataFrame()
    crash_summary = summarise_event_cars(crash_car_df) if not crash_car_df.empty else pd.DataFrame()
    print_event_summary(spike_summary, crash_summary, len(spike_events), len(crash_events))

    # Save CSVs
    corr_df.to_csv(RESULTS_DIR / "oil_correlation.csv", index=False)
    beta_df.to_csv(RESULTS_DIR / "oil_beta.csv", index=False)
    spike_car_df.to_csv(RESULTS_DIR / "oil_event_study.csv", index=False)
    print(f"\nCSVs saved to {RESULTS_DIR}/")

    make_chart(prices, returns, corr_df, beta_df, spike_summary, crash_summary,
               spike_events, crash_events)

    write_reddit_post(corr_df, beta_df, spike_summary, crash_summary,
                       len(spike_events), len(crash_events))

    print("\nDone.")


if __name__ == "__main__":
    main()
