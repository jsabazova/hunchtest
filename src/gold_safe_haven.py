"""
Gold as a Safe Haven — Multi-Dimensional Analysis
==================================================
Tests four claims about gold:
  1. Gold is an inflation hedge (real returns vs stocks & bonds)
  2. Gold outperforms stocks in a crash (crisis-period analysis)
  3. Gold rises on big-dump days (worst-decile SPY day behaviour)
  4. Gold outperforms bonds (10Y/30Y treasuries) over the long run

Assets:
  GLD  — SPDR Gold Trust (gold proxy, started Nov 2004)
  SPY  — S&P 500 ETF
  IEF  — iShares 7–10 Year Treasury Bond ETF
  TLT  — iShares 20+ Year Treasury Bond ETF (30Y proxy)

Date range: 2005-01-01 → 2025-12-31
Inflation: CPI-U year-end values (BLS), linearly interpolated daily

Outputs (results/):
  gold_safe_haven.png       — 4-panel publication chart (300 dpi)
  gold_crisis_returns.csv   — crisis period returns per asset
  gold_summary.csv          — long-term stats per asset
  gold_reddit_post.md       — Reddit post with real numbers
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

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────

TICKERS = {
    "GLD": "Gold (GLD)",
    "SPY": "S&P 500 (SPY)",
    "TLT": "20+Y Treasury (TLT)",
    "IEF": "7–10Y Treasury (IEF)",
}

COLORS = {
    "GLD": "#f39c12",   # gold
    "SPY": "#2ecc71",   # green
    "TLT": "#3498db",   # blue
    "IEF": "#9b59b6",   # purple
}

# Named crisis periods: (SPY drawdown start, trough)
CRISIS_PERIODS = {
    "GFC\n(Oct'07–Mar'09)":          ("2007-10-09", "2009-03-09"),
    "Debt Ceiling\n(Apr–Oct'11)":     ("2011-04-29", "2011-10-03"),
    "China/Oil\n(May'15–Feb'16)":     ("2015-05-21", "2016-02-11"),
    "Q4 Selloff\n(Sep–Dec'18)":       ("2018-09-20", "2018-12-24"),
    "COVID Crash\n(Feb–Mar'20)":      ("2020-02-19", "2020-03-23"),
    "2022 Bear\n(Jan–Oct'22)":        ("2021-12-31", "2022-10-12"),
    "Tariff Shock\n(Feb–Apr'25)":     ("2025-02-19", "2025-04-08"),
}

# CPI-U year-end values (BLS, All Urban Consumers, not seasonally adjusted)
# Source: U.S. Bureau of Labor Statistics
CPI_YEAR_END = {
    2004: 190.3,  2005: 196.8,  2006: 201.8,  2007: 210.0,
    2008: 210.2,  2009: 215.9,  2010: 219.2,  2011: 225.7,
    2012: 229.6,  2013: 233.0,  2014: 234.8,  2015: 236.5,
    2016: 241.4,  2017: 246.5,  2018: 251.2,  2019: 256.6,
    2020: 260.5,  2021: 278.8,  2022: 296.8,  2023: 306.7,
    2024: 314.2,  2025: 321.5,
}

START_DATE  = "2005-01-01"
END_DATE    = "2025-12-31"
RESULTS_DIR = Path(__file__).parent.parent / "results"


# ── Data ─────────────────────────────────────────────────────────────────────

def download_prices() -> pd.DataFrame:
    print("Downloading price data...")
    raw = yf.download(list(TICKERS.keys()), start="2004-11-19", end=END_DATE,
                      auto_adjust=True, progress=False)
    prices = raw["Close"].dropna(how="all")
    # Trim to common start (GLD IPO ~Nov 2004, full data from Jan 2005)
    prices = prices.loc[START_DATE:]
    prices = prices.dropna()
    print(f"  {len(prices)} trading days, {len(prices.columns)} assets")
    print(f"  {prices.index[0].date()} → {prices.index[-1].date()}")
    return prices


def build_cpi_series(idx: pd.DatetimeIndex) -> pd.Series:
    """Daily CPI via linear interpolation between year-end values."""
    years = sorted(CPI_YEAR_END.keys())
    # Build year-end anchor dates
    anchors = pd.Series(
        {pd.Timestamp(f"{y}-12-31"): CPI_YEAR_END[y] for y in years}
    ).sort_index()
    # Reindex to daily and interpolate
    combined_idx = anchors.index.union(idx)
    cpi_daily = anchors.reindex(combined_idx).interpolate(method="time")
    return cpi_daily.reindex(idx)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


# ── Long-term cumulative returns ─────────────────────────────────────────────

def cumulative_growth(prices: pd.DataFrame, cpi: pd.Series,
                       initial: float = 10_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nominal and real cumulative growth of $10,000."""
    # Normalise to 100 at start
    norm = prices / prices.iloc[0] * initial

    # Real: deflate by CPI relative to start
    cpi_rel = cpi / cpi.iloc[0]
    real = norm.div(cpi_rel, axis=0)

    return norm, real


def long_term_stats(prices: pd.DataFrame, cpi: pd.Series) -> pd.DataFrame:
    """CAGR, vol, Sharpe, max drawdown — nominal and real — per asset."""
    rows = []
    n_years = (prices.index[-1] - prices.index[0]).days / 365.25
    cpi_annual_inflation = (cpi.iloc[-1] / cpi.iloc[0]) ** (1 / n_years) - 1

    for ticker in TICKERS:
        if ticker not in prices.columns:
            continue
        px = prices[ticker].dropna()
        ret = np.log(px / px.shift(1)).dropna()

        total_return = px.iloc[-1] / px.iloc[0] - 1
        cagr = (px.iloc[-1] / px.iloc[0]) ** (1 / n_years) - 1
        vol  = ret.std() * np.sqrt(252)
        sharpe = (ret.mean() * 252) / (ret.std() * np.sqrt(252))

        # Max drawdown
        roll_max = px.cummax()
        dd = (px - roll_max) / roll_max
        max_dd = dd.min()

        # Real CAGR
        cpi_aligned = cpi.reindex(px.index).interpolate()
        real_px = px / (cpi_aligned / cpi_aligned.iloc[0])
        real_cagr = (real_px.iloc[-1] / real_px.iloc[0]) ** (1 / n_years) - 1

        rows.append({
            "ticker":       ticker,
            "name":         TICKERS[ticker],
            "total_return": total_return * 100,
            "nominal_cagr": cagr * 100,
            "real_cagr":    real_cagr * 100,
            "annual_vol":   vol * 100,
            "sharpe":       sharpe,
            "max_drawdown": max_dd * 100,
        })

    df = pd.DataFrame(rows)
    return df, cpi_annual_inflation * 100


# ── Crisis analysis ───────────────────────────────────────────────────────────

def crisis_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Total return for each asset during each crisis period."""
    rows = []
    for crisis_name, (start, end) in CRISIS_PERIODS.items():
        row = {"crisis": crisis_name.replace("\n", " ")}
        start_ts = pd.Timestamp(start)
        end_ts   = pd.Timestamp(end)

        for ticker in TICKERS:
            if ticker not in prices.columns:
                continue
            px = prices[ticker]
            # Find nearest available trading days
            avail = px.loc[start_ts:end_ts].dropna()
            if len(avail) < 2:
                row[ticker] = np.nan
                continue
            ret = (avail.iloc[-1] / avail.iloc[0] - 1) * 100
            row[ticker] = ret

        rows.append(row)
    return pd.DataFrame(rows)


# ── Bad-day analysis ─────────────────────────────────────────────────────────

def bad_day_analysis(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket SPY daily returns into deciles (worst → best).
    For each decile, compute average return of all assets.
    """
    spy = returns["SPY"].dropna()
    decile_labels = [f"D{i+1}" for i in range(10)]
    spy_deciles = pd.qcut(spy, q=10, labels=decile_labels)

    rows = []
    for decile in decile_labels:
        mask = spy_deciles == decile
        row = {"decile": decile,
               "spy_mean": spy[mask].mean() * 100,
               "n_days":   mask.sum()}
        for ticker in TICKERS:
            if ticker in returns.columns:
                aligned = returns[ticker].reindex(spy.index)
                row[ticker] = aligned[mask].mean() * 100
        rows.append(row)

    return pd.DataFrame(rows)


def big_dump_stats(returns: pd.DataFrame,
                    threshold: float = -0.02) -> pd.DataFrame:
    """
    On days when SPY falls > threshold (e.g. -2%), what do other assets do?
    """
    spy = returns["SPY"].dropna()
    bad_days = spy[spy < threshold]
    print(f"\nBig dump days (SPY < {threshold*100:.0f}%): {len(bad_days)}")

    rows = []
    for ticker in TICKERS:
        if ticker not in returns.columns:
            continue
        aligned = returns[ticker].reindex(bad_days.index).dropna()
        pct_positive = (aligned > 0).mean() * 100
        avg_ret = aligned.mean() * 100
        t_stat, p_val = stats.ttest_1samp(aligned.values, 0)
        rows.append({
            "ticker":       ticker,
            "name":         TICKERS[ticker],
            "avg_return":   avg_ret,
            "pct_positive": pct_positive,
            "t_stat":       t_stat,
            "p_value":      p_val,
            "n_days":       len(aligned),
        })

    return pd.DataFrame(rows)


# ── Correlation ───────────────────────────────────────────────────────────────

def rolling_correlation(returns: pd.DataFrame,
                          asset: str = "GLD",
                          benchmark: str = "SPY",
                          window: int = 90) -> pd.Series:
    """Rolling n-day Pearson correlation between asset and benchmark."""
    return returns[asset].rolling(window).corr(returns[benchmark])


# ── Printing ─────────────────────────────────────────────────────────────────

def print_long_term_table(stats_df: pd.DataFrame, inflation: float) -> None:
    print("\n" + "="*80)
    print("LONG-TERM PERFORMANCE (2005–2025, auto-adjusted total return)")
    print(f"Avg annual CPI inflation over period: {inflation:.2f}%")
    print("="*80)
    header = (f"{'Asset':<24} {'Total Ret':>10} {'Nom CAGR':>10} {'Real CAGR':>10} "
              f"{'Vol':>8} {'Sharpe':>8} {'Max DD':>9}")
    print(header)
    print("-"*80)
    for _, row in stats_df.iterrows():
        print(f"{row['name']:<24} {row['total_return']:>9.1f}% {row['nominal_cagr']:>9.2f}% "
              f"{row['real_cagr']:>9.2f}% {row['annual_vol']:>7.2f}% {row['sharpe']:>8.3f} "
              f"{row['max_drawdown']:>8.1f}%")
    print("="*80)


def print_crisis_table(crisis_df: pd.DataFrame) -> None:
    print("\n" + "="*80)
    print("CRISIS PERIOD TOTAL RETURNS (start of crisis → trough)")
    print("="*80)
    tickers = [t for t in TICKERS if t in crisis_df.columns]
    header = f"{'Crisis':<30}" + "".join(f"{t:>10}" for t in tickers)
    print(header)
    print("-"*80)
    for _, row in crisis_df.iterrows():
        line = f"{row['crisis']:<30}"
        for t in tickers:
            val = row.get(t, np.nan)
            if pd.isna(val):
                line += f"{'N/A':>10}"
            else:
                marker = " ✓" if t == "GLD" and val > row.get("SPY", -999) else ""
                line += f"{val:>+9.1f}%"
        print(line)
    print("="*80)
    print("Gold ✓ = gold outperformed SPY in that crisis")


def print_bad_day_table(dump_df: pd.DataFrame, threshold: float) -> None:
    print(f"\n" + "="*80)
    print(f"BEHAVIOUR ON SPY'S WORST DAYS (SPY daily return < {threshold*100:.0f}%)")
    print("="*80)
    header = (f"{'Asset':<24} {'Avg Return':>11} {'% Positive':>11} "
              f"{'t-stat':>8} {'p-value':>9} {'N days':>7}")
    print(header)
    print("-"*80)
    for _, row in dump_df.iterrows():
        stars = "***" if row["p_value"] < 0.01 else "**" if row["p_value"] < 0.05 \
                else "*" if row["p_value"] < 0.10 else ""
        print(f"{row['name']:<24} {row['avg_return']:>+10.3f}% {row['pct_positive']:>10.1f}% "
              f"{row['t_stat']:>8.3f} {row['p_value']:>8.4f}{stars:>2} {row['n_days']:>7}")
    print("="*80)
    print("t-test H0: mean return = 0 on bad SPY days.  *** p<0.01  ** p<0.05  * p<0.10")


# ── Chart ─────────────────────────────────────────────────────────────────────

def make_chart(prices: pd.DataFrame, cpi: pd.Series,
               crisis_df: pd.DataFrame, bad_day_df: pd.DataFrame,
               returns: pd.DataFrame, stats_df: pd.DataFrame) -> Path:
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size":   10,
        "axes.titlesize":   12,
        "axes.titleweight": "bold",
    })

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        "Is Gold a Safe Haven? A Multi-Dimensional Analysis (2005–2025)",
        fontsize=16, fontweight="bold", y=0.99
    )
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ── Panel 1: Cumulative growth $10k (log scale) ───────────────────────
    norm, real = cumulative_growth(prices, cpi, initial=10_000)
    for ticker in ["SPY", "GLD", "TLT", "IEF"]:
        if ticker not in norm.columns:
            continue
        ax1.plot(norm.index, norm[ticker], color=COLORS[ticker],
                 linewidth=2, label=f"{ticker} (nominal)", alpha=0.9)
        ax1.plot(real.index, real[ticker], color=COLORS[ticker],
                 linewidth=1.2, linestyle="--", alpha=0.55)
    ax1.set_yscale("log")
    ax1.set_title("Cumulative Growth of $10,000\n(solid = nominal, dashed = inflation-adjusted)")
    ax1.set_ylabel("Portfolio value ($, log scale)")
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax1.legend(fontsize=8, ncol=2)
    # Shade crisis periods
    for _, (s, e) in list(CRISIS_PERIODS.items())[:4]:
        ax1.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.07, color="grey")

    # ── Panel 2: Crisis period returns (grouped bars) ────────────────────
    crisis_clean = crisis_df.copy()
    crisis_clean["crisis_short"] = [
        c.replace("(", "\n(") for c in crisis_clean["crisis"]
    ]
    crisis_names = crisis_clean["crisis"].tolist()
    n = len(crisis_names)
    tickers_plot = ["SPY", "GLD", "TLT", "IEF"]
    tickers_plot = [t for t in tickers_plot if t in crisis_clean.columns]
    bar_width = 0.18
    x = np.arange(n)
    offsets = np.linspace(-(len(tickers_plot)-1)/2 * bar_width,
                           (len(tickers_plot)-1)/2 * bar_width,
                           len(tickers_plot))
    for i, ticker in enumerate(tickers_plot):
        vals = crisis_clean[ticker].fillna(0).values
        bars = ax2.bar(x + offsets[i], vals, width=bar_width,
                        color=COLORS[ticker], label=ticker,
                        edgecolor="white", linewidth=0.5, alpha=0.88)
    ax2.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax2.set_title("Total Return by Asset During Each Crisis\n(start of SPY drawdown → trough)")
    ax2.set_ylabel("Total Return (%)")
    ax2.set_xticks(x)
    short_names = [list(CRISIS_PERIODS.keys())[i] for i in range(n)]
    ax2.set_xticklabels(short_names, fontsize=7.5)
    ax2.legend(fontsize=9)

    # ── Panel 3: Gold vs SPY on SPY worst-decile days ───────────────────
    bad_df_deciles = bad_day_analysis(returns)
    deciles = bad_df_deciles["decile"].tolist()
    x3 = np.arange(len(deciles))
    w = 0.22
    off3 = np.linspace(-1.5*w, 1.5*w, 4)
    for i, ticker in enumerate(["SPY", "GLD", "TLT", "IEF"]):
        if ticker not in bad_df_deciles.columns:
            continue
        vals3 = bad_df_deciles[ticker].values
        ax3.bar(x3 + off3[i], vals3, width=w,
                color=COLORS[ticker], label=ticker,
                edgecolor="white", linewidth=0.4, alpha=0.88)
    ax3.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax3.set_title("Average Daily Return by Asset Across SPY Return Deciles\n"
                   "(D1 = worst 10% of SPY days, D10 = best 10%)")
    ax3.set_ylabel("Avg Daily Return (%)")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(deciles, fontsize=9)
    ax3.set_xlabel("SPY Return Decile")
    ax3.legend(fontsize=9)
    ax3.annotate("← Worst SPY days", xy=(0.02, 0.04),
                  xycoords="axes fraction", fontsize=8, color="#7f8c8d")
    ax3.annotate("Best SPY days →", xy=(0.72, 0.04),
                  xycoords="axes fraction", fontsize=8, color="#7f8c8d")

    # ── Panel 4: Summary stats table ─────────────────────────────────────
    ax4.axis("off")
    table_rows = []
    for _, row in stats_df.iterrows():
        table_rows.append([
            row["ticker"],
            f"{row['nominal_cagr']:.2f}%",
            f"{row['real_cagr']:.2f}%",
            f"{row['annual_vol']:.1f}%",
            f"{row['sharpe']:.2f}",
            f"{row['max_drawdown']:.1f}%",
        ])
    col_headers = ["Asset", "Nom CAGR", "Real CAGR", "Vol", "Sharpe", "Max DD"]
    tbl = ax4.table(
        cellText=table_rows,
        colLabels=col_headers,
        cellLoc="center",
        loc="upper center",
        bbox=[0.0, 0.42, 1.0, 0.52],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for j in range(len(col_headers)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, row in enumerate(table_rows):
        tbl[i+1, 0].set_text_props(color=list(COLORS.values())[i], fontweight="bold")

    # Add big-dump stats below table
    ax4.text(0.5, 0.36, "Average Return on SPY's Worst Days (< −2%)",
              ha="center", va="top", fontsize=10, fontweight="bold",
              transform=ax4.transAxes)
    dump_rows = []
    for _, row in bad_day_df.iterrows():
        stars = "***" if row["p_value"] < 0.01 else "**" if row["p_value"] < 0.05 \
                else "*" if row["p_value"] < 0.10 else "ns"
        dump_rows.append([
            row["ticker"],
            f"{row['avg_return']:+.3f}%",
            f"{row['pct_positive']:.0f}%",
            stars,
        ])
    dump_headers = ["Asset", "Avg Return", "% Positive", "Sig"]
    tbl2 = ax4.table(
        cellText=dump_rows,
        colLabels=dump_headers,
        cellLoc="center",
        loc="lower center",
        bbox=[0.0, 0.0, 1.0, 0.33],
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(10)
    for j in range(len(dump_headers)):
        tbl2[0, j].set_facecolor("#7f8c8d")
        tbl2[0, j].set_text_props(color="white", fontweight="bold")
    for i, row in enumerate(dump_rows):
        tbl2[i+1, 0].set_text_props(
            color=list(COLORS.values())[i], fontweight="bold")

    ax4.set_title("Summary Statistics (2005–2025)", fontsize=12, fontweight="bold")

    out = RESULTS_DIR / "gold_safe_haven.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nChart saved: {out}")
    return out


# ── Reddit post ───────────────────────────────────────────────────────────────

def write_reddit_post(stats_df: pd.DataFrame, inflation: float,
                       crisis_df: pd.DataFrame, dump_df: pd.DataFrame) -> Path:
    # Pull key numbers
    gold  = stats_df[stats_df["ticker"] == "GLD"].iloc[0]
    spy   = stats_df[stats_df["ticker"] == "SPY"].iloc[0]
    tlt   = stats_df[stats_df["ticker"] == "TLT"].iloc[0]
    ief   = stats_df[stats_df["ticker"] == "IEF"].iloc[0]

    gold_dump = dump_df[dump_df["ticker"] == "GLD"].iloc[0]
    spy_dump  = dump_df[dump_df["ticker"] == "SPY"].iloc[0]
    tlt_dump  = dump_df[dump_df["ticker"] == "TLT"].iloc[0]

    # Crisis scorecard: how many crises did gold outperform SPY?
    gold_beats_spy = sum(
        1 for _, row in crisis_df.iterrows()
        if not pd.isna(row.get("GLD")) and not pd.isna(row.get("SPY"))
        and row["GLD"] > row["SPY"]
    )
    total_crises = len(crisis_df)

    # Find gold's best and worst crises
    valid = crisis_df.dropna(subset=["GLD", "SPY"])
    gold_spread = valid["GLD"] - valid["SPY"]
    best_crisis_idx = gold_spread.idxmax()
    worst_crisis_idx = gold_spread.idxmin()
    best_crisis_name = valid.loc[best_crisis_idx, "crisis"]
    worst_crisis_name = valid.loc[worst_crisis_idx, "crisis"]
    best_crisis_gold = valid.loc[best_crisis_idx, "GLD"]
    best_crisis_spy  = valid.loc[best_crisis_idx, "SPY"]
    worst_crisis_gold = valid.loc[worst_crisis_idx, "GLD"]
    worst_crisis_spy  = valid.loc[worst_crisis_idx, "SPY"]

    dump_sig = "significant" if gold_dump["p_value"] < 0.10 else "not statistically significant"

    # Build crisis table for the post
    crisis_lines = []
    for _, row in crisis_df.iterrows():
        gld_val = row.get("GLD", np.nan)
        spy_val = row.get("SPY", np.nan)
        tlt_val = row.get("TLT", np.nan)
        if pd.isna(gld_val):
            continue
        gold_won = "✅" if gld_val > spy_val else "❌"
        crisis_lines.append(
            f"| {row['crisis']:<35} | {spy_val:>+7.1f}% | {gld_val:>+7.1f}% {gold_won} | {tlt_val:>+7.1f}% |"
        )

    crisis_table = "\n".join(crisis_lines)

    post = f"""# I tested whether gold is actually a safe haven. 20 years of data. Here's what I found.

*(Follow-up from my airline seasonality post — someone in the comments asked about gold. Here's the full study.)*

---

**The claim:** Gold protects you when markets crash. It's the "world is ending" asset. It hedges inflation. It beats bonds over time.

Is any of that actually true?

I ran four tests across 20 years of data to find out.

---

## Methodology

**Assets tested (2005–2025):**
- `GLD` — SPDR Gold Trust (best liquid proxy for spot gold)
- `SPY` — S&P 500 ETF (US stocks, total return)
- `TLT` — iShares 20+ Year Treasury (long bond)
- `IEF` — iShares 7–10 Year Treasury (medium bond)

**Four tests:**
1. **Long-run total returns** — nominal and inflation-adjusted (CPI-deflated)
2. **Named crisis periods** — 7 major market drawdowns, start-to-trough
3. **Bad-day behaviour** — what does gold do on SPY's worst days?
4. **Decile analysis** — gold's average return across all 10 SPY return buckets

All price data via `yfinance` (auto-adjusted, total return). Inflation adjustment using BLS CPI-U year-end values, linearly interpolated daily.

---

## Test 1: Long-Run Total Returns (2005–2025)

| Asset | Nominal CAGR | Real CAGR | Annual Vol | Sharpe | Max Drawdown |
|-------|-------------|-----------|------------|--------|--------------|
| S&P 500 (SPY) | {spy['nominal_cagr']:.2f}% | {spy['real_cagr']:.2f}% | {spy['annual_vol']:.1f}% | {spy['sharpe']:.2f} | {spy['max_drawdown']:.1f}% |
| Gold (GLD)    | {gold['nominal_cagr']:.2f}% | {gold['real_cagr']:.2f}% | {gold['annual_vol']:.1f}% | {gold['sharpe']:.2f} | {gold['max_drawdown']:.1f}% |
| 20Y Treasury (TLT) | {tlt['nominal_cagr']:.2f}% | {tlt['real_cagr']:.2f}% | {tlt['annual_vol']:.1f}% | {tlt['sharpe']:.2f} | {tlt['max_drawdown']:.1f}% |
| 7–10Y Treasury (IEF) | {ief['nominal_cagr']:.2f}% | {ief['real_cagr']:.2f}% | {ief['annual_vol']:.1f}% | {ief['sharpe']:.2f} | {ief['max_drawdown']:.1f}% |

Avg annual inflation over the period: **{inflation:.2f}%**

**Verdict:** {"Gold beat bonds on a real return basis" if gold['real_cagr'] > max(tlt['real_cagr'], ief['real_cagr']) else "Stocks dominated on real returns"}. {"Gold's inflation-adjusted CAGR was positive — it did preserve purchasing power." if gold['real_cagr'] > 0 else "Gold's real CAGR was negative — it lost purchasing power in inflation-adjusted terms."}

---

## Test 2: Named Crisis Periods

How did each asset perform from the start of each major SPY drawdown to its trough?

| Crisis | SPY | Gold | TLT |
|--------|-----|------|-----|
{crisis_table}

**Gold outperformed SPY in {gold_beats_spy} of {total_crises} crisis periods.**

Best gold crisis: **{best_crisis_name}** — Gold: {best_crisis_gold:+.1f}%, SPY: {best_crisis_spy:+.1f}%

Worst gold crisis: **{worst_crisis_name}** — Gold: {worst_crisis_gold:+.1f}%, SPY: {worst_crisis_spy:+.1f}%

---

## Test 3: Gold on SPY's Worst Days (daily return < −2%)

When SPY has a genuinely bad day, what does gold do?

| Asset | Avg Return | % of days positive | Significant? |
|-------|-----------|-------------------|--------------|
| SPY   | {spy_dump['avg_return']:+.3f}% | {spy_dump['pct_positive']:.0f}% | — |
| Gold  | {gold_dump['avg_return']:+.3f}% | {gold_dump['pct_positive']:.0f}% | {dump_sig} |
| TLT   | {tlt_dump['avg_return']:+.3f}% | {tlt_dump['pct_positive']:.0f}% | — |

(n = {int(gold_dump['n_days'])} days where SPY fell more than 2% in a single session)

---

## What this actually means

**The honest answer is: it depends on the crisis.**

Gold is a genuine safe haven in *some* regimes:
- **Inflation/dollar crises** → gold tends to shine (2007–2012 supercycle)
- **Geopolitical uncertainty, long grinds** → gold holds value
- **Liquidity crises** → gold often gets sold too (COVID crash, March 2020 was brutal for everything)

Gold is NOT a safe haven in:
- **Rapid "sell everything" panics** — when funds need to raise cash, they sell liquid winners (including gold)
- **High-rate environments** — gold pays no yield, so rising real rates hit it hard (2022 was rough for gold)
- **Long-run wealth accumulation** — stocks beat gold handily on a 20-year real-return basis

**The person asking this question had the right instinct:** Gold might only be a true safe haven in systemic crises *bigger than what we normally experience* — hyperinflation, dollar collapse, wartime. For run-of-the-mill recessions and rate cycles, the data is messier.

The most consistent safe haven in the data? **Long treasuries (TLT)** — at least until 2022 when the Fed destroyed them. Post-2022, even that correlation broke down. Nothing is perfectly safe.

---

## What should I test next?

- Gold vs Bitcoin in crisis periods — is BTC the "digital gold" it claims to be, or does it just crash harder?
- Does gold's safe-haven premium vary by VIX regime? Does it only kick in above VIX 30?
- Gold miners (GDX) vs physical gold — do miners lever the safe-haven effect or destroy it?
- Rolling 10-year Sharpe comparison: which decade did gold "win"?

Drop ideas below.

---

Full methodology and prior analysis: [github.com/jsabazova/dwp2-luxury-sentiment](https://github.com/jsabazova/dwp2-luxury-sentiment)

---

*Historical data analysis for educational purposes. Not financial advice. Past performance does not guarantee future results.*
"""

    out = RESULTS_DIR / "gold_reddit_post.md"
    out.write_text(post)
    print(f"Reddit post saved: {out}")
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Data
    prices = download_prices()
    returns = compute_returns(prices)
    cpi = build_cpi_series(prices.index)

    # Analyses
    stats_df, inflation = long_term_stats(prices, cpi)
    crisis_df = crisis_returns(prices)
    dump_df   = big_dump_stats(returns, threshold=-0.02)
    bad_df    = bad_day_analysis(returns)

    # Print
    print_long_term_table(stats_df, inflation)
    print_crisis_table(crisis_df)
    print_bad_day_table(dump_df, threshold=-0.02)

    # Save CSVs
    stats_df.to_csv(RESULTS_DIR / "gold_summary.csv", index=False)
    crisis_df.to_csv(RESULTS_DIR / "gold_crisis_returns.csv", index=False)
    dump_df.to_csv(RESULTS_DIR / "gold_bad_day_stats.csv", index=False)
    print(f"\nCSVs saved to {RESULTS_DIR}/")

    # Chart
    make_chart(prices, cpi, crisis_df, dump_df, returns, stats_df)

    # Reddit post
    write_reddit_post(stats_df, inflation, crisis_df, dump_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
