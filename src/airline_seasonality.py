"""
Airline Pre-Summer Seasonality Event Study
===========================================
Research question: Do US airline stocks show abnormal positive returns
in the 20 trading days BEFORE June 1 each year?

Methodology: CAPM market model event study (same as dwp2-luxury-sentiment)
  - Estimation window: [-80, -21] trading days before June 1
  - Event window:      [-20,  -1] trading days before June 1
  - Benchmark: S&P 500 (^GSPC)
  - Log returns: ln(P_t / P_{t-1})
  - OLS regression on estimation window → compute expected returns
  - CAR = sum of abnormal returns over event window
  - t-test across years: H0: mean CAR = 0

Years: 2015–2025, excluding 2020 (COVID)
Tickers: DAL, UAL, LUV, AAL, JBLU
"""

from __future__ import annotations

import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import yfinance as yf
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────

TICKERS = {
    "DAL": "Delta Air Lines",
    "UAL": "United Airlines",
    "LUV": "Southwest Airlines",
    "AAL": "American Airlines",
    "JBLU": "JetBlue",
}
BENCHMARK = "^GSPC"
YEARS = [y for y in range(2015, 2026) if y != 2020]
ESTIMATION_START = -80   # trading days before June 1
ESTIMATION_END   = -21   # trading days before June 1
EVENT_START      = -20   # trading days before June 1
EVENT_END        = -1    # trading days before June 1
RESULTS_DIR = Path(__file__).parent.parent / "results"


# ── Data download ────────────────────────────────────────────────────────────

def download_data() -> pd.DataFrame:
    """Download price data for all tickers + benchmark from 2014-01-01."""
    print("Downloading price data from yfinance...")
    all_tickers = list(TICKERS.keys()) + [BENCHMARK]
    raw = yf.download(all_tickers, start="2014-01-01", end="2025-12-31",
                      auto_adjust=True, progress=False)
    # Extract closing prices
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
    prices = prices.dropna(how="all")
    print(f"  Downloaded {len(prices)} trading days, {len(prices.columns)} tickers")
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns: ln(P_t / P_{t-1})."""
    returns = np.log(prices / prices.shift(1))
    return returns.dropna(how="all")


# ── Window helpers ───────────────────────────────────────────────────────────

def get_june1(year: int) -> pd.Timestamp:
    """Return June 1 of the given year as a Timestamp."""
    return pd.Timestamp(f"{year}-06-01")


def trading_days_offset(trading_idx: pd.DatetimeIndex,
                         anchor: pd.Timestamp,
                         offset_start: int,
                         offset_end: int) -> pd.DatetimeIndex:
    """
    Return trading days in [anchor + offset_start, anchor + offset_end]
    where offsets are in trading days (negative = before anchor).
    """
    # Find the position of anchor or the next trading day after it
    pos_arr = trading_idx.searchsorted(anchor)
    # Clamp to valid range
    pos = min(pos_arr, len(trading_idx) - 1)

    start_pos = pos + offset_start
    end_pos   = pos + offset_end + 1  # +1 for inclusive slice

    start_pos = max(start_pos, 0)
    end_pos   = min(end_pos, len(trading_idx))

    window = trading_idx[start_pos:end_pos]
    if len(window) == 0:
        raise ValueError(f"Empty window for anchor={anchor}, offsets=[{offset_start}, {offset_end}]")
    return window


# ── Market model ─────────────────────────────────────────────────────────────

def fit_market_model(ticker_ret: pd.Series, bench_ret: pd.Series) -> tuple[float, float]:
    """
    OLS: ticker_return = alpha + beta * benchmark_return
    Returns (alpha, beta).
    """
    aligned = pd.concat([ticker_ret, bench_ret], axis=1).dropna()
    if len(aligned) < 10:
        raise ValueError(f"Too few observations: {len(aligned)}")
    X = aligned.iloc[:, 1].values.reshape(-1, 1)
    y = aligned.iloc[:, 0].values
    model = LinearRegression().fit(X, y)
    return float(model.intercept_), float(model.coef_[0])


# ── Core event study ─────────────────────────────────────────────────────────

def compute_car_for_year(returns: pd.DataFrame, ticker: str,
                          year: int) -> dict | None:
    """
    Compute CAR for one ticker × year event (pre-summer window).
    Returns dict with car, mean_ar, std_ar, alpha, beta, n_days, year, ticker.
    """
    june1 = get_june1(year)
    trading_idx = returns.index

    # Estimation window
    try:
        est_idx = trading_days_offset(trading_idx, june1, ESTIMATION_START, ESTIMATION_END)
    except ValueError as e:
        print(f"  SKIP {ticker} {year} estimation window: {e}")
        return None

    # Event window
    try:
        ev_idx = trading_days_offset(trading_idx, june1, EVENT_START, EVENT_END)
    except ValueError as e:
        print(f"  SKIP {ticker} {year} event window: {e}")
        return None

    ticker_est = returns.loc[est_idx, ticker].dropna()
    bench_est  = returns.loc[est_idx, BENCHMARK].dropna()
    ticker_ev  = returns.loc[ev_idx, ticker].dropna()
    bench_ev   = returns.loc[ev_idx, BENCHMARK].dropna()

    if len(ticker_est) < 20 or len(ticker_ev) < 5:
        print(f"  SKIP {ticker} {year}: insufficient data (est={len(ticker_est)}, ev={len(ticker_ev)})")
        return None

    try:
        alpha, beta = fit_market_model(ticker_est, bench_est)
    except ValueError as e:
        print(f"  SKIP {ticker} {year}: {e}")
        return None

    # Align event window
    aligned_ev = pd.concat([ticker_ev, bench_ev], axis=1).dropna()
    aligned_ev.columns = ["ticker", "bench"]

    expected = alpha + beta * aligned_ev["bench"]
    ar = aligned_ev["ticker"] - expected
    car = ar.sum()

    return {
        "ticker":   ticker,
        "year":     year,
        "car":      car,
        "mean_ar":  ar.mean(),
        "std_ar":   ar.std(),
        "n_days":   len(ar),
        "alpha":    alpha,
        "beta":     beta,
        "ar_series": ar.values,
    }


def run_event_study(returns: pd.DataFrame) -> pd.DataFrame:
    """Run event study for all tickers × years. Return DataFrame of CARs."""
    records = []
    for ticker in TICKERS:
        for year in YEARS:
            result = compute_car_for_year(returns, ticker, year)
            if result is not None:
                records.append(result)

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "ar_series"}
                        for r in records])
    return df, records


# ── Statistics ───────────────────────────────────────────────────────────────

def significance_stars(p: float) -> str:
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


def compute_ticker_summary(car_df: pd.DataFrame) -> pd.DataFrame:
    """One-sample t-test on CARs per ticker: H0 = mean CAR = 0."""
    rows = []
    for ticker, grp in car_df.groupby("ticker"):
        cars = grp["car"].values
        n = len(cars)
        mean_car = cars.mean()
        t_stat, p_val = stats.ttest_1samp(cars, 0)
        win_rate = (cars > 0).mean() * 100
        rows.append({
            "ticker":   ticker,
            "name":     TICKERS[ticker],
            "n_years":  n,
            "mean_car": mean_car,
            "mean_car_pct": mean_car * 100,
            "t_stat":   t_stat,
            "p_value":  p_val,
            "stars":    significance_stars(p_val),
            "win_rate": win_rate,
        })
    return pd.DataFrame(rows).sort_values("mean_car", ascending=False)


def compute_basket(car_df: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight basket: average CAR across all tickers per year."""
    basket = car_df.groupby("year")["car"].mean().reset_index()
    basket.columns = ["year", "car"]
    basket["ticker"] = "BASKET"
    return basket


def compute_basket_summary(basket_df: pd.DataFrame) -> dict:
    cars = basket_df["car"].values
    t_stat, p_val = stats.ttest_1samp(cars, 0)
    return {
        "ticker":       "BASKET",
        "name":         "Equal-Weight Basket",
        "n_years":      len(cars),
        "mean_car":     cars.mean(),
        "mean_car_pct": cars.mean() * 100,
        "t_stat":       t_stat,
        "p_value":      p_val,
        "stars":        significance_stars(p_val),
        "win_rate":     (cars > 0).mean() * 100,
    }


# ── Printing results ─────────────────────────────────────────────────────────

def print_summary_table(summary: pd.DataFrame, basket_summary: dict) -> None:
    print("\n" + "="*75)
    print("AIRLINE PRE-SUMMER SEASONALITY — EVENT STUDY RESULTS")
    print(f"Event window: {EVENT_START} to {EVENT_END} trading days before June 1")
    print(f"Years: {YEARS}")
    print("="*75)
    header = f"{'Ticker':<8} {'Name':<22} {'Mean CAR':>10} {'t-stat':>8} {'p-val':>8} {'Sig':>5} {'Win%':>7} {'N':>4}"
    print(header)
    print("-"*75)
    for _, row in summary.iterrows():
        print(f"{row['ticker']:<8} {row['name']:<22} {row['mean_car_pct']:>9.2f}% "
              f"{row['t_stat']:>8.3f} {row['p_value']:>8.4f} {row['stars']:>5} "
              f"{row['win_rate']:>6.0f}% {row['n_years']:>4.0f}")
    print("-"*75)
    b = basket_summary
    print(f"{'BASKET':<8} {'Equal-Weight Basket':<22} {b['mean_car_pct']:>9.2f}% "
          f"{b['t_stat']:>8.3f} {b['p_value']:>8.4f} {b['stars']:>5} "
          f"{b['win_rate']:>6.0f}% {b['n_years']:>4}")
    print("="*75)
    print("Significance: *** p<0.01  ** p<0.05  * p<0.10")


def print_yearly_breakdown(car_df: pd.DataFrame, basket_df: pd.DataFrame) -> None:
    print("\n" + "="*75)
    print("YEAR-BY-YEAR CAR BREAKDOWN (log return, e.g. 0.05 = +5%)")
    print("="*75)
    pivot = car_df.pivot(index="year", columns="ticker", values="car")
    pivot["BASKET"] = basket_df.set_index("year")["car"]
    pivot = pivot.sort_index()
    cols = list(TICKERS.keys()) + ["BASKET"]
    cols = [c for c in cols if c in pivot.columns]
    print(pivot[cols].applymap(lambda x: f"{x*100:+.2f}%").to_string())
    print("="*75)


# ── Visualisation ────────────────────────────────────────────────────────────

def make_chart(car_df: pd.DataFrame, basket_df: pd.DataFrame,
               summary: pd.DataFrame, basket_summary: dict) -> Path:
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
    })

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "US Airline Stocks: Pre-Summer Seasonality Study\n"
        "Cumulative Abnormal Returns in 20 Trading Days Before June 1 (2015–2025, ex-2020)",
        fontsize=15, fontweight="bold", y=0.98
    )

    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ── Panel 1: Basket CAR by year ──────────────────────────────────────────
    years = basket_df["year"].values
    cars_pct = basket_df["car"].values * 100
    mean_car_pct = basket_summary["mean_car_pct"]
    colors = ["#2ecc71" if c >= 0 else "#e74c3c" for c in cars_pct]
    bars = ax1.bar(years, cars_pct, color=colors, edgecolor="white", linewidth=0.8, width=0.7)
    ax1.axhline(mean_car_pct, color="#2c3e50", linestyle="--", linewidth=1.8,
                label=f"Mean = {mean_car_pct:.2f}%")
    ax1.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax1.set_title("Equal-Weight Basket CAR by Year")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("CAR (%)")
    ax1.legend(fontsize=9)
    ax1.set_xticks(years)
    ax1.set_xticklabels(years, rotation=45, ha="right", fontsize=9)
    for bar, val in zip(bars, cars_pct):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15 * np.sign(val),
                 f"{val:.1f}%", ha="center", va="bottom" if val >= 0 else "top",
                 fontsize=7.5, color="#2c3e50")

    # ── Panel 2: Mean CAR by airline (horizontal bar) ────────────────────────
    sorted_sum = summary.sort_values("mean_car_pct")
    colors2 = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sorted_sum["mean_car_pct"]]
    bars2 = ax2.barh(sorted_sum["ticker"], sorted_sum["mean_car_pct"],
                      color=colors2, edgecolor="white", linewidth=0.8)
    ax2.axvline(0, color="black", linewidth=0.8, alpha=0.4)
    ax2.set_title("Mean CAR by Airline")
    ax2.set_xlabel("Mean CAR (%)")
    # Add sig stars and values
    for bar, (_, row) in zip(bars2, sorted_sum.iterrows()):
        x_pos = bar.get_width()
        offset = 0.08 if x_pos >= 0 else -0.08
        label = f"{row['mean_car_pct']:.2f}% {row['stars']}"
        ax2.text(x_pos + offset, bar.get_y() + bar.get_height()/2,
                 label, va="center", ha="left" if x_pos >= 0 else "right",
                 fontsize=9, color="#2c3e50")

    # ── Panel 3: Box plot of CARs by ticker ──────────────────────────────────
    ticker_order = summary.sort_values("mean_car_pct", ascending=False)["ticker"].tolist()
    box_data = [car_df[car_df["ticker"] == t]["car"].values * 100 for t in ticker_order]
    bp = ax3.boxplot(box_data, labels=ticker_order, patch_artist=True,
                      medianprops=dict(color="#2c3e50", linewidth=2),
                      whiskerprops=dict(color="#7f8c8d"),
                      capprops=dict(color="#7f8c8d"),
                      flierprops=dict(marker="o", markerfacecolor="#e74c3c",
                                      markersize=5, alpha=0.7))
    for patch, color in zip(bp["boxes"], ["#3498db", "#9b59b6", "#e67e22", "#1abc9c", "#e74c3c"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.axhline(0, color="black", linewidth=0.8, alpha=0.4, linestyle="--")
    ax3.set_title("Distribution of Pre-Summer CARs by Airline")
    ax3.set_xlabel("Ticker")
    ax3.set_ylabel("CAR (%)")

    # ── Panel 4: Summary statistics table ───────────────────────────────────
    ax4.axis("off")
    table_data = []
    for _, row in summary.iterrows():
        table_data.append([
            row["ticker"],
            f"{row['mean_car_pct']:.2f}%{row['stars']}",
            f"{row['t_stat']:.3f}",
            f"{row['p_value']:.4f}",
            f"{row['win_rate']:.0f}%",
        ])
    # Add basket row
    b = basket_summary
    table_data.append([
        "BASKET",
        f"{b['mean_car_pct']:.2f}%{b['stars']}",
        f"{b['t_stat']:.3f}",
        f"{b['p_value']:.4f}",
        f"{b['win_rate']:.0f}%",
    ])

    col_labels = ["Ticker", "Mean CAR", "t-stat", "p-value", "Win%"]
    table = ax4.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0.15, 1, 0.75],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Colour-code mean CAR cells
    for i, (_, row) in enumerate(list(summary.iterrows()) + [("", pd.Series(b))]):
        val = float(str(table_data[i][1]).replace("%", "").replace("*", ""))
        colour = "#d5f5e3" if val >= 0 else "#fadbd8"
        table[i+1, 1].set_facecolor(colour)

    ax4.set_title("Summary Statistics\n(*** p<0.01  ** p<0.05  * p<0.10)",
                   fontsize=11, fontweight="bold")

    out_path = RESULTS_DIR / "airline_presummer_seasonality.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nChart saved: {out_path}")
    return out_path


# ── Reddit post writer ────────────────────────────────────────────────────────

def write_reddit_post(summary: pd.DataFrame, basket_summary: dict,
                       basket_df: pd.DataFrame, car_df: pd.DataFrame) -> Path:
    b = basket_summary
    best_ticker = summary.iloc[0]
    worst_ticker = summary.iloc[-1]

    # Find best and worst years for basket
    best_year = basket_df.loc[basket_df["car"].idxmax(), "year"]
    worst_year = basket_df.loc[basket_df["car"].idxmin(), "year"]
    best_year_car = basket_df["car"].max() * 100
    worst_year_car = basket_df["car"].min() * 100

    # Determine narrative based on results
    sig_tickers = summary[summary["p_value"] < 0.10]
    n_sig = len(sig_tickers)
    basket_sig = b["p_value"] < 0.10
    basket_positive = b["mean_car_pct"] > 0

    if basket_sig and basket_positive:
        verdict = (f"The basket showed a mean CAR of **{b['mean_car_pct']:.2f}%** "
                   f"(t={b['t_stat']:.3f}, p={b['p_value']:.4f}{b['stars']}) — "
                   f"statistically significant at the 10% level.")
        verdict_summary = "The pattern is real, if modest."
    elif basket_positive and not basket_sig:
        verdict = (f"The basket averaged **{b['mean_car_pct']:.2f}%** over the pre-summer window "
                   f"(t={b['t_stat']:.3f}, p={b['p_value']:.4f}). Positive, but not statistically significant. "
                   f"With only {b['n_years']} data points, this could easily be noise.")
        verdict_summary = "Interesting but inconclusive."
    else:
        verdict = (f"The basket averaged **{b['mean_car_pct']:.2f}%** over the pre-summer window "
                   f"(t={b['t_stat']:.3f}, p={b['p_value']:.4f}). The summer anticipation premium "
                   f"apparently doesn't show up in the data.")
        verdict_summary = "The hypothesis doesn't hold up."

    # Individual ticker highlights
    ticker_lines = []
    for _, row in summary.iterrows():
        star_str = f" {row['stars']}" if row["stars"] else ""
        ticker_lines.append(
            f"- **{row['ticker']} ({row['name']}):** {row['mean_car_pct']:.2f}%{star_str} "
            f"(t={row['t_stat']:.3f}, p={row['p_value']:.4f}, win rate: {row['win_rate']:.0f}%)"
        )

    post = f"""# I tested whether airline stocks actually rise before summer. 10 years of data. Here are the results.

---

**The question:** Do US airline stocks price in the summer travel rush *before* June 1 — showing abnormal positive returns in the 20 trading days leading up to summer?

The intuition seemed reasonable. Airlines report strong summer bookings well in advance. Analysts update models in April/May. Retail investors notice beach Instagram posts and buy Delta. Or at least, that's the narrative.

I ran the numbers to find out.

---

## Methodology

I used a standard **event study** — the same framework that academic finance has used since Fama, Fisher, Jensen & Roll (1969). The idea is simple: strip out what the stock "would have done anyway" based on how the market moved, and look at what's left over. That residual is the **abnormal return**.

**Setup:**
- **Tickers:** DAL (Delta), UAL (United), LUV (Southwest), AAL (American), JBLU (JetBlue)
- **Benchmark:** S&P 500 (^GSPC)
- **Event:** June 1 each year, {min(YEARS)}–{max(YEARS)}, excluding 2020 (COVID ruins everything)
- **Event window:** 20 trading days before June 1 (approximately mid-April to end of May)
- **Estimation window:** 60 trading days even earlier (used to fit the market model)

**Market model (CAPM baseline):**

For each ticker × year, I ran OLS regression on the estimation window:
```
stock_return = α + β × SP500_return
```

Then applied those fitted parameters to compute expected returns during the event window. Abnormal return = actual − expected. CAR = sum of abnormal returns across all 20 days.

Then ran a one-sample t-test across years: is the mean CAR significantly different from zero?

---

## Results

{verdict}

**By airline (mean CAR over 20-day pre-summer window):**

{chr(10).join(ticker_lines)}

**Basket (equal-weight average):**
- Mean CAR: **{b['mean_car_pct']:.2f}%**
- t-statistic: {b['t_stat']:.3f}
- p-value: {b['p_value']:.4f} {b['stars']}
- Win rate (positive CAR years): **{b['win_rate']:.0f}%**
- Years tested: {b['n_years']}

**Year-by-year extremes:**
- Best pre-summer run: {best_year} ({best_year_car:+.2f}% basket CAR)
- Worst pre-summer run: {worst_year} ({worst_year_car:+.2f}% basket CAR)

**{verdict_summary}**

---

## What might explain this?

A few honest thoughts:

1. **Analysts aren't naive.** If summer seasonality is real and predictable, sell-side desks have been pricing it in for decades. Predictable patterns get arbed away.

2. **Airlines are macro-sensitive.** Fuel prices, interest rates, and consumer confidence dominate airline stock movements far more than seasonal booking patterns. The CAPM model strips out the S&P 500 move, but sector-specific shocks still contaminate the signal.

3. **Small sample problem.** {b['n_years']} years (excluding 2020) is not a lot of data for a t-test. You need the effect to be large and consistent to clear the significance threshold. A few bad years (hello, 2022 inflation shock) drag down the average.

4. **The "wall of worry."** Some years the best airline period is actually *after* June — when actual summer traffic comes in better than feared. Pre-summer anticipation might be priced in earlier than 20 days out, or later.

---

## The chart

Saved locally — four panels: basket CAR by year, mean CAR by airline, box plot distribution, and summary table. [GitHub link below]

---

## What should I test next?

A few ideas I'm considering:

- Does the *post-earnings* window (after Q1 airline earnings in April) show stronger signals?
- What about comparing airline CARs vs hotel/cruise stocks pre-summer — is it sector-wide or airline-specific?
- Does the signal strengthen in years when jet fuel prices are falling vs rising?
- Rolling window: is the pattern getting weaker over time as markets get more efficient?

Drop your suggestions below.

---


---

*Historical data analysis for educational purposes. Not financial advice. Past performance does not guarantee future results.*
"""

    out_path = RESULTS_DIR / "reddit_post.md"
    out_path.write_text(post)
    print(f"Reddit post saved: {out_path}")
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data
    prices = download_data()
    returns = compute_log_returns(prices)
    print(f"Returns shape: {returns.shape}")
    print(f"Date range: {returns.index[0].date()} → {returns.index[-1].date()}")

    # 2. Event study
    print("\nRunning event study...")
    car_df, records = run_event_study(returns)
    print(f"CARs computed: {len(car_df)} ticker-year observations")

    # 3. Basket
    basket_df = compute_basket(car_df)
    basket_summary = compute_basket_summary(basket_df)

    # 4. Summary
    summary = compute_ticker_summary(car_df)

    # 5. Print results
    print_summary_table(summary, basket_summary)
    print_yearly_breakdown(car_df, basket_df)

    # 6. Save CSVs
    car_df.to_csv(RESULTS_DIR / "car_results.csv", index=False)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    basket_df.to_csv(RESULTS_DIR / "basket_by_year.csv", index=False)
    print(f"\nCSVs saved to {RESULTS_DIR}/")

    # 7. Chart
    make_chart(car_df, basket_df, summary, basket_summary)

    # 8. Reddit post
    write_reddit_post(summary, basket_summary, basket_df, car_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
