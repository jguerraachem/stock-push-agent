import os
import math
import time
import requests
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

# =======================
# SECRETS / ENV
# =======================
PUSHOVER_USER_KEY = os.environ.get("PUSHOVER_USER_KEY")
PUSHOVER_APP_TOKEN = os.environ.get("PUSHOVER_APP_TOKEN")

# Optional knobs
TOP_N = int(os.environ.get("TOP_N", "5"))
MAX_TICKERS = int(os.environ.get("MAX_TICKERS", "503"))  # set 200 if you want it faster
THREADS = int(os.environ.get("THREADS", "10"))           # 8–16 usually ok

# If 1, require FCF yield to be present and >= min. If 0, treat missing FCF as neutral.
REQUIRE_FCF = os.environ.get("REQUIRE_FCF", "0") == "1"

# =======================
# GRAHAM MODERN RULES
# =======================
RULES = {
    "pe_max": 20.0,
    "pb_max": 2.5,
    "pe_pb_max": 35.0,           # modernized 22.5
    "earnings_yield_min": 0.05,  # 5%
    "debt_to_equity_max": 0.75,
    "current_ratio_min": 1.5,
    "roe_min": 0.10,             # 10%
    "market_cap_min": 5e9,       # 5B
    "fcf_yield_min": 0.05,       # 5% (if REQUIRE_FCF=1)
}

# Score weights (ranking)
WEIGHTS = {
    "earnings_yield": 0.45,
    "fcf_yield": 0.25,           # if missing, becomes 0 unless REQUIRE_FCF=0 (we keep 0)
    "low_pb": 0.15,
    "low_de": 0.10,
    "roe": 0.05,
}

# Wikipedia page for S&P 500 constituents
WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def must_env():
    missing = []
    if not PUSHOVER_USER_KEY:
        missing.append("PUSHOVER_USER_KEY")
    if not PUSHOVER_APP_TOKEN:
        missing.append("PUSHOVER_APP_TOKEN")
    if missing:
        raise RuntimeError(f"Missing secrets: {', '.join(missing)}")


def pushover_push(title: str, message: str):
    title = (title or "").strip()[:100]
    message = (message or "").strip()[:1024]  # pushover limit

    r = requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": PUSHOVER_APP_TOKEN.strip(),
            "user": PUSHOVER_USER_KEY.strip(),
            "title": title,
            "message": message,
        },
        timeout=30,
    )

    if r.status_code != 200:
        print("Pushover HTTP:", r.status_code)
        print("Pushover body:", r.text)

    r.raise_for_status()


def get_sp500_symbols() -> list[str]:
    # Read constituents table from Wikipedia
    tables = pd.read_html(WIKI_SP500)
    if not tables:
        raise RuntimeError("Could not read Wikipedia tables for S&P 500.")

    # The first table is typically the constituents with 'Symbol'
    df = tables[0]
    if "Symbol" not in df.columns:
        # Try to find the table that contains Symbol
        found = None
        for t in tables:
            if "Symbol" in t.columns:
                found = t
                break
        if found is None:
            raise RuntimeError("Could not find 'Symbol' column in Wikipedia tables.")
        df = found

    symbols = df["Symbol"].astype(str).str.upper().tolist()

    # Normalize for yfinance: BRK.B -> BRK-B, BF.B -> BF-B, etc.
    symbols = [s.replace(".", "-") for s in symbols]

    # Remove empties, dedupe, preserve order
    out = []
    seen = set()
    for s in symbols:
        s = s.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)

    return out[:MAX_TICKERS]


def safe_float(x):
    try:
        if x is None:
            return None
        # yfinance sometimes returns numpy types
        return float(x)
    except Exception:
        return None


def fetch_metrics(ticker: str) -> dict:
    """
    Uses yfinance .info. This is free but not perfect; missing data is common.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        return {"ticker": ticker, "error": f"yfinance_error: {e.__class__.__name__}"}

    pe = safe_float(info.get("trailingPE") or info.get("forwardPE"))
    pb = safe_float(info.get("priceToBook"))
    mcap = safe_float(info.get("marketCap"))
    de = safe_float(info.get("debtToEquity"))  # often in % terms? yfinance typically returns numeric ratio or percent-like
    cr = safe_float(info.get("currentRatio"))
    roe = safe_float(info.get("returnOnEquity"))  # often decimal (0.15 = 15%)
    fcf = safe_float(info.get("freeCashflow"))     # absolute dollars
    price = safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))

    # Normalize debtToEquity: yfinance sometimes uses "percent" style (e.g., 120.0 means 1.20)
    # Heuristic: if > 10, assume it's percent and divide by 100.
    if de is not None and de > 10:
        de = de / 100.0

    earnings_yield = (1.0 / pe) if (pe is not None and pe > 0) else None
    fcf_yield = (fcf / mcap) if (fcf is not None and mcap is not None and mcap > 0) else None

    return {
        "ticker": ticker,
        "price": price,
        "market_cap": mcap,
        "pe": pe,
        "pb": pb,
        "debt_to_equity": de,
        "current_ratio": cr,
        "roe": roe,
        "earnings_yield": earnings_yield,
        "fcf_yield": fcf_yield,
    }


def passes_rules(r: dict) -> bool:
    mcap = r["market_cap"]
    pe = r["pe"]
    pb = r["pb"]
    ey = r["earnings_yield"]
    de = r["debt_to_equity"]
    cr = r["current_ratio"]
    roe = r["roe"]
    fy = r["fcf_yield"]

    if mcap is None or mcap < RULES["market_cap_min"]:
        return False

    if pe is None or pe <= 0 or pe > RULES["pe_max"]:
        return False

    if pb is None or pb <= 0 or pb > RULES["pb_max"]:
        return False

    if (pe * pb) > RULES["pe_pb_max"]:
        return False

    if ey is None or ey < RULES["earnings_yield_min"]:
        return False

    if de is None or de > RULES["debt_to_equity_max"]:
        return False

    if cr is None or cr < RULES["current_ratio_min"]:
        return False

    if roe is None or roe < RULES["roe_min"]:
        return False

    if REQUIRE_FCF:
        if fy is None or fy < RULES["fcf_yield_min"]:
            return False

    return True


def score(r: dict) -> float:
    ey = r["earnings_yield"] or 0.0
    fy = r["fcf_yield"] or 0.0
    roe = r["roe"] or 0.0

    pb = r["pb"]
    de = r["debt_to_equity"]

    low_pb = (1.0 / pb) if (pb is not None and pb > 0) else 0.0
    low_de = (1.0 / (1.0 + de)) if (de is not None and de >= 0) else 0.0

    return (
        WEIGHTS["earnings_yield"] * ey +
        WEIGHTS["fcf_yield"] * fy +
        WEIGHTS["low_pb"] * low_pb +
        WEIGHTS["low_de"] * low_de +
        WEIGHTS["roe"] * roe
    )


def fmt_pct(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x*100:.1f}%"


def fmt_num(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.2f}"


def fmt_mcap(x):
    if x is None:
        return "—"
    if x >= 1e12:
        return f"{x/1e12:.2f}T"
    if x >= 1e9:
        return f"{x/1e9:.2f}B"
    return f"{x/1e6:.0f}M"


def main():
    must_env()

    tickers = get_sp500_symbols()
    print(f"Loaded {len(tickers)} S&P 500 tickers from Wikipedia")

    rows = []
    errors = 0

    # Parallel fetch to speed up in GitHub Actions
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futures = {ex.submit(fetch_metrics, t): t for t in tickers}
        for fut in as_completed(futures):
            r = fut.result()
            if "error" in r:
                errors += 1
            rows.append(r)

    df = pd.DataFrame(rows)

    # Save artifacts (optional)
    df.to_csv("sp500_value_raw.csv", index=False)

    # Apply rules
    df["pass"] = df.apply(lambda x: passes_rules(x.to_dict()), axis=1)
    passed = df[df["pass"] == True].copy()

    if passed.empty:
        msg = (
            "No S&P 500 stocks passed Graham-Modern rules today.\n"
            f"(REQUIRE_FCF={int(REQUIRE_FCF)}) Consider relaxing: P/E, P/B, D/E, Current Ratio."
        )
        pushover_push("Daily Value Top 5 (Free)", msg)
        print(msg)
        return

    passed["score"] = passed.apply(lambda x: score(x.to_dict()), axis=1)
    passed = passed.sort_values("score", ascending=False).head(TOP_N)

    lines = []
    for _, r in passed.iterrows():
        pe = r.get("pe")
        pb = r.get("pb")
        pe_pb = (pe * pb) if (pe and pb) else None

        lines.append(
            f"{r['ticker']}: EY {fmt_pct(r.get('earnings_yield'))} | "
            f"P/E {fmt_num(pe)} | P/B {fmt_num(pb)} | PExPB {fmt_num(pe_pb)} | "
            f"FCF {fmt_pct(r.get('fcf_yield'))} | ROE {fmt_pct(r.get('roe'))} | "
            f"D/E {fmt_num(r.get('debt_to_equity'))} | CR {fmt_num(r.get('current_ratio'))} | "
            f"MCap {fmt_mcap(r.get('market_cap'))}"
        )

    header = f"Top {TOP_N} — Graham Modern (Free)"
    footer = f"\nErrors: {errors}/{len(tickers)} (missing data is normal on free sources)"
    message = header + "\n" + "\n".join(lines)
    # stay under 1024 chars
    if len(message) + len(footer) <= 1024:
        message += footer

    pushover_push("Daily Value Top 5 (Free)", message)
    print(message)


if __name__ == "__main__":
    main()
