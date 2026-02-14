import os
import math
import time
from io import StringIO
from typing import List, Dict, Optional

import requests
import pandas as pd
import yfinance as yf

# =======================
# SECRETS / ENV
# =======================
PUSHOVER_USER_KEY = os.environ.get("PUSHOVER_USER_KEY")
PUSHOVER_APP_TOKEN = os.environ.get("PUSHOVER_APP_TOKEN")

TOP_N = int(os.environ.get("TOP_N", "5"))
MAX_TICKERS = int(os.environ.get("MAX_TICKERS", "503"))  # set 200 if you want less load
INFO_RETRIES = int(os.environ.get("INFO_RETRIES", "2"))  # retries per ticker for yfinance info

# =======================
# "GRAHAM MODERN" RULES (practical)
# =======================
RULES = {
    "pe_max": 20.0,
    "pb_max": 2.5,
    "pe_pb_max": 35.0,
    "earnings_yield_min": 0.05,     # 5%
    "debt_to_equity_max": 0.75,
    "current_ratio_min": 1.5,
    "roe_min": 0.10,                # 10%
    "market_cap_min": 5e9,          # 5B
}

WEIGHTS = {
    "earnings_yield": 0.50,
    "low_pb": 0.20,
    "low_de": 0.15,
    "high_current_ratio": 0.10,
    "roe": 0.05,
}

# =======================
# Ticker sources (try in order)
# =======================
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
GITHUB_DATASET_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
FALLBACK_FILE = "sp500_tickers_fallback.txt"


def must_env():
    missing = []
    if not PUSHOVER_USER_KEY:
        missing.append("PUSHOVER_USER_KEY")
    if not PUSHOVER_APP_TOKEN:
        missing.append("PUSHOVER_APP_TOKEN")
    if missing:
        raise RuntimeError(f"Missing GitHub Secrets: {', '.join(missing)}")


def pushover_push(title: str, message: str):
    title = (title or "").strip()[:100]
    message = (message or "").strip()[:1024]

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


def normalize_symbol(sym: str) -> str:
    sym = (sym or "").strip().upper()
    sym = sym.replace(".", "-")  # BRK.B -> BRK-B
    return sym


def load_tickers_from_wikipedia() -> Optional[List[str]]:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; value-agent/1.0; +https://github.com/)",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "close",
        }
        resp = requests.get(WIKI_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        df = None
        for t in tables:
            if "Symbol" in t.columns:
                df = t
                break
        if df is None:
            return None
        syms = [normalize_symbol(s) for s in df["Symbol"].astype(str).tolist()]
        syms = [s for s in syms if s]
        return dedupe_preserve_order(syms)[:MAX_TICKERS]
    except Exception as e:
        print("Wikipedia tickers failed:", e.__class__.__name__)
        return None


def load_tickers_from_github_dataset() -> Optional[List[str]]:
    try:
        headers = {"User-Agent": "value-agent/1.0"}
        resp = requests.get(GITHUB_DATASET_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        if "Symbol" not in df.columns:
            return None
        syms = [normalize_symbol(s) for s in df["Symbol"].astype(str).tolist()]
        syms = [s for s in syms if s]
        return dedupe_preserve_order(syms)[:MAX_TICKERS]
    except Exception as e:
        print("GitHub dataset tickers failed:", e.__class__.__name__)
        return None


def load_tickers_from_fallback_file() -> List[str]:
    try:
        with open(FALLBACK_FILE, "r", encoding="utf-8") as f:
            lines = [normalize_symbol(x) for x in f.read().splitlines()]
        lines = [x for x in lines if x]
        return dedupe_preserve_order(lines)[:MAX_TICKERS]
    except FileNotFoundError:
        return []


def get_sp500_tickers() -> List[str]:
    # Try multiple sources; always return something if fallback exists
    for fn in (load_tickers_from_wikipedia, load_tickers_from_github_dataset):
        tickers = fn()
        if tickers:
            return tickers

    fallback = load_tickers_from_fallback_file()
    if fallback:
        return fallback

    raise RuntimeError(
        "Could not load S&P 500 tickers from web sources, and fallback file is missing/empty. "
        "Create sp500_tickers_fallback.txt with one ticker per line."
    )


def dedupe_preserve_order(items: List[str]) -> List[str]:
    out, seen = [], set()
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def fetch_info_with_retries(t: yf.Ticker, retries: int) -> Dict:
    last_exc = None
    for i in range(retries + 1):
        try:
            info = t.info  # may rate-limit / be incomplete
            if isinstance(info, dict) and info:
                return info
        except Exception as e:
            last_exc = e
            # small backoff
            time.sleep(0.6 * (i + 1))
    raise last_exc or RuntimeError("Unknown yfinance info error")


def compute_metrics(ticker: str, info: Dict) -> Dict:
    pe = safe_float(info.get("trailingPE") or info.get("forwardPE"))
    pb = safe_float(info.get("priceToBook"))
    mcap = safe_float(info.get("marketCap"))
    de = safe_float(info.get("debtToEquity"))
    cr = safe_float(info.get("currentRatio"))
    roe = safe_float(info.get("returnOnEquity"))
    price = safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))

    # Normalize debtToEquity if percent-like (e.g., 120 -> 1.2)
    if de is not None and de > 10:
        de = de / 100.0

    earnings_yield = (1.0 / pe) if (pe and pe > 0) else None

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
    }


def passes_rules(r: Dict) -> bool:
    pe = r["pe"]
    pb = r["pb"]
    ey = r["earnings_yield"]
    de = r["debt_to_equity"]
    cr = r["current_ratio"]
    roe = r["roe"]
    mcap = r["market_cap"]

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
    return True


def score(r: Dict) -> float:
    ey = r["earnings_yield"] or 0.0
    roe = r["roe"] or 0.0

    pb = r["pb"]
    de = r["debt_to_equity"]
    cr = r["current_ratio"] or 0.0

    low_pb = (1.0 / pb) if (pb and pb > 0) else 0.0
    low_de = (1.0 / (1.0 + de)) if (de is not None and de >= 0) else 0.0
    high_cr = min(cr / 3.0, 1.0)

    return (
        WEIGHTS["earnings_yield"] * ey
        + WEIGHTS["low_pb"] * low_pb
        + WEIGHTS["low_de"] * low_de
        + WEIGHTS["high_current_ratio"] * high_cr
        + WEIGHTS["roe"] * roe
    )


def fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x*100:.1f}%"


def fmt_num(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.2f}"


def fmt_mcap(x: Optional[float]) -> str:
    if x is None:
        return "—"
    if x >= 1e12:
        return f"{x/1e12:.2f}T"
    if x >= 1e9:
        return f"{x/1e9:.2f}B"
    return f"{x/1e6:.0f}M"


def main():
    must_env()

    tickers = get_sp500_tickers()
    print(f"Loaded {len(tickers)} tickers")

    # Use yf.Tickers to reduce overhead
    bundle = yf.Tickers(" ".join(tickers))

    ok_rows = []
    info_errors = 0
    missing_data = 0

    for tk in tickers:
        try:
            t = bundle.tickers.get(tk) or yf.Ticker(tk)
            info = fetch_info_with_retries(t, INFO_RETRIES)
            r = compute_metrics(tk, info)

            # Require core fields for the screen; count missing
            core_needed = ["pe", "pb", "market_cap", "debt_to_equity", "current_ratio", "roe", "earnings_yield"]
            if any(r.get(k) is None for k in core_needed):
                missing_data += 1
                continue

            if passes_rules(r):
                r["score"] = score(r)
                ok_rows.append(r)

        except Exception as e:
            info_errors += 1
            print(f"{tk} info error: {e.__class__.__name__}")

        # small pacing to be kind to the upstream
        time.sleep(0.05)

    if not ok_rows:
        msg = (
            "No candidates today (Free yfinance).\n"
            f"Tickers loaded: {len(tickers)}\n"
            f"Info errors: {info_errors}\n"
            f"Missing-data skipped: {missing_data}\n"
            "Tip: set MAX_TICKERS=200 to reduce rate limits."
        )
        pushover_push("Daily Value Top 5 (Free)", msg)
        print(msg)
        return

    df = pd.DataFrame(ok_rows).sort_values("score", ascending=False).head(TOP_N)

    lines = []
    for _, r in df.iterrows():
        pe = r["pe"]; pb = r["pb"]
        lines.append(
            f"{r['ticker']}: EY {fmt_pct(r['earnings_yield'])} | "
            f"P/E {fmt_num(pe)} | P/B {fmt_num(pb)} | PExPB {fmt_num(pe*pb)} | "
            f"ROE {fmt_pct(r['roe'])} | D/E {fmt_num(r['debt_to_equity'])} | "
            f"CR {fmt_num(r['current_ratio'])} | MCap {fmt_mcap(r['market_cap'])}"
        )

    message = "Top 5 — Graham Modern (Free)\n" + "\n".join(lines)
    footer = f"\nLoaded:{len(tickers)} | InfoErr:{info_errors} | SkippedMissing:{missing_data}"
    if len(message) + len(footer) <= 1024:
        message += footer

    pushover_push("Daily Value Top 5 (Free)", message)
    print(message)


if __name__ == "__main__":
    main()
