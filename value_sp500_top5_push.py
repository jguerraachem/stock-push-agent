import os
import math
import time
import random
from datetime import datetime
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
MAX_TICKERS = int(os.environ.get("MAX_TICKERS", "200"))
INFO_RETRIES = int(os.environ.get("INFO_RETRIES", "2"))

# Raise quality: default market-cap floor is now 10B
# You can still override via workflow env MARKET_CAP_MIN
MARKET_CAP_MIN = float(os.environ.get("MARKET_CAP_MIN", "10000000000"))  # 10B

# =======================
# "GRAHAM MODERN" RULES
# =======================
RULES = {
    # Hard valuation constraints
    "pe_max": 20.0,
    "pb_max": 2.5,
    "pe_pb_max": 35.0,
    "earnings_yield_min": 0.05,
    "market_cap_min": MARKET_CAP_MIN,

    # ROE floors (only enforced when ROE exists; missing ROE is allowed)
    "roe_floor_nonfin_if_present": 0.08,  # 8%
    "roe_floor_fin_if_present": 0.10,     # 10% for Financials
}

# Score weights (quality tilt)
WEIGHTS = {
    "earnings_yield": 0.50,
    "low_pb": 0.20,
    "low_de": 0.10,
    "high_current_ratio": 0.05,  # non-financials only
    "roe": 0.15,
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
    return sym.replace(".", "-")


def dedupe_preserve_order(items: List[str]) -> List[str]:
    out, seen = [], set()
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


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
        return dedupe_preserve_order(syms)
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
        return dedupe_preserve_order(syms)
    except Exception as e:
        print("GitHub dataset tickers failed:", e.__class__.__name__)
        return None


def load_tickers_from_fallback_file() -> List[str]:
    try:
        with open(FALLBACK_FILE, "r", encoding="utf-8") as f:
            lines = [normalize_symbol(x) for x in f.read().splitlines()]
        lines = [x for x in lines if x]
        return dedupe_preserve_order(lines)
    except FileNotFoundError:
        return []


def get_sp500_tickers() -> List[str]:
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
            info = t.info
            if isinstance(info, dict) and info:
                return info
        except Exception as e:
            last_exc = e
            time.sleep(0.6 * (i + 1))
    raise last_exc or RuntimeError("Unknown yfinance info error")


def is_financial(info: Dict) -> bool:
    sector = (info.get("sector") or "").lower()
    industry = (info.get("industry") or "").lower()
    if "financial" in sector:
        return True
    if any(k in industry for k in ("insurance", "bank", "capital markets", "asset management", "credit", "reinsurance")):
        return True
    return False


def compute_metrics(ticker: str, info: Dict) -> Dict:
    pe = safe_float(info.get("trailingPE") or info.get("forwardPE"))
    pb = safe_float(info.get("priceToBook"))
    book_value = safe_float(info.get("bookValue"))
    mcap = safe_float(info.get("marketCap"))
    de = safe_float(info.get("debtToEquity"))
    cr = safe_float(info.get("currentRatio"))
    roe = safe_float(info.get("returnOnEquity"))
    price = safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))
    sector = info.get("sector")
    industry = info.get("industry")

    # Normalize debtToEquity if percent-like
    if de is not None and de > 10:
        de = de / 100.0

    # P/B sanity + fallback
    if pb is None or pb < 0.1:
        if price is not None and book_value is not None and book_value > 0:
            pb = price / book_value
    if pb is not None and (pb < 0.1 or pb > 20):
        pb = None

    earnings_yield = (1.0 / pe) if (pe and pe > 0) else None
    fin = is_financial(info)

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
        "sector": sector,
        "industry": industry,
        "is_financial": fin,
    }


def passes_rules(r: Dict) -> bool:
    pe = r["pe"]
    pb = r["pb"]
    ey = r["earnings_yield"]
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
    if pe > 100:
        return False

    fin = bool(r.get("is_financial", False))

    # Guards if present
    roe = r.get("roe")
    if roe is not None:
        if roe < 0:
            return False
        floor = RULES["roe_floor_fin_if_present"] if fin else RULES["roe_floor_nonfin_if_present"]
        if roe < floor:
            return False

    de = r.get("debt_to_equity")
    if de is not None and de > 1.0:
        return False

    cr = r.get("current_ratio")
    if not fin and cr is not None and cr < 1.0:
        return False

    return True


def score(r: Dict) -> float:
    ey = r["earnings_yield"] or 0.0
    pb = r["pb"]
    de = r.get("debt_to_equity")
    cr = r.get("current_ratio")
    roe = r.get("roe")
    fin = bool(r.get("is_financial", False))

    low_pb = (1.0 / pb) if (pb and pb > 0) else 0.0

    if de is None:
        low_de = 0.50
    else:
        low_de = 1.0 / (1.0 + de)

    if roe is None:
        roe_score = 0.08
    else:
        roe_score = max(0.0, min(roe, 0.30))

    if fin:
        cr_score = 0.0
    else:
        cr_score = min(((cr or 0.0) / 3.0), 1.0)

    return (
        WEIGHTS["earnings_yield"] * ey
        + WEIGHTS["low_pb"] * low_pb
        + WEIGHTS["low_de"] * low_de
        + WEIGHTS["high_current_ratio"] * cr_score
        + WEIGHTS["roe"] * roe_score
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

    all_tickers = get_sp500_tickers()
    print(f"Loaded {len(all_tickers)} tickers (pre-sample)")

    random.seed(datetime.utcnow().strftime("%Y-%m-%d"))
    random.shuffle(all_tickers)
    tickers = all_tickers[:min(MAX_TICKERS, len(all_tickers))]
    print(f"Sampling {len(tickers)} tickers for today (MAX_TICKERS={MAX_TICKERS})")

    bundle = yf.Tickers(" ".join(tickers))

    ok_rows = []
    info_errors = 0
    missing_core = 0

    for tk in tickers:
        try:
            t = bundle.tickers.get(tk) or yf.Ticker(tk)
            info = fetch_info_with_retries(t, INFO_RETRIES)
            r = compute_metrics(tk, info)

            core_needed = ["pe", "pb", "market_cap", "earnings_yield"]
            if any(r.get(k) is None for k in core_needed):
                missing_core += 1
                continue

            if passes_rules(r):
                r["score"] = score(r)
                ok_rows.append(r)

        except Exception as e:
            info_errors += 1
            print(f"{tk} info error: {e.__class__.__name__}")

        time.sleep(0.05)

    if not ok_rows:
        msg = (
            "No candidates today (Free yfinance).\n"
            f"Tickers sampled: {len(tickers)}\n"
            f"Info errors: {info_errors}\n"
            f"Missing core skipped: {missing_core}\n"
            "Tip: reduce MAX_TICKERS or relax valuation thresholds."
        )
        pushover_push("Daily Value Top 5 (Free)", msg)
        print(msg)
        return

    df = pd.DataFrame(ok_rows).sort_values("score", ascending=False).head(TOP_N)

    lines = []
    for _, r in df.iterrows():
        pe = r["pe"]
        pb = r["pb"]
        fin_tag = "FIN" if r.get("is_financial") else "NON-FIN"
        sector = r.get("sector") or "—"
        cr_display = fmt_num(r.get("current_ratio")) if not r.get("is_financial") else "—"

        lines.append(
            f"{r['ticker']} [{fin_tag}] ({sector}): EY {fmt_pct(r['earnings_yield'])} | "
            f"P/E {fmt_num(pe)} | P/B {fmt_num(pb)} | PExPB {fmt_num(pe*pb)} | "
            f"ROE {fmt_pct(r.get('roe'))} | D/E {fmt_num(r.get('debt_to_equity'))} | "
            f"CR {cr_display} | MCap {fmt_mcap(r['market_cap'])}"
        )

    message = "Top 5 — Graham Modern (Free)\n" + "\n".join(lines)
    footer = (
        f"\nSampled:{len(tickers)} | InfoErr:{info_errors} | MissingCore:{missing_core} | "
        f"MCapMin:{fmt_mcap(RULES['market_cap_min'])} | ROE(FIN):10% | ROE(NON):8%"
    )
    if len(message) + len(footer) <= 1024:
        message += footer

    pushover_push("Daily Value Top 5 (Free)", message)
    print(message)


if __name__ == "__main__":
    main()
