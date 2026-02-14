import os
import requests
import pandas as pd

# =======================
# ENV / SECRETS (GitHub)
# =======================
FMP_API_KEY = os.environ.get("FMP_API_KEY")
PUSHOVER_USER_KEY = os.environ.get("PUSHOVER_USER_KEY")
PUSHOVER_APP_TOKEN = os.environ.get("PUSHOVER_APP_TOKEN")

# Optional: push formatting
PUSH_TITLE = os.environ.get("PUSH_TITLE", "Daily Value Top 5 (Graham Modern)")
PUSH_PRIORITY = os.environ.get("PUSH_PRIORITY", None)  # e.g. "1"
PUSH_SOUND = os.environ.get("PUSH_SOUND", None)        # e.g. "cashregister"

FMP_BASE = "https://financialmodelingprep.com/stable"  # stable endpoints


# =======================
# GRAHAM MODERN RULES
# =======================
RULES = {
    "pe_max": 20.0,
    "pb_max": 2.5,
    "pe_pb_max": 35.0,          # modernized "22.5"
    "earnings_yield_min": 0.05, # 5%
    "debt_to_equity_max": 0.75,
    "current_ratio_min": 1.5,
    "roe_min": 0.10,            # 10%
    "market_cap_min": 5e9,      # 5B
}

# Score weights (tweakable)
WEIGHTS = {
    "earnings_yield": 0.45,
    "low_pb": 0.20,
    "low_de": 0.15,
    "high_current_ratio": 0.10,
    "roe": 0.10,
}

TOP_N = 5


def require_env():
    missing = []
    if not FMP_API_KEY:
        missing.append("FMP_API_KEY")
    if not PUSHOVER_USER_KEY:
        missing.append("PUSHOVER_USER_KEY")
    if not PUSHOVER_APP_TOKEN:
        missing.append("PUSHOVER_APP_TOKEN")
    if missing:
        raise RuntimeError(f"Missing env vars/secrets: {', '.join(missing)}")


def fmp_get(path: str, params=None):
    if params is None:
        params = {}
    params = {**params, "apikey": FMP_API_KEY}
    url = f"{FMP_BASE}{path}"
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def pushover_push(title: str, message: str):
    title = (title or "").strip()[:100]
    message = (message or "").strip()[:1024]  # Pushover limit safety

    data = {
        "token": PUSHOVER_APP_TOKEN.strip(),
        "user": PUSHOVER_USER_KEY.strip(),
        "title": title,
        "message": message,
    }
    if PUSH_PRIORITY is not None:
        data["priority"] = str(PUSH_PRIORITY).strip()
    if PUSH_SOUND is not None:
        data["sound"] = str(PUSH_SOUND).strip()

    r = requests.post(
        "https://api.pushover.net/1/messages.json",
        data=data,
        timeout=30,
    )

    # Helpful debug if something goes wrong in GitHub Actions logs
    if r.status_code != 200:
        print("Pushover HTTP:", r.status_code)
        print("Pushover body:", r.text)

    r.raise_for_status()


def to_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def load_sp500_symbols() -> set[str]:
    """
    FMP stable endpoint for S&P 500 constituents.
    """
    data = fmp_get("/sp500-constituent")
    syms = set()
    for row in data:
        s = row.get("symbol") or row.get("ticker") or row.get("Symbol")
        if s:
            # normalize BRK.B -> BRK-B
            syms.add(str(s).upper().replace(".", "-"))
    if not syms:
        raise RuntimeError("No S&P 500 symbols returned. Check FMP endpoint/plan.")
    return syms


def load_key_metrics_bulk() -> pd.DataFrame:
    """
    Bulk key metrics TTM dataset for many symbols.
    We'll filter to S&P 500 after loading.
    """
    data = fmp_get("/key-metrics-ttm-bulk")
    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError("No data from key-metrics-ttm-bulk. Check plan/limits.")
    return pd.DataFrame(data)


def pick_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resilient mapping: FMP field names can vary.
    Create standardized columns even if source keys differ.
    """
    def first_existing(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    symbol_col = first_existing("symbol", "Symbol", "ticker")
    if not symbol_col:
        raise RuntimeError("Bulk dataset missing symbol column.")

    colmap = {
        "symbol": symbol_col,
        "pe": first_existing("peRatioTTM", "peRatio", "pe"),
        "pb": first_existing("pbRatioTTM", "priceToBookRatioTTM", "pbRatio", "pb"),
        "earnings_yield": first_existing("earningsYieldTTM", "earningsYield"),
        "roe": first_existing("roeTTM", "returnOnEquityTTM", "roe"),
        "debt_to_equity": first_existing("debtToEquityTTM", "debtToEquity", "debtEquityRatioTTM", "debtEquityRatio"),
        "current_ratio": first_existing("currentRatioTTM", "currentRatio"),
        "market_cap": first_existing("marketCapTTM", "marketCap"),
    }

    out = pd.DataFrame()
    out["symbol"] = df[colmap["symbol"]].astype(str).str.upper().str.replace(".", "-", regex=False)

    for k, src in colmap.items():
        if k == "symbol":
            continue
        out[k] = df[src].map(to_float) if src else None

    return out


def passes_rules(r: pd.Series) -> bool:
    mcap = r["market_cap"]
    pe = r["pe"]
    pb = r["pb"]
    ey = r["earnings_yield"]
    de = r["debt_to_equity"]
    cr = r["current_ratio"]
    roe = r["roe"]

    if mcap is None or mcap < RULES["market_cap_min"]:
        return False

    if pe is None or pe <= 0 or pe > RULES["pe_max"]:
        return False

    if pb is None or pb <= 0 or pb > RULES["pb_max"]:
        return False

    # Graham-style combined valuation constraint (modernized)
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


def score(r: pd.Series) -> float:
    ey = r["earnings_yield"] or 0.0
    roe = r["roe"] or 0.0

    low_pb = (1.0 / r["pb"]) if (r["pb"] and r["pb"] > 0) else 0.0
    low_de = (1.0 / (1.0 + r["debt_to_equity"])) if (r["debt_to_equity"] is not None and r["debt_to_equity"] >= 0) else 0.0
    high_cr = min((r["current_ratio"] or 0.0) / 3.0, 1.0)  # cap at ~3

    return (
        WEIGHTS["earnings_yield"] * ey +
        WEIGHTS["low_pb"] * low_pb +
        WEIGHTS["low_de"] * low_de +
        WEIGHTS["high_current_ratio"] * high_cr +
        WEIGHTS["roe"] * roe
    )


def fmt_pct(x):
    return "—" if x is None else f"{x*100:.1f}%"


def fmt_num(x):
    return "—" if x is None else f"{x:.2f}"


def fmt_mcap(x):
    if x is None:
        return "—"
    if x >= 1e12:
        return f"{x/1e12:.2f}T"
    if x >= 1e9:
        return f"{x/1e9:.2f}B"
    if x >= 1e6:
        return f"{x/1e6:.2f}M"
    return f"{x:.0f}"


def main():
    require_env()

    sp500 = load_sp500_symbols()
    bulk = load_key_metrics_bulk()
    df = pick_cols(bulk)

    # filter to S&P 500
    df = df[df["symbol"].isin(sp500)].copy()

    if df.empty:
        raise RuntimeError("After filtering to S&P 500, no rows remain. Check symbol normalization.")

    df["pass"] = df.apply(passes_rules, axis=1)
    screened = df[df["pass"] == True].copy()

    if screened.empty:
        msg = (
            "No S&P 500 stocks passed today's Graham-Modern rules.\n"
            "Tip: relax thresholds (P/E, P/B, D/E, Current Ratio) if you want more hits."
        )
        pushover_push(PUSH_TITLE, msg)
        print(msg)
        return

    screened["score"] = screened.apply(score, axis=1)
    screened = screened.sort_values("score", ascending=False).head(TOP_N)

    lines = []
    for _, r in screened.iterrows():
        pe = r["pe"]
        pb = r["pb"]
        pe_pb = (pe * pb) if (pe is not None and pb is not None) else None

        lines.append(
            f"{r['symbol']}: EY {fmt_pct(r['earnings_yield'])} | "
            f"P/E {fmt_num(pe)} | P/B {fmt_num(pb)} | PExPB {fmt_num(pe_pb)} | "
            f"ROE {fmt_pct(r['roe'])} | D/E {fmt_num(r['debt_to_equity'])} | "
            f"CR {fmt_num(r['current_ratio'])} | MCap {fmt_mcap(r['market_cap'])}"
        )

    message = "Top 5 — Graham Modern (S&P 500)\n" + "\n".join(lines)
    pushover_push(PUSH_TITLE, message)
    print(message)


if __name__ == "__main__":
    main()
