import os
from datetime import datetime, timezone

import requests
import yfinance as yf
import pandas_market_calendars as mcal

# =======================
# CONFIG
# =======================
TICKERS = ["QQQM", "MAGS", "IBIT", "AMZN", "V", "GLDM", "VOO", "JEPQ", "O", "MCD", "VYM", "VTI", "ELV", "SMH"]

# Pushover env vars (set these as GitHub Actions Secrets)
PUSHOVER_USER_KEY = os.environ.get("PUSHOVER_USER_KEY")
PUSHOVER_APP_TOKEN = os.environ.get("PUSHOVER_APP_TOKEN")

# Optional controls
PUSH_PRIORITY = int(os.environ.get("PUSH_PRIORITY", "0"))  # 0 normal, 1 high
PUSH_SOUND = os.environ.get("PUSH_SOUND", "pushover")      # e.g. "pushover", "cashregister", "siren"
FORCE_PUSH = os.environ.get("FORCE_PUSH", "0") == "1"      # set to 1 to bypass market-hours check (testing)

# Market calendar (NYSE)
NYSE = mcal.get_calendar("NYSE")


def push(title: str, message: str):
    """
    Send one Pushover notification.
    Prints detailed error body in GitHub logs if something goes wrong.
    """
    if not PUSHOVER_USER_KEY or not PUSHOVER_APP_TOKEN:
        raise RuntimeError(
            "Missing PUSHOVER_USER_KEY and/or PUSHOVER_APP_TOKEN. "
            "Add them as GitHub Actions Secrets and pass via workflow env."
        )

    title = title.strip()[:100]
    message = message.strip()[:1024]  # Pushover message limit safety

    r = requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": PUSHOVER_APP_TOKEN.strip(),
            "user": PUSHOVER_USER_KEY.strip(),
            "title": title,
            "message": message,
            "priority": PUSH_PRIORITY,
            "sound": PUSH_SOUND,
        },
        timeout=20,
    )

    if r.status_code != 200:
        print("Pushover HTTP:", r.status_code)
        print("Pushover body:", r.text)

    r.raise_for_status()


def is_nyse_open_now(now_utc: datetime) -> bool:
    """
    True if NYSE is open at this moment.
    Uses official exchange calendar (handles DST + holidays).
    """
    day = now_utc.date()
    sched = NYSE.schedule(start_date=day, end_date=day)

    if sched.empty:
        return False

    market_open = sched.iloc[0]["market_open"].to_pydatetime()
    market_close = sched.iloc[0]["market_close"].to_pydatetime()

    return market_open <= now_utc <= market_close


def get_quote_line(ticker: str) -> str:
    """
    Returns: 'TICKER: 123.45 (+1.23%)'
    Uses fast_info when available; falls back to daily history.
    """
    t = yf.Ticker(ticker)
    fi = t.fast_info

    last = fi.get("last_price")
    prev_close = fi.get("previous_close")

    # Fallback if fast_info missing / flaky
    if last is None or prev_close is None:
        hist = t.history(period="5d", interval="1d")
        if hist.shape[0] < 2:
            return f"{ticker}: ERROR (not enough price history)"
        last = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2])

    last = float(last)
    prev_close = float(prev_close)

    pct_change = ((last / prev_close) - 1.0) * 100.0
    sign = "+" if pct_change >= 0 else ""
    return f"{ticker}: {last:,.2f} ({sign}{pct_change:.2f}%)"


def main():
    now_utc = datetime.now(timezone.utc)

    if not FORCE_PUSH and not is_nyse_open_now(now_utc):
        print("NYSE is closed now. No push sent.")
        return

    lines = []
    for tk in TICKERS:
        try:
            lines.append(get_quote_line(tk))
        except Exception as e:
            lines.append(f"{tk}: ERROR ({e.__class__.__name__})")

    message = "\n".join(lines)

    push(title="Watchlist: Price & % Day", message=message)
    print("Push sent:\n" + message)


if __name__ == "__main__":
    main()
