import os
import json
from pathlib import Path
import requests
import yfinance as yf

# =======================
# CONFIG (EDITA AQUÍ)
# =======================
WATCHLIST = [
    {"ticker": "QQQM", "push_every_run": True, "above": None, "below": None},
    {"ticker": "MAGS", "push_every_run": True, "above": None, "below": None},
    {"ticker": "IBIT", "push_every_run": True, "above": None, "below": None},
    {"ticker": "AMZN", "push_every_run": True, "above": None, "below": None},
    {"ticker": "V",    "push_every_run": True, "above": None, "below": None},
    {"ticker": "GLDM", "push_every_run": True, "above": None, "below": None},
    {"ticker": "VOO",  "push_every_run": True, "above": None, "below": None},
    {"ticker": "JEPQ", "push_every_run": True, "above": None, "below": None},
    {"ticker": "O",    "push_every_run": True, "above": None, "below": None},
    {"ticker": "MCD",  "push_every_run": True, "above": None, "below": None},
    {"ticker": "VYM",  "push_every_run": True, "above": None, "below": None},
    {"ticker": "VTI",  "push_every_run": True, "above": None, "below": None},
]

# Pushover keys (variables de entorno)
PUSHOVER_USER_KEY = os.environ.get("PUSHOVER_USER_KEY")
PUSHOVER_APP_TOKEN = os.environ.get("PUSHOVER_APP_TOKEN")

STATE_FILE = Path("alert_state.json")


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def push(title: str, message: str, priority: int = 0):
    if not PUSHOVER_USER_KEY or not PUSHOVER_APP_TOKEN:
        raise RuntimeError("Faltan PUSHOVER_USER_KEY y/o PUSHOVER_APP_TOKEN.")
    r = requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": PUSHOVER_APP_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "title": title,
            "message": message,
            "priority": priority,
        },
        timeout=20,
    )
    r.raise_for_status()


def get_quote(ticker: str):
    t = yf.Ticker(ticker)
    fi = t.fast_info

    last = fi.get("last_price")
    prev_close = fi.get("previous_close")

    if last is None or prev_close is None:
        hist = t.history(period="5d", interval="1d")
        last = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2])

    last = float(last)
    prev_close = float(prev_close)

    pct_change = ((last / prev_close) - 1.0) * 100.0
    return last, pct_change, prev_close


def crossed(prev: float, curr: float, level: float, direction: str) -> bool:
    # direction: "above" or "below"
    if direction == "above":
        return prev < level <= curr
    if direction == "below":
        return prev > level >= curr
    raise ValueError("direction must be 'above' or 'below'")


def check_once():
    state = load_state()

    for rule in WATCHLIST:
        ticker = rule["ticker"].upper()
        push_every_run = bool(rule.get("push_every_run", False))
        above = rule.get("above")
        below = rule.get("below")

        price, pct_change, prev_close = get_quote(ticker)

        # Estado por ticker
        if ticker not in state:
            state[ticker] = {
                "last_price_seen": None,
                "was_above": False,
                "was_below": False
            }

        last_seen = state[ticker]["last_price_seen"]
        should_push = False
        reason = ""

        if push_every_run:
            should_push = True
            reason = "scheduled update"
        else:
            # Solo push si cruza umbrales (anti-spam)
            if above is not None and last_seen is not None:
                if crossed(float(last_seen), price, float(above), "above"):
                    should_push = True
                    reason = f"crossed ABOVE {above}"
            if below is not None and last_seen is not None:
                if crossed(float(last_seen), price, float(below), "below"):
                    should_push = True
                    reason = f"crossed BELOW {below}"

        # Mensaje
        sign = "+" if pct_change >= 0 else ""
        title = f"{ticker} {price:,.2f}"
        msg = (
            f"Price: {price:,.2f}\n"
            f"Day change: {sign}{pct_change:.2f}% (vs {prev_close:,.2f})\n"
            f"Trigger: {reason if reason else '—'}"
        )

        if should_push:
            push(title=title, message=msg)

        # Log local
        print(f"{ticker}: {price:,.2f} | {pct_change:+.2f}% | pushed={should_push} | {reason}")

        # Actualiza estado
        state[ticker]["last_price_seen"] = price
        if above is not None:
            state[ticker]["was_above"] = price >= float(above)
        if below is not None:
            state[ticker]["was_below"] = price <= float(below)

    save_state(state)


if __name__ == "__main__":
    check_once()