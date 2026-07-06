"""Claude Haiku trade reviewer for the HighSame bot.

Takes the feature bundle for a candidate trade and returns a structured
decision: BUY_UP / BUY_DOWN / SKIP, plus Claude's own confidence (0-100)
and a short reason string. The bot may flip direction on the model's
suggestion when market priors strongly favor the other side.

Temperature is 0 and the prompt is fixed so decisions are auditable.
"""

import json
import os
from typing import Any

from anthropic import Anthropic


_SYSTEM = """You review trade candidates for binary 5-minute BTC up/down markets on Polymarket.

Inputs you receive (JSON):
- predicted_direction ("UP" or "DOWN") — the predictor's directional call
- predicted_confidence_pct — the predictor's *signed* strength (0-100 scale). Not P(UP).
- model_prob_up — the predictor's actual P(UP) on 0..1. Use THIS for edge calc.
- drift_pct: live BTC price vs the window's price-to-beat, expressed in percent
- signals: sub-model votes (ptb, ob, lstm, pm) as 0-100 "prob UP" percentages
- top_ask_up, top_ask_down: lowest asks per side on Polymarket, 0..1, ~market-implied prob
- recent_trades: last few realized trades (may be empty)

Edge calculation (CRITICAL — do not confuse):
- BUY_UP edge = model_prob_up - top_ask_up
- BUY_DOWN edge = (1 - model_prob_up) - top_ask_down

A POSITIVE edge means your probability exceeds what you pay; a NEGATIVE edge means you are paying more than the event is worth. NEVER trade on negative edge.

Also: payoff is $1/share if you win, $0 if you lose. So buying at ask = 0.95 pays 1/0.95 ≈ 1.05x on win; after the 5% you already paid, profit per $1 stake is only ~$0.05. Buying near-certainty favorites (ask > 0.80) is a bad trade unless your model assigns >0.90+ to that side.

Quick sanity examples:
- top_ask_up=0.05, top_ask_down=0.95, model_prob_up=0.40 → BUY_UP edge = 0.40 - 0.05 = +0.35 GOOD; BUY_DOWN edge = 0.60 - 0.95 = -0.35 BAD. Choose BUY_UP.
- top_ask_up=0.55, top_ask_down=0.45, model_prob_up=0.58 → BUY_UP edge = +0.03 thin; BUY_DOWN edge = -0.03 bad. SKIP unless strong signal alignment.
- top_ask_up=0.99, top_ask_down=0.01, model_prob_up=0.95 → BUY_UP edge = -0.04 BAD (paying 99c for 95c expected); BUY_DOWN edge = +0.04 but tiny payoff. SKIP — favorite overpriced, dog underbacked.

Minimum actionable edge: roughly +0.05 (5 cents per dollar staked). Below that, SKIP.

CRITICAL RULE — EXTREME-CERTAINTY MARKET OVERRIDE:
When either side is priced < 0.05 or > 0.95, the market is signaling near-certainty. On 5-minute BTC up/down markets, extreme prices almost always reflect live price action that our model's features (sampled at window open) haven't caught up to. The apparent "huge edge" is an illusion — the market has newer information.

Rule: If top_ask_up < 0.05 or top_ask_up > 0.95 (same for top_ask_down), you MUST return SKIP unless predicted_confidence_pct >= 25. In practice, this means don't take contrarian bets against a 95%+ certain market just because your model still assigns 40-50% to the underdog — the market is almost certainly right and your model is stale.

Example of the trap:
- model_prob_up=0.43, top_ask_up=0.02, top_ask_down=0.98
- Naive edge for BUY_UP: 0.43 - 0.02 = +0.41 (looks huge!)
- Reality: market priced UP at 2% because BTC already moved and UP can't recover in the remaining seconds. Model is lagging. SKIP.

The market is binary: up_price + down_price ~= 1.0. An ask of 0.17 for UP means the market prices UP at 17%.

Decide ONE of:
- BUY_UP: take the UP share at top_ask_up
- BUY_DOWN: take the DOWN share at top_ask_down
- SKIP: no trade this window

Rules of thumb:
1. Edge matters. If the ask already prices in our view (model says UP 60%, market says UP 55%), the edge is small — SKIP unless confidence is high.
2. Flip when signals disagree with the predictor in a concentrated way. If predictor says UP 10% but LSTM is the only UP voter, PTB/OB/PM all favor DOWN, and market prices UP at 0.15 — prefer BUY_DOWN.
3. Contra-drift is a yellow flag, not a hard stop. A small adverse drift is fine on a high-confidence call; a large one usually isn't.
4. Extreme prices (<0.20 or >0.80) on our side mean we're betting against heavy consensus. Only do so on very strong multi-signal agreement.
5. When signals conflict badly (e.g. LSTM 80% UP but PTB and PM say DOWN), prefer SKIP.

Return ONLY valid JSON with these exact keys:
{"action": "BUY_UP" | "BUY_DOWN" | "SKIP", "confidence": <int 0-100>, "reason": "<one-sentence explanation>"}

No prose outside the JSON. No markdown fences."""


_client: Anthropic | None = None


def _get_client(api_key: str | None) -> Anthropic:
    global _client
    if _client is None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        _client = Anthropic(api_key=key)
    return _client


def _extract_json(text: str) -> dict[str, Any]:
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in response: {text!r}")
    return json.loads(s[start : end + 1])


def review_trade(
    features: dict[str, Any],
    *,
    model: str = "claude-haiku-4-5",
    api_key: str | None = None,
    max_tokens: int = 300,
) -> dict[str, Any]:
    client = _get_client(api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        system=_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(features, indent=2)}],
    )
    text = resp.content[0].text
    parsed = _extract_json(text)

    action = parsed.get("action", "SKIP")
    if action not in ("BUY_UP", "BUY_DOWN", "SKIP"):
        action = "SKIP"
    try:
        conf = int(parsed.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0
    reason = str(parsed.get("reason", ""))[:200]
    return {"action": action, "confidence": conf, "reason": reason}
