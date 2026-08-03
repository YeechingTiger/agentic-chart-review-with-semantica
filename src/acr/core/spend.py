"""What a run is allowed to COST, priced from the same table the cost report uses.

WHY A COST CEILING AND NOT A STEP CAP. Measured over ten real charts on the current runtime:
the busiest run used 30 model calls against a cap of 50, a call costs about $0.008, and a full
50-call run is roughly $0.40. Nothing came near the cap — but two runs were cut short by a
rejection-loop brake, and one of those was stopped into a wrong answer it had turns left to
correct. The limits were the binding constraint on QUALITY while the budget was not binding at
all, which is backwards.

A limit is here to stop the absurd case: tens of dollars, millions of tokens, on one patient.
It should never be what ends a run that is working. So the default is deliberately about thirty
times a typical run — generous enough that hitting it means something is wrong, not that the
chart was hard.

Priced, not counted. `max_tokens` treats a cached prompt token and a fresh output token as the
same thing; they differ by 60x on this deployment, and 86% of this workload's prompt tokens are
cache reads. A token cap therefore mostly measures how well the cache is working.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import site

#: Where the rates live. One table, shared with `cost_report.py`, because two price tables is
#: two answers to "what did this cost".
PRICES = site.PRICES

#: USD per 1M, used when the table has no row for the model. Absent rather than zero: a run
#: whose model is unpriced must not be reported as free.
UNKNOWN = None


@dataclass
class Spend:
    """Running cost of one run, and the ceiling it may not cross."""

    max_usd: float = 5.0
    model: str = ""
    prompt: int = 0
    cached: int = 0
    completion: int = 0
    rates: dict | None = field(default=None)

    def __post_init__(self) -> None:
        if self.rates is None:
            self.rates = _rates_for(self.model)

    def add(self, usage: dict | None) -> None:
        """Accumulate one call's `usage_metadata`."""
        if not usage:
            return
        self.prompt += int(usage.get("input_tokens") or 0)
        self.completion += int(usage.get("output_tokens") or 0)
        self.cached += int((usage.get("input_token_details") or {}).get("cache_read") or 0)

    @property
    def usd(self) -> float | None:
        """None when the model is unpriced — never 0.0, which reads as free."""
        r = self.rates or {}
        if r.get("input_per_1m") is None or r.get("output_per_1m") is None:
            return None
        cached_rate = r.get("cached_input_per_1m")
        if cached_rate is None:
            # The table's own instruction: bill cache reads at full input rate and say so.
            # Overstating is the safe direction for a ceiling.
            cached_rate = r["input_per_1m"]
        full = max(0, self.prompt - self.cached)
        return (full * r["input_per_1m"] + self.cached * cached_rate
                + self.completion * r["output_per_1m"]) / 1e6

    def exceeded(self) -> str | None:
        u = self.usd
        if u is None:
            return None          # unpriced: the ceiling cannot be enforced, and says so
        return (f"cost ceiling reached: ${u:.2f} of ${self.max_usd:.2f}"
                if u >= self.max_usd else None)

    def report(self) -> dict:
        return {"max_usd": self.max_usd, "usd": self.usd, "model": self.model,
                "prompt_tokens": self.prompt, "cached_tokens": self.cached,
                "completion_tokens": self.completion,
                "priced": self.usd is not None,
                "cache_hit_rate": round(self.cached / self.prompt, 4) if self.prompt else None,
                "note": ("priced from " + str(PRICES) if self.usd is not None else
                         f"model {self.model!r} is not in {PRICES}; the ceiling is NOT enforced "
                         f"and this run's cost is unknown, not zero")}


def _rates_for(model: str) -> dict:
    try:
        table = json.loads(PRICES.read_text(encoding="utf-8")).get("models") or {}
    except (OSError, json.JSONDecodeError):
        return {}
    for key in (model, model.split("/")[-1], f"openai/{model}"):
        if key in table:
            return table[key]
    return {}
