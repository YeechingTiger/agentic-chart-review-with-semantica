"""Token and cost accounting, wired to whichever client the runtime happens to use.

SEPARATE FROM EITHER RUNTIME ON PURPOSE. This was defined inside `deep_runner`, so the module
that owns the audit wiring was one of the two things being audited — and `agent.py` had to
import a legacy runtime to record its own spend. It has no framework dependency, so it does
not belong behind one.

Two plumbing paths for one ledger: `sitecustomize` hooks LiteLLM for the LangGraph arm, and
`lc_callback` hooks LangChain callbacks for the hooks arm, both emitting the same JSON rows so
`cost_report.py` works across both. That equality is the point — the two arms have to be
comparable on spend as well as on answers.
"""
from __future__ import annotations

import os
import sys

#: Where `lc_callback` lives. Outside the tree because it is shared with the LiteLLM path's
#: `sitecustomize` hook and predates this package; overridable for a different checkout.
AUDIT_DIR = os.getenv("ACR_AUDIT_DIR", "/N/project/computable_phenotype/llm/audit")


def _callbacks(tracer=None):
    """Audit + optional langfuse. deepagents bypasses LiteLLM, so sitecustomize's hook never
    fires here and a run's cost would otherwise be invisible.

    IT WAS INVISIBLE. `lc_callback` is not on the path from this package, the import raised
    ModuleNotFoundError, `except Exception: pass` ate it, and every deepagents run since has
    written `usage: null` — five real charts with no cost recorded at all, on the one arm
    whose cost we most needed to compare. The module was there and working the whole time.

    So: the search path is explicit, and a failure is REPORTED. Auditing that is off because
    nobody set `ACR_AUDIT_LOG` and auditing that is off because the import broke look
    identical in a manifest, and only one of them is a decision.
    """
    if AUDIT_DIR and AUDIT_DIR not in sys.path:
        sys.path.insert(0, AUDIT_DIR)
    cbs, why = [], []
    try:
        from lc_callback import langfuse_handler, make_handler
    except Exception as e:  # noqa: BLE001
        why.append(f"import lc_callback from {AUDIT_DIR!r}: {type(e).__name__}: {e}")
    else:
        for name, fn in (("audit", make_handler), ("langfuse", langfuse_handler)):
            try:
                h = fn()
            except Exception as e:  # noqa: BLE001
                why.append(f"{name}: {type(e).__name__}: {e}")
                continue
            if h is None:
                why.append(f"{name}: not configured "
                           f"({'ACR_AUDIT_LOG' if name == 'audit' else 'LANGFUSE_HOST'} unset)")
            else:
                cbs.append(h)
    if tracer is not None:
        tracer.emit("callbacks_attached", severity="warning" if not cbs else "info",
                    n=len(cbs), audit_dir=AUDIT_DIR, why=why,
                    message=("no callback attached: this run's token usage will not be "
                             "recorded" if not cbs else "token usage is being recorded"))
    return cbs
