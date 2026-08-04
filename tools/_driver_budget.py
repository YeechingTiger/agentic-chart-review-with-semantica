"""What a driver script is about to spend, in the unit it is actually denominated in.

`acr batch --max-usd` is a **per-run** priced ceiling: `cli_common.MAX_USD` says so — "priced
per-run ceiling in USD; stops an unfinished run when reached". Both driver scripts passed it
straight through and reported it as an arm total:

    ap.add_argument("--max-usd", type=float, default=3.0, help="per arm")
    print(f"... ceiling=${args.max_usd:.2f}/arm")

So `run_ladder.py --dry-run` on the 27-chart cohort printed `$3.00/arm` where the worst case was
`27 × $3.00 = $81.00` per arm and `$567.00` across seven. Understated 27-fold — and `--dry-run`
exists for exactly one purpose, which is to let somebody decide whether to spend the money.

BOTH NUMBERS, ALWAYS, with the unit on each. The per-run ceiling alone is the defect above. The
total alone reads as a committed spend rather than a bound nothing is expected to reach: a run that
answers in four turns costs cents against a $3 ceiling, and this repo's own measured batches came in
around $0.13 per chart. A reader given one number cannot tell which they were given.

One function, called by both drivers, because "what does this cost" having two implementations is
how one of them ended up wrong for a fortnight without the other noticing.
"""

from __future__ import annotations


def budget_report(*, n_arms: int, n_charts: int, max_usd_per_run: float) -> dict:
    """The two numbers and the arithmetic between them.

    `worst_case_usd` is a BOUND, not a forecast. Every run stops at the ceiling or earlier, so the
    product is what the operator has authorised, not what the batch is expected to draw.
    """
    runs = max(int(n_arms), 0) * max(int(n_charts), 0)
    return {"n_arms": int(n_arms), "n_charts": int(n_charts), "runs": runs,
            "per_run_ceiling_usd": round(float(max_usd_per_run), 2),
            "worst_case_usd": round(runs * float(max_usd_per_run), 2)}


def budget_line(report: dict) -> str:
    """One line naming both numbers and which is which."""
    return (f"{report['n_arms']} arm(s) x {report['n_charts']} chart(s) = "
            f"{report['runs']} run(s);  ${report['per_run_ceiling_usd']:.2f} per run ceiling"
            f"  ->  ${report['worst_case_usd']:.2f} worst case, if every run reaches its ceiling")
