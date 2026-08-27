"""Launch Semantica Explorer on one ACR decision-narrative provenance bundle.

ACR owns only the chart-review domain projection.  The page, API, review validation, and durable
human-review entries belong to Semantica Explorer; this module merely selects the graph and its
run-local provenance store and opens Semantica's Decisions workspace.
"""
from __future__ import annotations

import os
import threading
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode


def _semantica_ui_imports() -> tuple[Any, Any]:
    try:
        # Importing this module is the feature check.  Released 0.6.6 has the generic Explorer,
        # but not durable decision narratives/reviews; ACR ships an upstreamable patch for it.
        from semantica.explorer import decision_review as _decision_review  # noqa: F401
        from semantica.explorer.app import create_app
        from semantica.explorer.session import GraphSession
    except ImportError as exc:  # pragma: no cover - install profile dependent
        raise ImportError(
            "install the patched Semantica Explorer with "
            "`tools/install_semantica_decision_review.sh`"
        ) from exc
    return create_app, GraphSession


def _selected_review(ledger_path: Path, run_id: str,
                     analysis_id: str | None) -> tuple[Any, str, Path]:
    from acr.mvp.ledger import SemanticaLedger

    ledger = SemanticaLedger(Path(ledger_path))
    chain = ledger.chain(str(run_id), analysis_id)
    if chain.get("status") != "OK":
        available = ", ".join(chain.get("available_analysis_ids") or []) or "none"
        raise ValueError(
            f"select one analysis for run {run_id!r} before review; available: {available}"
        )
    chosen = str(chain["analysis_id"])
    # Human-readable narrative projections are append-only and revisioned separately
    # from the sealed reconstruction.  Opening an older run must materialize the current
    # presentation revision before Semantica Explorer asks for its bundle id.
    ledger.project_analysis(ledger.load_analysis_artifact(str(run_id), chosen))
    metadata = ledger.analysis_metadata(str(run_id), chosen)
    provenance_path = Path(str(metadata.get("provenance_path") or ""))
    if not provenance_path.is_file():
        raise ValueError(f"Semantica provenance store is missing: {provenance_path}")
    bundle_id = ledger.decision_narrative_bundle_id(str(run_id), chosen)
    return ledger, bundle_id, provenance_path.resolve()


def review_url(host: str, port: int, bundle_id: str) -> str:
    query = urlencode({"workspace": "decisions", "bundle_id": bundle_id})
    return f"http://{host}:{port}/?{query}"


def create_review_app(ledger_path: Path, run_id: str, analysis_id: str | None = None, *,
                      run_dir: Path | None = None) -> Any:
    """Create the stock Semantica app bound to one graph and provenance database."""
    create_app, GraphSession = _semantica_ui_imports()
    ledger, bundle_id, provenance_path = _selected_review(
        Path(ledger_path), str(run_id), analysis_id)
    if run_dir is not None:
        from acr.mvp.semantica_audit import PROVENANCE_FILENAME

        expected = (Path(run_dir).resolve() / PROVENANCE_FILENAME).resolve()
        if expected != provenance_path:
            raise ValueError(
                f"run_dir points to {expected}, but the selected Semantica analysis uses "
                f"{provenance_path}"
            )
    session = GraphSession(
        ledger.graph, provenance_storage_path=str(provenance_path))
    app = create_app(session=session)
    app.title = "Semantica Knowledge Explorer · ACR Decision Review"
    app.state.acr_decision_bundle_id = bundle_id
    return app


def serve_review_ui(ledger_path: Path, run_id: str, analysis_id: str | None = None, *,
                    run_dir: Path | None = None, host: str = "127.0.0.1", port: int = 8765,
                    open_browser: bool = True) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("the Semantica review UI may bind only to a loopback host")
    if os.environ.get("SEMANTICA_ALLOW_ANONYMOUS", "").strip().lower() == "true":
        raise ValueError("anonymous Semantica access is forbidden for the reviewer UI")
    if not os.environ.get("SEMANTICA_API_KEY"):
        raise ValueError("set SEMANTICA_API_KEY before starting the Semantica reviewer UI")
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - selected install extras
        raise ImportError('install -e ".[ledger,ledger-ui]" to run the reviewer UI') from exc

    app = create_review_app(
        ledger_path, run_id, analysis_id, run_dir=run_dir)
    url = review_url(host, port, str(app.state.acr_decision_bundle_id))
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Semantica Decisions review: {url}")
    uvicorn.run(app, host=host, port=port, log_level="info")
