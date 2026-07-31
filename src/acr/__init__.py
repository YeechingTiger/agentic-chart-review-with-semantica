"""agentic-chart-review — an EHR chart-review agent that works like a human abstractor."""

from typing import TYPE_CHECKING, Any

from .chartstore.corpus import Corpus, PatientChart
from .contract.spec import ExtractionSpec, load_spec, load_specs
from .core.kernel import AssetRef, SignalEnvelope, TargetRef, Trajectory, TrajectoryAdapter
from .core.modules import (
    CertificationSuite,
    ModuleAsset,
    PipelineProfile,
)
from .core.state import Budget

if TYPE_CHECKING:
    from .core.llm import LLMClient, LLMConfig
    from .diagnosis.attribution import (
        AdjudicationEvent,
        AttributionPacket,
        AttributionProbe,
        AttributionReport,
        CauseFinding,
        ErrorCaseEvent,
        ErrorCluster,
    )
    from .review.agent import run_patient

__version__ = "0.1.0"
__all__ = [
    "AdjudicationEvent",
    "AssetRef",
    "AttributionPacket",
    "AttributionProbe",
    "AttributionReport",
    "Budget",
    "CauseFinding",
    "CertificationSuite",
    "Corpus",
    "ErrorCaseEvent",
    "ErrorCluster",
    "ExtractionSpec",
    "LLMClient",
    "LLMConfig",
    "ModuleAsset",
    "PatientChart",
    "PipelineProfile",
    "SignalEnvelope",
    "TargetRef",
    "Trajectory",
    "TrajectoryAdapter",
    "load_spec",
    "load_specs",
    "run_patient",
]


def __getattr__(name: str) -> Any:
    """Keep the public API lazy so metadata-only commands do not import model SDKs."""
    if name == "run_patient":
        from .review.agent import run_patient
        return run_patient
    if name in {"LLMClient", "LLMConfig"}:
        from .core.llm import LLMClient, LLMConfig
        return {"LLMClient": LLMClient, "LLMConfig": LLMConfig}[name]
    if name in {
            "AttributionPacket", "AttributionProbe", "CauseFinding", "AttributionReport",
            "ErrorCaseEvent", "AdjudicationEvent", "ErrorCluster"}:
        from .diagnosis import attribution
        return getattr(attribution, name)
    raise AttributeError(name)
