"""agentic-chart-review — an EHR chart-review agent that works like a human abstractor."""

from typing import TYPE_CHECKING, Any

from .corpus import Corpus, PatientChart
from .kernel import AssetRef, SignalEnvelope, TargetRef, Trajectory, TrajectoryAdapter
from .modules import (
    CertificationSuite,
    ModuleAsset,
    PipelineProfile,
)
from .spec import ExtractionSpec, load_spec, load_specs
from .state import Budget

if TYPE_CHECKING:
    from .agent import run_patient
    from .diagnosis.attribution import (
        AdjudicationEvent,
        AttributionPacket,
        AttributionProbe,
        AttributionReport,
        CauseFinding,
        ErrorCaseEvent,
        ErrorCluster,
    )
    from .llm import LLMClient, LLMConfig

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
        from .agent import run_patient
        return run_patient
    if name in {"LLMClient", "LLMConfig"}:
        from .llm import LLMClient, LLMConfig
        return {"LLMClient": LLMClient, "LLMConfig": LLMConfig}[name]
    if name in {
            "AttributionPacket", "AttributionProbe", "CauseFinding", "AttributionReport",
            "ErrorCaseEvent", "AdjudicationEvent", "ErrorCluster"}:
        from .diagnosis import attribution
        return getattr(attribution, name)
    raise AttributeError(name)
