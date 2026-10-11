"""Small stable facade while legacy script entry points remain supported."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict


@dataclass(frozen=True)
class PipelineConfig:
    output_dir: Path
    ma_detector: Literal['classic', 'trained-2048'] = 'classic'
    referral_probe: Literal['none', 'idrid'] = 'none'


class PipelineResult(TypedDict, total=False):
    schema_version: int
    status: str
    research_only: bool
    grade: int
    triage_action: str
    report_path: str
    bundle_sha256: str
    explanation_target: str


def predict_image(image: str | Path, config: PipelineConfig) -> PipelineResult:
    from inference import run_pipeline
    return run_pipeline(str(image), str(config.output_dir),
                        ma_detector=config.ma_detector, referral_probe=config.referral_probe)
