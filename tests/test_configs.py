from pathlib import Path

import pytest

from groundguard.evaluation import EvalConfig

CONFIGS = sorted((Path(__file__).parents[1] / "configs").rglob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_config_parses(path: Path) -> None:
    config = EvalConfig.from_yaml(path)
    assert config.name == path.stem


def test_ragtruth_systems_share_the_same_sample() -> None:
    configs = [EvalConfig.from_yaml(p) for p in CONFIGS if p.parent.name == "ragtruth"]
    assert len(configs) >= 4
    assert len({(c.seed, c.dataset.split, c.dataset.n) for c in configs}) == 1
