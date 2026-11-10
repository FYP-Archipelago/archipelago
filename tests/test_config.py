import pytest
from conftest import make_run
from pydantic import ValidationError

from archipelago import ClusterStream, Config, cluster_run


def test_config_requires_an_enabled_stage() -> None:
    with pytest.raises(ValidationError, match="enable at least one clustering stage"):
        Config()


@pytest.mark.parametrize("name", ["stream", "birch_thresold"])
def test_unknown_or_misplaced_settings_are_rejected(name: str) -> None:
    with pytest.raises(ValidationError) as error:
        Config(lsh=True, **{name: True})

    assert error.value.errors()[0]["loc"] == (name,)
    assert error.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seed_outside_supported_range_is_rejected(seed: int) -> None:
    with pytest.raises(ValidationError) as error:
        Config(lsh=True, seed=seed)

    assert error.value.errors()[0]["loc"] == ("seed",)


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
@pytest.mark.parametrize("streaming", [False, True])
def test_seed_boundaries_work_in_both_modes(seed: int, streaming: bool) -> None:
    run = make_run(per_island=10, islands=2)
    config = Config(lsh=True, seed=seed)
    if streaming:
        stream = ClusterStream(config)
        stream.cluster(run.rows, run.events)
        result = stream.finish()
    else:
        result = cluster_run(run.rows, run.events, config)

    assert result.clustering.counts.rows_clustered == 20
    assert result.clustering.counts.nodes > 0
