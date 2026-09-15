from types import SimpleNamespace

import pytest

from xtuner.v1.rl.trainer.worker import TrainingWorker


class TestTrainingWorkerDataParallelRank:
    @pytest.mark.parametrize(
        ("rank", "tp_size", "sp_size", "expected_dp_rank"),
        [
            (0, 1, 1, 0),
            (3, 1, 1, 3),
            (0, 2, 1, 0),
            (1, 2, 1, 0),
            (2, 2, 1, 1),
            (3, 2, 1, 1),
            (0, 2, 2, 0),
            (3, 2, 2, 0),
            (4, 2, 2, 1),
            (7, 2, 2, 1),
        ],
    )
    def test_get_dp_rank_accounts_for_all_data_replicas(
        self,
        rank: int,
        tp_size: int,
        sp_size: int,
        expected_dp_rank: int,
    ) -> None:
        worker = SimpleNamespace(
            rank=rank,
            _engine=SimpleNamespace(data_replicate_size=tp_size),
            sp_mesh=SimpleNamespace(size=lambda: sp_size),
        )

        assert TrainingWorker.get_dp_rank(worker) == expected_dp_rank
