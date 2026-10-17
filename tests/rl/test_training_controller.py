import unittest
from unittest.mock import MagicMock

from xtuner.v1.rl.trainer.controller import TrainingController


class TestTrainingControllerActorGroup(unittest.TestCase):
    def test_workers_are_registered_as_actor_group(self):
        worker = object()
        controller = TrainingController(workers=[worker])

        self.assertIs(controller.group("actor").workers[0], worker)
        self.assertTrue(controller.group("actor").can_sync_rollout)

    def test_duplicate_group_names_are_rejected(self):
        controller = TrainingController(workers=[])

        with self.assertRaisesRegex(ValueError, "duplicate worker group"):
            controller.register(controller.group("actor"))

    def test_fit_delegates_to_grpo_batch_without_touching_packing(self):
        controller = TrainingController(workers=[])
        controller.train_grpo_batch = MagicMock(return_value=["log"])
        data_batches = [MagicMock()]

        result = controller.fit(data_batches, pack_max_length=128, rollout_idx=3)

        self.assertEqual(result, ["log"])
        controller.train_grpo_batch.assert_called_once_with(data_batches, 128, 3)

    def test_weight_update_uses_actor_group_only(self):
        class Worker:
            def __init__(self):
                self.calls = []

            def weight_update(self, **kwargs):
                self.calls.append(kwargs)

        worker = Worker()
        controller = TrainingController(workers=[worker])

        controller.weight_update(need_update=True)

        self.assertEqual(worker.calls, [{"need_update": True}])

    def test_only_actor_role_is_available_in_first_stage(self):
        controller = TrainingController(workers=[])

        controller.switch_role_modules("actor")
        with self.assertRaisesRegex(KeyError, "actor-only stage"):
            controller.switch_role_modules("critic")


if __name__ == "__main__":
    unittest.main()
