import unittest

from xtuner.v1.ray.rollout.controller import SessionRouter


class TestSessionRouter(unittest.IsolatedAsyncioTestCase):
    async def test_update_worker_status_updates_existing_sessions(self):
        worker_a = object()
        worker_b = object()
        router = SessionRouter({worker_a: True, worker_b: True})

        chosen = await router.get_worker(session_id=1)
        self.assertIn(chosen, (worker_a, worker_b))

        router.update_worker_status(chosen, False)

        next_worker = await router.get_worker(session_id=1)
        self.assertNotEqual(next_worker, chosen)
