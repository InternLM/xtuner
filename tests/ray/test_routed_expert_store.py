import numpy as np
import pytest
import ray

from xtuner.v1.ray.rollout.routed_expert_store import RoutedExpertStore, get_store


@pytest.fixture(scope="module")
def ray_cluster():
    ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def store(ray_cluster):
    actor = RoutedExpertStore.remote()
    yield actor
    ray.kill(actor)


class TestRoutedExpertStore:
    def test_put_get_release_roundtrip(self, store):
        arr = np.arange(6).reshape(2, 3)
        ref = ray.put(arr)
        key = ray.get(store.put_ref.remote([ref]))

        got_ref = ray.get(store.get_ref.remote(key))
        assert np.array_equal(ray.get(got_ref), arr)

        ray.get(store.release.remote(key))
        stats = ray.get(store.stats.remote())
        assert stats["live"] == 0
        assert stats["n_put"] == 1
        assert stats["n_get"] == 1
        assert stats["n_release"] == 1

    def test_get_missing_raises(self, store):
        with pytest.raises(ray.exceptions.RayTaskError) as exc_info:
            ray.get(store.get_ref.remote("deadbeef"))
        assert "key not found" in str(exc_info.value)

        stats = ray.get(store.stats.remote())
        assert stats["n_missing_get"] == 1

    def test_put_ref_rejects_bare_ref(self, store):
        ref = ray.put(np.zeros(4))
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(store.put_ref.remote(ref))

    def test_release_many_idempotent(self, store):
        keys = [ray.get(store.put_ref.remote([ray.put(np.arange(i + 1))])) for i in range(3)]
        ray.get(store.release_many.remote(keys + keys))

        stats = ray.get(store.stats.remote())
        assert stats["live"] == 0
        assert stats["n_release"] == 3
        assert stats["n_missing_release"] == 3

    def test_ttl_eviction_via_gc(self, ray_cluster):
        actor = RoutedExpertStore.remote(ttl_sec=0, gc_interval_sec=0)
        ray.get(actor.put_ref.remote([ray.put(np.zeros(1))]))
        ray.get(actor.put_ref.remote([ray.put(np.zeros(1))]))

        stats = ray.get(actor.stats.remote())
        assert stats["live"] == 1
        ray.kill(actor)

    def test_get_store_returns_same_handle(self, ray_cluster):
        h1 = get_store()
        h2 = get_store()
        assert h1._actor_id == h2._actor_id
