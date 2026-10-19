import threading
import time
from types import SimpleNamespace

import pytest
from xtuner.v1.train import lazy_media_prefetch as m


def batch(n):
    return [{'seq_ctx': SimpleNamespace(media_plan={'id': n})}]


@pytest.mark.parametrize('leader', [True, False])
def test_bounded_order_and_close(monkeypatch, leader):
    fetched, decoded = [], []
    def raw():
        for n in range(100):
            fetched.append(n)
            yield batch(n)
    def decode(plan):
        decoded.append(plan['id'])
        return plan['id']
    monkeypatch.setattr(m, 'materialize_image_plan', decode)
    it = m.lookahead_iter(raw(), decode_here=leader, count=20)
    for n in range(3):
        value = next(it)
        assert value[0]['seq_ctx'].media_plan['id'] == n
        time.sleep(.02)
        assert len(fetched) <= n + 2
    it.close()
    assert len(fetched) <= 4
    assert decoded == fetched if leader else decoded == []
    assert not any(t.name.startswith('lazy-media-lookahead') for t in threading.enumerate())


def test_exact_count_text_and_error(monkeypatch):
    def decode(plan):
        if plan['id'] == 2:
            raise ValueError('bad image')
        return plan['id']
    monkeypatch.setattr(m, 'materialize_image_plan', decode)
    inputs = [batch(0), [{'seq_ctx': SimpleNamespace(media_plan=None)}], batch(2)]
    values = list(m.lookahead_iter(iter(inputs), decode_here=True, count=3))
    assert m.take_media_future(values[0][0]['seq_ctx'].media_plan).result() == 0
    with pytest.raises(ValueError, match='bad image'):
        m.take_media_future(values[2][0]['seq_ctx'].media_plan).result()
    assert not any(t.name.startswith('lazy-media-lookahead') for t in threading.enumerate())


def test_fetch_failure_closes():
    def raw():
        yield batch(0)
        raise ValueError('data failure')
    with pytest.raises(ValueError, match='data failure'):
        list(m.lookahead_iter(raw(), decode_here=False, count=3))
    assert not any(t.name.startswith('lazy-media-lookahead') for t in threading.enumerate())
