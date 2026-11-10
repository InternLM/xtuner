"""Bound connect retries without replaying an ambiguous write after read timeout."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests


@pytest.fixture
def admin():
    # Exercise the actual helpers without importing the GPU training stack.
    source = Path(__file__).parents[2] / 'xtuner/v1/rl/utils/misc.py'
    tree = ast.parse(source.read_text())
    names = {'_post_proxy_admin', 'register_to_routedapiproxy', 'delete_from_routedapiproxy'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == 3
    post = Mock()
    ns = {'requests': SimpleNamespace(post=post, exceptions=requests.exceptions),
          'time': Mock(), 'logger': Mock()}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    return ns, post


def test_connect_timeout_recovers(admin):
    ns, post = admin
    response = Mock()
    response.json.return_value = {'code': 0}
    post.side_effect = [requests.exceptions.ConnectTimeout(), response]
    assert ns['register_to_routedapiproxy']('http://admin', 'owned-model', 'http://backend') == {'code': 0}
    assert post.call_count == 2
    assert post.call_args.kwargs['timeout'] == (5, 180)
    assert post.call_args.kwargs['json']['api_base'] == 'http://backend'
    ns['time'].sleep.assert_called_once_with(1)


def test_connect_retries_are_bounded(admin):
    ns, post = admin
    post.side_effect = requests.exceptions.ConnectTimeout()
    with pytest.raises(requests.exceptions.ConnectTimeout):
        ns['delete_from_routedapiproxy']('http://admin', 'owned-model', 'http://backend')
    assert post.call_count == 5
    assert ns['time'].sleep.call_count == 4
    assert post.call_args.kwargs['timeout'] == (5, 30)
    assert post.call_args.kwargs['json'] == {'model_name': 'owned-model', 'api_base': 'http://backend'}


@pytest.mark.parametrize('error', [requests.exceptions.ReadTimeout,
                                    requests.exceptions.ConnectionError,
                                    requests.exceptions.HTTPError])
def test_non_connect_errors_are_not_retried(admin, error):
    ns, post = admin
    post.side_effect = error()
    with pytest.raises(error):
        ns['delete_from_routedapiproxy']('http://admin', 'owned-model')
    assert post.call_count == 1
    ns['time'].sleep.assert_not_called()


def test_http_error_is_not_retried(admin):
    ns, post = admin
    post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError()
    with pytest.raises(requests.exceptions.HTTPError):
        ns['register_to_routedapiproxy']('http://admin', 'owned-model', 'http://backend')
    assert post.call_count == 1
    ns['time'].sleep.assert_not_called()
