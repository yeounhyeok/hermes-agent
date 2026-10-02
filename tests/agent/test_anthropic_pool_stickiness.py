"""Pool-bound Anthropic agents must keep the credential the pool selected.

Regression: ``_try_refresh_anthropic_client_credentials`` used to call
``resolve_anthropic_token()`` on every request, which returns the FIRST pool
OAuth entry regardless of strategy (least_quota / rotation), silently undoing
the pool's selection so one account absorbed all traffic.
"""
import types
from unittest import mock

import run_agent
from agent.credential_pool import PooledCredential


def _entry(eid, token, priority):
    return PooledCredential(
        provider="anthropic",
        id=eid,
        label=f"acct-{eid}",
        auth_type="oauth",
        priority=priority,
        source="manual:test",
        access_token=token,
    )


class _FakePool:
    def __init__(self, entries):
        self._entries = entries

    def entries(self):
        return list(self._entries)

    def entry_id_for_api_key(self, key):
        for e in self._entries:
            if e.access_token == key:
                return e.id
        return None


def _agent(pool, token, entry_id):
    return types.SimpleNamespace(
        api_mode="anthropic_messages",
        provider="anthropic",
        model="claude-test",
        _anthropic_api_key=token,
        _anthropic_base_url="https://api.anthropic.com",
        _anthropic_client=mock.MagicMock(),
        _credential_pool=pool,
        _credential_pool_entry_id=entry_id,
    )


def test_refresh_keeps_pool_selected_entry_not_first_entry():
    first, second = _entry("a1", "tok-first", 0), _entry("b2", "tok-second", 1)
    pool = _FakePool([first, second])
    agent = _agent(pool, "tok-second", "b2")

    with mock.patch("agent.credential_pool.load_pool", return_value=pool), \
         mock.patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="tok-first"), \
         mock.patch("agent.anthropic_adapter.build_anthropic_client", return_value=mock.MagicMock()):
        changed = run_agent.AIAgent._try_refresh_anthropic_client_credentials(agent)

    assert changed is False
    assert agent._anthropic_api_key == "tok-second"


def test_refresh_picks_up_rotated_token_of_same_entry():
    pool = _FakePool([_entry("a1", "tok-first", 0), _entry("b2", "tok-second-NEW", 1)])
    agent = _agent(pool, "tok-second-OLD", "b2")

    with mock.patch("agent.credential_pool.load_pool", return_value=pool), \
         mock.patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="tok-first"), \
         mock.patch("agent.anthropic_adapter.build_anthropic_client", return_value=mock.MagicMock()):
        changed = run_agent.AIAgent._try_refresh_anthropic_client_credentials(agent)

    assert changed is True
    assert agent._anthropic_api_key == "tok-second-NEW"
