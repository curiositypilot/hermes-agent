"""Camofox tab-gone recovery: 404 (tab reaped) and 410 Gone (server restarted, old tab id dead).

Regression for intake t_b88ef19e: after a camofox OOM restart every browser_* call failed with 410
because only ``_navigate_tab`` recovered (and only from 404); the dead ``tab_id`` stayed in the session
and ``browser_console`` misreported a dead tab as "evaluation not supported".
"""

import json

import pytest
import requests

from tools import browser_camofox as mod


def _http_error(status: int) -> requests.HTTPError:
    resp = requests.Response()
    resp.status_code = status
    resp.url = "http://camofox.test/tabs/dead-tab/x"
    return requests.HTTPError(f"{status} Client Error for url: {resp.url}", response=resp)


def _raise(status: int):
    def _fn(*_args, **_kwargs):
        raise _http_error(status)
    return _fn


@pytest.fixture
def session(monkeypatch):
    """A tracked session holding a tab id the server no longer knows."""
    sess = {"user_id": "hermes_test", "tab_id": "dead-tab", "session_key": "task_t",
            "managed": False, "adopt_existing_tab": False}
    monkeypatch.setattr(mod, "_sessions", {"t": sess})
    monkeypatch.setattr(mod, "get_camofox_url", lambda: "http://camofox.test")
    monkeypatch.setattr(mod, "_camofox_private_page_block", lambda *a, **k: None)
    return sess


@pytest.mark.parametrize("status", [404, 410])
def test_navigate_recovers_from_410(monkeypatch, session, status):
    calls = []

    def fake_post(path, body, timeout=None):
        calls.append(path)
        if path.endswith("/navigate"):
            raise _http_error(status)
        assert path == "/tabs"
        return {"tabId": "fresh-tab"}

    monkeypatch.setattr(mod, "_post", fake_post)
    sess, data = mod._navigate_tab("t", "https://www.olx.pt/")

    assert calls == ["/tabs/dead-tab/navigate", "/tabs"]
    assert sess["tab_id"] == "fresh-tab"
    assert data["url"] == "https://www.olx.pt/"


def test_navigate_reraises_other_http_errors(monkeypatch, session):
    monkeypatch.setattr(mod, "_post", _raise(500))
    with pytest.raises(requests.HTTPError):
        mod._navigate_tab("t", "https://www.olx.pt/")
    assert session["tab_id"] == "dead-tab"


@pytest.mark.parametrize("status", [404, 410])
@pytest.mark.parametrize("action", ["snapshot", "click", "type", "scroll"])
def test_tab_action_410_clears_tab(monkeypatch, session, status, action):
    monkeypatch.setattr(mod, "_post", _raise(status))
    monkeypatch.setattr(mod, "_get", _raise(status))
    call = {
        "snapshot": lambda: mod.camofox_snapshot(task_id="t"),
        "click": lambda: mod.camofox_click("@e3", task_id="t"),
        "type": lambda: mod.camofox_type("@e4", "hello", task_id="t"),
        "scroll": lambda: mod.camofox_scroll("down", task_id="t"),
    }[action]

    result = json.loads(call())

    assert result["success"] is False
    assert "browser_navigate" in result["error"]
    assert session["tab_id"] is None
    # Later calls report the missing session instead of hammering the dead id again.
    assert "browser_navigate" in json.loads(mod.camofox_click("@e3", task_id="t"))["error"]


def test_tab_action_other_error_keeps_tab(monkeypatch, session):
    monkeypatch.setattr(mod, "_post", _raise(500))
    result = json.loads(mod.camofox_click("@e3", task_id="t"))
    assert result["success"] is False
    assert session["tab_id"] == "dead-tab"


def test_cleared_tab_is_not_readopted(monkeypatch, session):
    """Managed persistence re-adopts tabs from the server's list; a dead id must not come back."""
    session.update(managed=True, adopt_existing_tab=True)
    monkeypatch.setattr(mod, "_post", _raise(410))
    mod.camofox_click("@e3", task_id="t")
    assert session["tab_id"] is None

    monkeypatch.setattr(mod, "_get", lambda path, params=None, timeout=None: {"tabs": []})
    assert mod._get_session("t")["tab_id"] is None

    live = {"tabs": [{"tabId": "live-tab", "listItemId": "task_t"}]}
    monkeypatch.setattr(mod, "_get", lambda path, params=None, timeout=None: live)
    assert mod._get_session("t")["tab_id"] == "live-tab"


@pytest.mark.parametrize("status", [404, 410])
def test_camofox_eval_tab_gone_not_unsupported(monkeypatch, session, status):
    from tools import browser_tool

    monkeypatch.setattr(mod, "_post", _raise(status))
    result = json.loads(browser_tool._camofox_eval("document.title", "t"))

    assert result["success"] is False
    assert "not supported" not in result["error"]
    assert "browser_navigate" in result["error"]
    assert session["tab_id"] is None


@pytest.mark.parametrize("status", [405, 501])
def test_camofox_eval_unsupported_status_keeps_message(monkeypatch, session, status):
    from tools import browser_tool

    monkeypatch.setattr(mod, "_post", _raise(status))
    result = json.loads(browser_tool._camofox_eval("document.title", "t"))

    assert "not supported" in result["error"]
    assert session["tab_id"] == "dead-tab"


@pytest.mark.parametrize("status", [404, 410])
def test_vault_camofox_eval_410_clears_tab(monkeypatch, session, status):
    from tools import browser_vault_tool

    monkeypatch.setattr(mod, "_post", _raise(status))
    result = browser_vault_tool._camofox_eval("t", "window.location.origin")

    assert result["success"] is False
    assert "browser_navigate" in result["error"]
    assert session["tab_id"] is None
