"""Fork: stt.second_pass (off-list language / repetition loop -> Deepgram -> forced local)."""

from tools import transcription_second_pass as sp

CFG = {"local": {"allowed_languages": ["en", "uk", "ru"]}, "second_pass": {"provider": "deepgram"}}


def test_repetition_loop_detection():
    assert sp.has_repetition_loop("так так так так так так")
    assert sp.has_repetition_loop("я не знаю, я не знаю, я не знаю, я не знаю, я не знаю, я не знаю")
    assert not sp.has_repetition_loop("так так так так так")
    assert not sp.has_repetition_loop("a normal sentence")


def test_reason_requires_provider_and_offlist_autodetect():
    assert sp.second_pass_reason({"local": CFG["local"]}, "lv", "x", False) is None
    assert "not in allowed" in sp.second_pass_reason(CFG, "lv", "x", False)
    assert sp.second_pass_reason(CFG, "uk", "x", False) is None
    assert sp.second_pass_reason(CFG, "lv", "x", True) is None  # language was forced
    assert "repetition" in sp.second_pass_reason(CFG, "ru", "да " * 8, False)


def test_best_allowed_language():
    assert sp.best_allowed_language(["en", "uk", "ru"], [("lv", .5), ("ru", .2), ("uk", .25)]) == "uk"
    assert sp.best_allowed_language(["en", "uk", "ru"], None) == "en"


def test_deepgram_success_wins(monkeypatch):
    seen = {}
    def fake(path, cfg, hints):
        seen["hints"] = hints
        return {"success": True, "transcript": "clean", "provider": "deepgram"}
    monkeypatch.setitem(sp._SECOND_PASS_PROVIDERS, "deepgram", fake)
    r = sp.run_second_pass("a.ogg", CFG, "why", "lv", None, lambda lang: "nope", "garbage")
    assert r["transcript"] == "clean" and seen["hints"] == ["en", "uk", "ru"]


def test_deepgram_failure_forces_best_local_language(monkeypatch):
    monkeypatch.setitem(sp._SECOND_PASS_PROVIDERS, "deepgram",
                        lambda *a: {"success": False, "transcript": "", "error": "down"})
    forced = []
    r = sp.run_second_pass("a.ogg", CFG, "why", "lv", [("lv", .6), ("ru", .3), ("uk", .1)],
                           lambda lang: forced.append(lang) or f"text-{lang}", "garbage")
    assert forced == ["ru"] and r == {"success": True, "transcript": "text-ru", "provider": "local"}
