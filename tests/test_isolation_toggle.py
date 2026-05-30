"""Unit tests for the runtime context-isolation toggle (wellness_copilot.isolation).

These verify each of the three isolation mechanisms actually flips when toggled,
without invoking any LLM:

  profile  — build_personalization_ctx role cards stop being role-cropped.
  history  — the transcript section appears only when history isolation is OFF.
  peer     — dispatcher._run_plan runs sequentially and threads same-batch notes.

The full create_agent capture path is covered at integration level by
scripts/evaluate_architecture.py (arch_isolation_*).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from wellness_copilot import isolation
from wellness_copilot import personalization as P
from wellness_copilot.agents import dispatcher as D
from wellness_copilot import profile_store


# ---------------------------------------------------------------------------
# Toggle contract
# ---------------------------------------------------------------------------

def test_default_is_fully_isolated():
    cfg = isolation.current()
    assert (cfg.profile, cfg.peer, cfg.history) == (True, True, True)


def test_override_restores_previous_state():
    before = isolation.current()
    with isolation.isolation_override(profile=False, peer=False, history=False):
        cur = isolation.current()
        assert (cur.profile, cur.peer, cur.history) == (False, False, False)
    assert isolation.current() == before


def test_partial_override_leaves_other_dims_untouched():
    with isolation.isolation_override(profile=False):
        cur = isolation.current()
        assert cur.profile is False
        assert cur.peer is True and cur.history is True
    assert isolation.current().profile is True


# ---------------------------------------------------------------------------
# ① profile crop
# ---------------------------------------------------------------------------

_RICH_PROFILE = {
    "physical_stats": {"age": 28, "weight": 75, "height": 178, "injuries": ["半月板损伤"]},
    "dietary_context": {"goal": "增肌", "preferences": ["素食", "海鲜过敏"]},
    "mental_state": {"stress_sources": ["工作压力"]},
}


def _seed_temp_profile(monkeypatch, tmp_path, user_id):
    store = tmp_path / "profile_store.json"
    monkeypatch.setattr(profile_store, "PROFILE_STORE_PATH", str(store))
    profile_store.update_user_profile(user_id, _RICH_PROFILE)


def test_profile_isolated_trainer_card_drops_cross_domain(monkeypatch, tmp_path):
    uid = "iso_unit_profile"
    _seed_temp_profile(monkeypatch, tmp_path, uid)

    # Isolated (default): Trainer's crop keeps weight, drops diet prefs / stress.
    card = P.build_personalization_ctx(uid)["role_user_cards"]["Trainer"]
    assert "75" in card                       # weight is in Trainer's crop
    assert "海鲜过敏" not in card              # dietary preference cropped out
    assert "工作压力" not in card              # mental_state cropped out


def test_profile_off_trainer_card_carries_full_profile(monkeypatch, tmp_path):
    uid = "iso_unit_profile_off"
    _seed_temp_profile(monkeypatch, tmp_path, uid)

    with isolation.isolation_override(profile=False):
        card = P.build_personalization_ctx(uid)["role_user_cards"]["Trainer"]
    assert "海鲜过敏" in card                  # full profile now reaches Trainer
    assert "工作压力" in card


# ---------------------------------------------------------------------------
# ③ history injection
# ---------------------------------------------------------------------------

def test_history_section_empty_when_isolated():
    # Even if a transcript is present in pctx, isolation ON suppresses it.
    pctx = {isolation.PCTX_HISTORY_KEY: "用户：你好"}
    assert isolation.noniso_history_section(pctx) == ""


def test_history_section_present_when_off():
    transcript = isolation.render_transcript([
        SystemMessage(content="sys-ignored"),
        HumanMessage(content="我有半月板损伤"),
        AIMessage(content="建议低冲击训练"),
    ])
    assert "半月板损伤" in transcript
    assert "sys-ignored" not in transcript     # system messages are skipped
    with isolation.isolation_override(history=False):
        section = isolation.noniso_history_section({isolation.PCTX_HISTORY_KEY: transcript})
    assert "半月板损伤" in section


# ---------------------------------------------------------------------------
# ② peer notes (sequential vs parallel)
# ---------------------------------------------------------------------------

def _fake_runner(role):
    """A runner that records the peer_text it received and emits a note."""
    def run(user_id, user_question, peer_text, pctx, episode_context):
        run.seen_peer_text = peer_text
        return {
            "expert_responses": {role: f"{role}-answer"},
            "agent_notes": {role: f"{role}-note"},
            "last_tools": [],
            "retrieval_hits": 0,
        }
    run.seen_peer_text = None
    return run


def test_peer_isolated_batch_hides_same_batch_notes(monkeypatch):
    runners = {"Trainer": _fake_runner("Trainer"), "Nutritionist": _fake_runner("Nutritionist")}
    monkeypatch.setattr(D, "EXPERT_RUNNERS", runners)

    # Default (peer isolation ON) → parallel, no peer sees the other's note.
    D._run_plan(["Trainer", "Nutritionist"], "u", "q", {}, {}, "")
    assert "Nutritionist-note" not in (runners["Trainer"].seen_peer_text or "")
    assert "Trainer-note" not in (runners["Nutritionist"].seen_peer_text or "")


def test_peer_off_batch_threads_notes_sequentially(monkeypatch):
    runners = {"Trainer": _fake_runner("Trainer"), "Nutritionist": _fake_runner("Nutritionist")}
    monkeypatch.setattr(D, "EXPERT_RUNNERS", runners)

    with isolation.isolation_override(peer=False):
        D._run_plan(["Trainer", "Nutritionist"], "u", "q", {}, {}, "")
    # First expert sees no peer; the second sees the first's note.
    assert "Trainer-note" in (runners["Nutritionist"].seen_peer_text or "")
