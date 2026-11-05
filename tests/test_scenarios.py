"""End-to-end behaviour of Stages 3-5 and the alert state machine on synthetic drivers."""

import pytest

from dms.alerts import AlertStateMachine
from dms.config import AlertConfig, load_config
from dms.simulation import ScenarioBuilder, max_tier, run_scenario

CAL = 8.0  # every scenario starts with 8 s of alert driving (calibration takes 5 s)


def _run(builder, cfg=None, **kw):
    frames = builder.build()
    out = run_scenario(frames, cfg, **kw)
    cmds = [a for o in out for a in o.alerts]
    return frames, out, cmds


def _events(cmds, *names):
    return [c for c in cmds if c.event in names]


def _check_lifecycle(cmds):
    """Every alert that starts also stops (no buzzer left on), and ids are unique."""
    open_alert = False
    for c in cmds:
        if c.event == "start":
            assert not open_alert, "an alert started while another was still sounding"
            open_alert = True
        elif c.event == "stop":
            assert open_alert
            open_alert = False
    assert len({c.id for c in cmds}) == len(cmds)
    return open_alert


def test_calibrates_on_synthetic_driver():
    _, out, _ = _run(ScenarioBuilder().alert(10))
    assert out[-1].calibration_status == "CALIBRATED"


@pytest.mark.parametrize(
    "name,build",
    [
        ("alert driver, 2 minutes", lambda b: b.alert(120)),
        ("normal blinking", lambda b: [b.alert(3.0).blink() for _ in range(30)]),
        ("talking (speech is not yawning)", lambda b: b.talk(30)),
        ("prolonged blink 0.6 s", lambda b: b.eyes_closed(0.6).alert(8)),
        ("mirror glance 1.5 s", lambda b: b.look_away(1.5).alert(8)),
        ("dashboard glances (eyes look down)", lambda b: [b.alert(4).glance_down(0.8) for _ in range(10)]),
        ("phone visible 2.5 s (< 3 s rule)", lambda b: b.phone(2.5).alert(8)),
        ("camera covered 4 s while facing forward", lambda b: b.face_lost(4).alert(8)),
        ("camera stall 5 s", lambda b: b.alert(2).camera_stall(5).alert(8)),
        ("road bump 0.2 g", lambda b: b.alert(20)),
    ],
)
def test_no_false_alarms(name, build):
    b = ScenarioBuilder().alert(CAL)
    build(b)
    frames, out, cmds = _run(b, motion_at=(10, 11, 0.2))
    assert cmds == [], name
    assert max_tier(out, CAL, frames) == "NONE"
    assert all(o.behavior.drowsiness_state == "alert" for o in out), name


def test_single_yawn_is_low_risk_soft_beep_then_stops():
    frames, out, cmds = _run(ScenarioBuilder().alert(CAL).yawn(3).alert(15))
    starts = _events(cmds, "start")
    assert [c.tier for c in starts] == ["LOW"] and starts[0].action == "soft_beep"
    assert not _check_lifecycle(cmds)
    stop = _events(cmds, "stop")[0]
    assert stop.t - (CAL + 3.0) <= 3.0  # silent within ~clear_s after the mouth closes
    assert out[-1].alert_state == "CLEARED"  # risk still LOW (yawn counted for 5 min), but no sound
    assert out[-1].behavior.drowsiness_state == "alert"  # one yawn is not "frequent yawning"


def test_frequent_yawning_is_medium_risk_bounded_and_slightly_drowsy():
    b = ScenarioBuilder().alert(CAL)
    for _ in range(3):
        b.yawn(3).alert(20)
    b.alert(60)
    frames, out, cmds = _run(b)
    assert max_tier(out, CAL, frames) == "MEDIUM"
    assert not _check_lifecycle(cmds)
    medium = [c for c in cmds if c.tier == "MEDIUM" and c.event in ("start", "escalate", "beep")]
    assert 1 <= len(medium) <= 6
    assert out[-1].active_tier == "NONE"  # nothing sounds during the final minute
    assert any(o.behavior.drowsiness_state == "slightly_drowsy" for o in out)


def test_microsleep_escalates_then_stops_when_eyes_open():
    frames, out, cmds = _run(ScenarioBuilder().alert(CAL).eyes_closed(2.0).alert(30))
    tiers = [c.tier for c in _events(cmds, "start", "escalate")]
    assert tiers[0] == "LOW" and "MEDIUM" in tiers and "HIGH" not in tiers
    assert not _check_lifecycle(cmds)
    reopened = CAL + 2.0
    assert _events(cmds, "stop")[-1].t - reopened <= 2.0 + 3 * 2.0  # clear_s + one MEDIUM burst at most
    assert any(o.behavior.drowsiness_state == "moderately_drowsy" for o in out)  # closure >= 1.5 s


def test_eyes_kept_closed_keep_the_buzzer_going():
    frames, out, cmds = _run(ScenarioBuilder().alert(CAL).eyes_closed(10.0).alert(10))
    beeps = [c.t for c in _events(cmds, "beep") if c.tier == "MEDIUM"]
    assert beeps and max(beeps) >= CAL + 9.0  # no burst limit while the eyes stay closed
    assert not _check_lifecycle(cmds)


def test_optional_escalation_for_long_closure():
    cfg = load_config(overrides={"alerts": {"escalate_closure_s": 3.0}})
    frames, out, cmds = _run(ScenarioBuilder().alert(CAL).eyes_closed(6.0).alert(10), cfg)
    assert any(c.event == "escalate" and c.tier == "HIGH" for c in cmds)
    assert any(c.push for c in cmds)


@pytest.mark.parametrize("builder", [lambda b: b.look_away(4.0), lambda b: b.look_down(4.0), lambda b: b.turn_away_and_lose_face(4.0)])
def test_sustained_gaze_deviation_is_high_risk_with_push(builder):
    b = ScenarioBuilder().alert(CAL)
    builder(b)
    b.alert(10)
    frames, out, cmds = _run(b)
    assert max_tier(out, CAL, frames) == "HIGH"
    pushes = [c for c in cmds if c.push]
    assert len(pushes) == 1
    first_high = next(c for c in cmds if c.tier == "HIGH" and c.event in ("start", "escalate"))
    assert first_high.t - CAL == pytest.approx(3.3, abs=0.25)  # 3 s rule + 0.3 s activation filter
    assert not _check_lifecycle(cmds)
    assert _events(cmds, "stop")[-1].t - (CAL + 4.0) <= 3.5  # quiet soon after looking back


def test_alarm_stops_when_driver_looks_back_although_risk_stays_elevated():
    """The reported bug: alerts continued after the driver returned to normal.

    Two 6-s look-aways leave the 30-s off-road ratio high, so the RISK stays
    elevated for a while - but the alert must stop and stay silent."""
    b = ScenarioBuilder(seed=3).alert(20).look_away(6).alert(2).look_away(6).alert(60)
    frames, out, cmds = _run(b)
    back = 20 + 6 + 2 + 6
    after = [c for c in cmds if c.t >= back and c.event in ("start", "escalate", "beep")]
    assert all(c.t - back <= 3.0 for c in after), [(c.t, c.event) for c in after]
    assert _events(cmds, "stop")[-1].t - back <= 3.0
    elevated = [o for f, o in zip(frames, out) if back + 5 <= f.t <= back + 10]
    assert any(o.risk.tier != "NONE" for o in elevated)  # assessed risk still elevated ...
    assert all(o.active_tier == "NONE" and o.alert_state == "CLEARED" for o in elevated)  # ... but silent
    assert not _check_lifecycle(cmds)


def test_new_behaviour_after_clearing_alerts_again():
    b = ScenarioBuilder(seed=4).alert(CAL).look_away(4).alert(8).look_away(4).alert(10)
    frames, out, cmds = _run(b)
    assert len(_events(cmds, "start")) == 2  # one per look-away
    assert sum(c.push for c in cmds) == 1  # second HIGH within 60 s: push suppressed, not duplicated
    assert len(_events(cmds, "push_suppressed")) == 1
    assert not _check_lifecycle(cmds)


def test_microsleeps_then_recovery_returns_to_alert():
    b = ScenarioBuilder(seed=1).alert(20).eyes_closed(3).alert(5).eyes_closed(3).alert(5).eyes_closed(2.5).alert(90)
    frames, out, cmds = _run(b)
    last_closure_end = 20 + 3 + 5 + 3 + 5 + 2.5
    assert not _check_lifecycle(cmds)
    assert max(c.t for c in cmds) - last_closure_end <= 2.0 + 6.0
    state_at = {round(f.t, 1): o.behavior.drowsiness_state for f, o in zip(frames, out)}
    assert state_at[round(last_closure_end + 1.0, 1)] == "moderately_drowsy"
    assert state_at[round(last_closure_end + 75.0, 1)] == "alert"  # two recovery steps of 30 s


def test_phone_use_confirmed_after_three_seconds_and_hysteresis():
    b = ScenarioBuilder().alert(CAL)
    for _ in range(10):  # real use: YOLO confidence drops on some frames (hand, motion blur)
        b.phone(0.4, conf=0.6).phone(0.2, conf=0.25)
    b.alert(10)
    frames, out, cmds = _run(b)
    first = next(c for c in cmds if c.event == "start")
    assert first.tier == "HIGH" and any(c.push for c in cmds)
    assert first.t - CAL == pytest.approx(3.3, abs=0.25)
    assert any(o.behavior.phone_usage for o in out)
    assert not _check_lifecycle(cmds)


def test_weak_phone_detections_alone_do_not_start_an_episode():
    frames, out, cmds = _run(ScenarioBuilder().alert(CAL).phone(5.0, conf=0.25).alert(5))
    assert cmds == [] and not any(o.behavior.phone_usage for o in out)


def test_phone_while_stopped_is_downgraded():
    frames, out, _ = _run(ScenarioBuilder().alert(CAL).phone(4.0).alert(8), speed_kmh=0.0)
    assert max_tier(out, CAL, frames) == "MEDIUM"


def test_erratic_motion_is_high_risk_and_stops_after():
    frames, out, cmds = _run(ScenarioBuilder().alert(25), motion_at=(10, 11, 0.5))
    assert max_tier(out, CAL, frames) == "HIGH"
    assert not _check_lifecycle(cmds)
    assert _events(cmds, "stop")[-1].t <= 11.0 + 3.5


def test_camera_lost_stops_alarm_and_reports_unavailable():
    cfg = load_config()
    from dms.engine import DecisionEngine
    from dms.types import ContextReading

    frames = ScenarioBuilder().alert(CAL).look_away(5).build()
    eng = DecisionEngine(cfg, alert_id_prefix="t")
    cmds = []
    for f in frames:
        cmds += eng.step(f, ContextReading(speed_kmh=40.0, driving_s=f.t)).alerts
    assert eng.alerts.state == "ALERTING"
    lost = eng.camera_lost(frames[-1].t + 3.0)
    assert [c.event for c in lost] == ["stop", "unavailable"] and lost[0].action == "off"
    later = ScenarioBuilder().alert(3).build()
    back = eng.step(later[0].__class__(**{**later[0].__dict__, "t": frames[-1].t + 6.0}), ContextReading(speed_kmh=40.0))
    assert back.alerts[0].event == "available"


def test_state_machine_push_once_per_alert_and_cooldown():
    m = AlertStateMachine(AlertConfig(), id_prefix="x")
    cmds = []
    for i in range(1200):  # 120 s with a sustained HIGH behaviour
        cmds += m.update(i * 0.1, "HIGH", ["gaze_deviation"], [], "")
    assert sum(c.push for c in cmds) == 1  # one alert -> one push, not one per minute
    assert m.episodes["HIGH"] == 1
    for i in range(1200, 1240):  # behaviour over, risk back to NONE
        cmds += m.update(i * 0.1, "NONE", [], [], "")
    assert m.state == "MONITORING" and m.active_tier == "NONE"
    assert not _check_lifecycle(cmds)


def test_activation_filter_ignores_one_frame_spikes():
    m = AlertStateMachine(AlertConfig(), id_prefix="y")
    cmds = []
    for i in range(100):
        tier = "HIGH" if i == 50 else "NONE"
        cmds += m.update(i * 0.1, tier, [], [], "")
    assert cmds == []
