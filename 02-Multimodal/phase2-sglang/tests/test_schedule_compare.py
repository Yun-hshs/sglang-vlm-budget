from types import SimpleNamespace

import pytest

from deployment import schedule_compare


@pytest.mark.parametrize(("mode", "waited_ms", "first_cost"), [
    ("fcfs", 150, 300), ("cost", 400, 20), ("aging100", 150, 300),
    ("aging220", 150, 20), ("aging220", 275, 300),
    ("aging350", 275, 20), ("aging350", 400, 300),
])
def test_threshold_selects_when_overdue_requests_take_priority(monkeypatch, mode, waited_ms, first_cost):
    monkeypatch.setattr(schedule_compare, "RULES", {
        "aging_rate": 0.35,
        "overdue_ms": 220,
        "aging_thresholds_ms": {"aging100": 100, "aging220": 220, "aging350": 350},
    })
    monkeypatch.setattr(schedule_compare, "perf_counter", lambda: 10.0)
    queue = [
        SimpleNamespace(
            rid=f"{schedule_compare.PREFIX}{mode}:{cost * 1000}:test",
            time_stats=SimpleNamespace(wait_queue_entry_time=entered),
        )
        for cost, entered in [(300, 10.0 - waited_ms / 1000), (20, 9.99)]
    ]
    policy = SimpleNamespace()
    schedule_compare.arrange_experiment_queue(policy, queue, "lpm", "fcfs")
    assert policy.policy == "fcfs"
    assert int(queue[0].rid.split(":")[2]) == first_cost * 1000


def test_native_spf_switch_preserves_queue_for_native_sort(monkeypatch):
    monkeypatch.setattr(schedule_compare, "RULES", {"overdue_ms": 100})
    queue = [SimpleNamespace(rid=f"{schedule_compare.PREFIX}spf:{cost}:test") for cost in (300, 20)]
    policy = SimpleNamespace()
    schedule_compare.arrange_experiment_queue(policy, queue, "lpm", "fcfs", spf="native-spf")
    assert policy.policy == "native-spf"
    assert [req.rid.split(":")[2] for req in queue] == ["300", "20"]


def test_local_score_uses_live_features_and_fixed_aging(monkeypatch):
    from deployment import prefill_queue_features

    monkeypatch.setattr(schedule_compare, "RULES", {
        "prefill_predictor_experiment": True, "overdue_ms": 100, "aging_rate": 0.35,
        "aging_thresholds_ms": {"local": 100},
    })
    monkeypatch.setattr(schedule_compare, "perf_counter", lambda: 10.0)
    rows = {"stage2-mm-session-sched:local:1000:test": {"visual_miss": 300},
            "stage2-mm-session-sched:local:900000:test": {"visual_miss": 20}}
    monkeypatch.setattr(prefill_queue_features, "snapshot", lambda queue, parts, batch: rows)
    monkeypatch.setattr(prefill_queue_features, "local_scores", lambda snapshot: {
        rid: f["visual_miss"] for rid, f in snapshot.items()
    })
    queue = [SimpleNamespace(rid=rid, time_stats=SimpleNamespace(wait_queue_entry_time=9.99))
             for rid in rows]
    policy = SimpleNamespace()
    schedule_compare.arrange_experiment_queue(policy, queue, "lpm", "fcfs")
    assert queue[0].rid.split(":")[2] == "900000"
    queue[1].time_stats.wait_queue_entry_time = 9.8
    schedule_compare.arrange_experiment_queue(policy, queue, "lpm", "fcfs")
    assert queue[0].rid.split(":")[2] == "1000"
