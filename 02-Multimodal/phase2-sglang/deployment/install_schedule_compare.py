"""Install a mode switch at the single-engine SGLang 0.5.23 policy boundary."""
import argparse
from pathlib import Path

from deployment.install_session_trace import replace, install_session


def install_schedule(runtime):
    install_session(runtime)
    signature = "        processed_tokens: int = 0,\n    ) -> None:\n"
    statement = "        policy = self._determine_active_policy(waiting_queue)"
    anchor = signature + statement
    path = runtime / "sglang/srt/managers/schedule_policy.py"
    old = signature + "        from deployment.schedule_compare import arrange_experiment_queue\n" + (
        "        arrange_experiment_queue(self, waiting_queue, CacheAwarePolicy.LPM, CacheAgnosticPolicy.FCFS)\n"
    ) + statement
    updated = signature + (
        "        from deployment.schedule_compare import arrange_experiment_queue\n"
        "        stage2_features = arrange_experiment_queue(self, waiting_queue, CacheAwarePolicy.LPM,\n"
        "            CacheAgnosticPolicy.FCFS, running_batch, CacheAwarePolicy.SHORTEST_PREFILL_FIRST)\n"
    ) + statement
    replace(path, old if old in path.read_text() else anchor, updated)
    finish = "        self._touch_waiting_prefixes(policy, waiting_queue)"
    replace(path, finish, finish + (
        "\n        if stage2_features:\n"
        "            from deployment.prefill_queue_features import record\n"
        "            record(stage2_features, waiting_queue)"
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    install_schedule(parser.parse_args().runtime.resolve())
