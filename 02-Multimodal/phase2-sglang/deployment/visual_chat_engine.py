"""Clean up the isolated SGLang worker group when its server process exits."""
import os
import signal
import subprocess
from contextlib import suppress
from time import monotonic, sleep


def signal_group(pid, sig):
    with suppress(ProcessLookupError):
        os.killpg(pid, sig)


def main():
    deadline = None

    def stop(_sig, _frame):
        nonlocal deadline
        if deadline is None:
            deadline = monotonic() + 15

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    worker = subprocess.Popen(["bash", "scripts/start_sglang_smoke.sh"], start_new_session=True)
    stopping = False
    try:
        while worker.poll() is None:
            if deadline is not None and not stopping:
                signal_group(worker.pid, signal.SIGTERM)
                stopping = True
            if deadline is not None and monotonic() > deadline:
                signal_group(worker.pid, signal.SIGKILL)
                break
            sleep(0.25)
        return 0 if stopping else 1
    finally:
        signal_group(worker.pid, signal.SIGTERM)
        # The server master may have crashed while its scheduler still owns GPU memory.
        sleep(1)
        signal_group(worker.pid, signal.SIGKILL)
        worker.wait(timeout=3)


if __name__ == "__main__":
    raise SystemExit(main())
