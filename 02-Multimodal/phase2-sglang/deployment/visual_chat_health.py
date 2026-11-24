"""Restart a running chat process after repeated failed local health checks."""
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from time import monotonic, sleep


# Allow model initialization and CUDA compilation before counting failures.
TARGETS = {
    "sglang": ("http://127.0.0.1:30002/health", 200, 150),
    "web": ("http://127.0.0.1:7864/", 401, 30),
}
INTERVAL = 10
FAILURE_LIMIT = 3
STATUS = re.compile(r"^(\w+)\s+RUNNING\s+pid (\d+),", re.MULTILINE)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def status(url):
    try:
        with OPENER.open(url, timeout=4) as response:
            return response.status
    except urllib.error.HTTPError as error:
        code = error.code
        error.close()
        return code
    except (OSError, urllib.error.URLError):
        return None


def main():
    control = [
        str(Path(os.environ["CHAT_SUPERVISOR_BIN"]) / "supervisorctl"),
        "-c", os.environ["CHAT_STACK_CONFIG"],
    ]
    observed = {}
    while True:
        result = subprocess.run([*control, "status"], capture_output=True, text=True, timeout=8)
        running = dict(STATUS.findall(result.stdout))
        for name, (url, expected, grace) in TARGETS.items():
            pid = running.get(name)
            if pid is None:
                observed.pop(name, None)
                continue
            previous, since, misses = observed.get(name, (None, 0, 0))
            if pid != previous:
                since, misses = monotonic(), 0
            if monotonic() - since >= grace:
                misses = 0 if status(url) == expected else misses + 1
            if misses >= FAILURE_LIMIT:
                # Recheck the PID so an already recovered process is not restarted again.
                current = subprocess.run([*control, "pid", name], capture_output=True, text=True, timeout=8)
                if current.stdout.strip() == pid:
                    print(f"{name} PID {pid} failed {misses} health checks; restarting", flush=True)
                    subprocess.run([*control, "restart", name], timeout=55, check=True)
                observed.pop(name, None)
            else:
                observed[name] = (pid, since, misses)
        sleep(INTERVAL)


if __name__ == "__main__":
    main()
