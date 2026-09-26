from pathlib import Path
import os
import subprocess

here = Path(__file__).resolve().parent
env = os.environ.copy()
env["PATH"] = str(Path.home() / ".cargo" / "bin") + os.pathsep + env.get("PATH", "")
env["CARGO_INCREMENTAL"] = "0"
env["CARGO_NET_OFFLINE"] = "true"

cmd = [
    "flock",
    "/tmp/furiosa-extreme-20260923.lock",
    "cargo",
    "furiosa-opt",
    "compile",
    "ops::sliding_project_qkv",
    "--exact",
    "--dump-schedule",
    "schedule-k1-p067.json",
]

with (here / "static-p067.log").open("w") as log:
    result = subprocess.run(cmd, cwd=here, env=env, stdout=log, stderr=subprocess.STDOUT)
(here / "static-p067.exit").write_text(str(result.returncode))
raise SystemExit(result.returncode)
