from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_clean_room_copy_and_symlink_scan():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_clean_room.py"), "--scan-only"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert '"copied_tree": "PASS"' in result.stdout
    assert '"symlinks": "PASS"' in result.stdout
