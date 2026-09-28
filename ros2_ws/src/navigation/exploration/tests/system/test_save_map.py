"""Tests for nav2's save_map.sh, the single implementation of map persistence.

The script is tested directly rather than through a Python wrapper because it IS
the implementation: the recorder shells out to the same script an operator runs
by hand, so testing the script tests both callers at once.

No ROS graph is needed. A fake `ros2` executable is placed first on PATH and
scripted per test to report whatever service list and result codes the case
needs. That lets the failure modes that matter (slam_toolbox absent, a service
returning a non-zero result, files silently not landing) be exercised
deterministically, which a live slam_toolbox could not do on demand.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[3] / "nav2" / "scripts" / "save_map.sh")


def _fake_ros2(bin_dir: Path, *, services: str, result: str = "0",
               write_files: bool = True, stem: str = "map") -> None:
    """Install a fake `ros2` on PATH.

    services    what `ros2 service list` prints
    result      the result= code every `ros2 service call` reports
    write_files whether the fake creates the four output files, so the "services
                said ok but nothing landed" case can be reproduced (this is what
                a container path mismatch looks like from the caller's side)
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "ros2"
    touch = ""
    if write_files:
        # The path is the last argument of the request; recover it from the filename: field the same way slam_toolbox would.
        touch = (
            'base=$(printf "%s" "$*" | grep -oE "(/[^\'\\" ]+)/' + stem + '" | head -1)\n'
            '    if [ -n "$base" ]; then\n'
            '      for ext in posegraph data pgm yaml; do echo x > "$base.$ext"; done\n'
            '    fi\n'
        )
    script.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "service" ] && [ "$2" = "list" ]; then\n'
        f'  printf "%s\\n" {services}\n'
        "  exit 0\n"
        "fi\n"
        'if [ "$1" = "service" ] && [ "$2" = "call" ]; then\n'
        f"    {touch}"
        f'  echo "response:"\n'
        f'  echo "slam_toolbox.srv.Response(result={result})"\n'
        "  exit 0\n"
        "fi\n"
        "exit 1\n"
    )
    script.chmod(0o755)


def _run(out_dir: Path, bin_dir: Path, *extra: str):
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    return subprocess.run([str(SCRIPT), str(out_dir), *extra],
                          capture_output=True, text=True, env=env, timeout=60)


BOTH_SERVICES = '"/slam_toolbox/serialize_map" "/slam_toolbox/save_map"'


def test_script_exists_and_is_executable():
    assert SCRIPT.is_file(), f"save_map.sh missing at {SCRIPT}"
    assert os.access(SCRIPT, os.X_OK), "save_map.sh must be executable"


def test_bad_usage_exits_1(tmp_path):
    r = subprocess.run([str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 1
    assert "usage" in r.stderr.lower()


def test_missing_slam_toolbox_exits_2(tmp_path):
    """A known_map run has no slam_toolbox. That is normal, not a failure, and must be distinguishable from a real error so the recorder does not warn."""
    _fake_ros2(tmp_path / "bin", services='"/some/other/service"')
    r = _run(tmp_path / "out", tmp_path / "bin")
    assert r.returncode == 2
    assert "not running" in r.stderr


def test_successful_save_writes_all_four_files(tmp_path):
    """Both formats must appear: the pose-graph for slam_toolbox localization and the PGM/YAML for the exploration planner."""
    _fake_ros2(tmp_path / "bin", services=BOTH_SERVICES)
    out = tmp_path / "out"
    r = _run(out, tmp_path / "bin")
    assert r.returncode == 0, r.stderr
    for ext in ("posegraph", "data", "pgm", "yaml"):
        f = out / f"map.{ext}"
        assert f.is_file() and f.stat().st_size > 0, f"missing/empty map.{ext}"


def test_provenance_is_txt_not_yaml(tmp_path):
    """source.txt must NOT be a .yaml: exploration's _resolve_map_paths takes glob('*.yaml')[0] and glob order is not guaranteed, so a second YAML here could be loaded as the map descriptor instead of map.yaml."""
    _fake_ros2(tmp_path / "bin", services=BOTH_SERVICES)
    out = tmp_path / "out"
    _run(out, tmp_path / "bin", "map", "/runs/some_run")
    assert (out / "source.txt").is_file()
    assert list(out.glob("*.yaml")) == [out / "map.yaml"], \
        "exactly one .yaml must exist in a map folder"
    assert "/runs/some_run" in (out / "source.txt").read_text()


def test_nonzero_result_code_fails(tmp_path):
    """slam_toolbox reports failure in the response body, not by failing the call, so a result=255 must not be read as success."""
    _fake_ros2(tmp_path / "bin", services=BOTH_SERVICES, result="255",
               write_files=False)
    r = _run(tmp_path / "out", tmp_path / "bin")
    assert r.returncode == 3
    assert "result=255" in r.stderr


def test_services_ok_but_no_files_is_a_failure(tmp_path):
    """The container-path trap: slam_toolbox resolves the output path in its own process, so it can report success while writing somewhere the caller cannot see. Without the file check this would look like a good save."""
    _fake_ros2(tmp_path / "bin", services=BOTH_SERVICES, result="0",
               write_files=False)
    r = _run(tmp_path / "out", tmp_path / "bin")
    assert r.returncode == 3
    assert "MISSING" in r.stderr


def test_out_dir_is_created(tmp_path):
    _fake_ros2(tmp_path / "bin", services=BOTH_SERVICES)
    out = tmp_path / "deep" / "nested" / "out"
    r = _run(out, tmp_path / "bin")
    assert r.returncode == 0, r.stderr
    assert out.is_dir()


def test_custom_stem_is_honoured(tmp_path):
    _fake_ros2(tmp_path / "bin", services=BOTH_SERVICES, stem="lab_ghent")
    out = tmp_path / "out"
    r = _run(out, tmp_path / "bin", "lab_ghent")
    assert r.returncode == 0, r.stderr
    assert (out / "lab_ghent.posegraph").is_file()


class TestLaunchBranching:
    """The known-map default must keep starting AMCL, unchanged."""

    @staticmethod
    def _branch(map_path, localization, serialized_map, graph_exists=True):
        # Mirrors _make_nav_actions in nav2.launch.py. Kept in sync by hand; the end-to-end check is launching it for real.
        if localization not in ('amcl', 'slam_toolbox'):
            return "error"
        if map_path and localization == 'slam_toolbox':
            if not serialized_map or not graph_exists:
                return "error"
            return "slam_toolbox"
        elif map_path:
            return "amcl"
        return "slam_mapping"

    def test_known_map_still_defaults_to_amcl(self):
        assert self._branch("/m/map.yaml", "amcl", "") == "amcl"

    def test_no_map_still_means_slam_mapping(self):
        assert self._branch("", "amcl", "") == "slam_mapping"

    def test_slam_toolbox_localization_selected(self):
        assert self._branch("/m/map.yaml", "slam_toolbox", "/m/map") == "slam_toolbox"

    def test_slam_toolbox_without_stem_errors(self):
        assert self._branch("/m/map.yaml", "slam_toolbox", "") == "error"

    def test_typo_errors_rather_than_falling_back_to_amcl(self):
        assert self._branch("/m/map.yaml", "slamtoolbox", "/m/map") == "error"


class TestRecorderGating:
    """The recorder must only attempt a save on SLAM runs.

    _save_slam_map is exercised without constructing a real recorder (that needs
    a live ROS context); the mode gate is the whole contract being pinned, and
    calling it unbound keeps the test free of rclpy.
    """

    @staticmethod
    def _call_with_mode(mode: str):
        """Run _save_slam_map against a stand-in holding just the fields it reads."""
        import types
        from pathlib import Path as P

        recorder_py = P(__file__).with_name("recorder.py")
        src = recorder_py.read_text()
        # Pull the method out of the class body rather than importing recorder.py, which drags in rclpy, tf2_ros and numpy.
        start = src.index("    def _save_slam_map(self)")
        end = src.index("    # Default staging dir shared with run_with_log.sh")
        body = "\n".join(line[4:] if line.startswith("    ") else line
                         for line in src[start:end].splitlines())
        ns: dict = {"os": os, "time": __import__("time"), "Path": P,
                    "subprocess": subprocess}
        exec(body, ns)

        calls = []
        log = types.SimpleNamespace(
            warn=lambda m: calls.append(("warn", m)),
            info=lambda m: calls.append(("info", m)))
        fake = types.SimpleNamespace(
            _cfg={"mode": mode, "scene": "lab_ghent"},
            MAP_STORE_DEFAULT="/nonexistent/store",
            run_dir=P("/tmp/run"),
            _saved_map_dir="",
            get_logger=lambda: log,
            _append_meta=lambda extra: calls.append(("meta", extra)))
        ns["_save_slam_map"](fake)
        return calls

    def test_known_map_run_does_nothing(self):
        """A known_map run has no slam_toolbox; it must not warn or attempt a save."""
        assert self._call_with_mode("known_map") == []

    def test_slam_run_attempts_a_save(self):
        """A slam run must get as far as looking for the script, so the gate is not inverted."""
        calls = self._call_with_mode("slam")
        assert calls, "slam run should have attempted the save"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
