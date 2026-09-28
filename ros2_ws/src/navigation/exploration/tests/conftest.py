"""
Add all TRAVIS package roots to sys.path so tests can import packages
without installing them.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent

for _pkg in ("travis_brain", "navigation", "perception", "speech"):
    sys.path.insert(0, str(_ROOT / _pkg))

sys.path.insert(0, str(_ROOT))

# Repo-root assets/ holds the pristine reference maps (unmodified). Tests read
# from here; only maps modified/produced by a test belong under tests/.


def _find_repo_root(start: Path) -> Path:
    """Walk upward from `start` to the repo root, identified by a top-level
    marker. Independent of where the repo is checked out and of how deeply
    nested this file is, so moving or restructuring the tree won't break it.
    """
    for parent in (start, *start.parents):
        if (parent / ".git").exists() or (parent / "docker-compose.yml").is_file():
            return parent
    raise RuntimeError(f"repo root not found above {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
ASSETS_DIR = REPO_ROOT / "assets"

# Reference map used by the standalone tools/demos that need a single map.
# Swap the folder name here to change it; no tool hardcodes a map path.
REFERENCE_MAP_DIR = ASSETS_DIR / "lab_ghent"
REFERENCE_MAP_PGM = REFERENCE_MAP_DIR / "map.pgm"
REFERENCE_MAP_YAML = REFERENCE_MAP_DIR / "map.yaml"

# Inflation radius (metres) shared by tests. Matches the nav2 global costmap.
INFLATION_M = 0.9


def discover_maps() -> "list[Path]":
    """Every map folder under assets/ that has both map.yaml and map.pgm.

    Tests are parametrised over this so each map (lab_05, lab_ghent, …) is
    exercised; drop a new folder into assets/ and it is picked up automatically.
    """
    import pytest  # local import so non-pytest importers don't need it
    maps = sorted(
        d for d in ASSETS_DIR.glob("*")
        if (d / "map.yaml").is_file() and (d / "map.pgm").is_file()
    )
    if not maps:
        raise FileNotFoundError(f"No usable map folders under {ASSETS_DIR}")
    return maps


def pytest_generate_tests(metafunc):
    """Parametrise any test requesting the `assets_map_dir` fixture over all
    assets maps. The name is test-infra specific (not the generic ``map_dir``)
    so it can never collide with a test's own ``map_dir`` parametrization."""
    if "assets_map_dir" in metafunc.fixturenames:
        maps = discover_maps()
        metafunc.parametrize("assets_map_dir", maps, ids=[d.name for d in maps])
