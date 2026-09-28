#!/usr/bin/env python3
"""Count stuck episodes and human-unstuck interventions from an exploration log.

The exploration node has no dedicated STUCK status/topic: when the robot parks in
Nav2's inflation band it emits a throttled WARN, and when it gives up it emits an
ERROR — both only in the node's console log (logs/exploration.log, produced when
the run is launched via run_with_log.sh). This module reads that log and reports:

  human_unstuck   episodes the robot could NOT escape on its own — no Recovery
                  Spin fired and it resumed navigating from a clearly different
                  pose, i.e. a human physically moved it to free space.
  self_recovered  episodes the robot escaped itself — a Recovery Spin fired
                  during the episode, or it resumed at ~the same pose.
  ended_stuck     episodes where the run ENDED while stuck (give-up ERROR, or the
                  log stops mid-episode): no resume pose, so a human either
                  stopped it or never freed it. Counted separately on purpose.

An "episode" is a contiguous run of inflation-band WARN lines; the `Stuck for Ns`
counter resetting toward 0 (or a Navigating line in between) starts a new one.

Log-only, no clock cross-referencing: the resume pose is read from the
`Navigating to waypoint N (col=.., row=..)` line the node prints on the next
successful plan, and the pre-stuck pose from the last such line before the block.
"""
from __future__ import annotations

import re
from pathlib import Path

# A recovery spin translates little; only a human moves the robot far. Used only
# as a fallback when NO Recovery Spin line is present in the episode.
_SELF_RECOVER_MOVE_PX = 30.0

# A human can only be classified as having freed the robot if it was actually
# stuck long enough for a human to react and move it. A momentary "Stuck for 0s"
# WARN that the node resolves on the very next plan (Nav2 replans to a different,
# far-away waypoint on its own) is NOT a human assist even though the resume pose
# is far — distance alone mislabels those blips. Require the inflation block to
# persist to at least this many seconds before a far resume counts as human.
_HUMAN_MIN_STUCK_S = 15

_NAV_RE = re.compile(r"Navigating to waypoint \d+ \(col=(\d+), row=(\d+)\)")
_STUCK_RE = re.compile(r"inside the inflation band.*Stuck for (\d+)s")
_GIVEUP_RE = re.compile(
    r"inside the inflation band.*Stopping exploration"
    r"|Robot stuck:.*Stopping exploration")
_RECOVERY_RE = re.compile(r"Recovery Spin")

# Why a run reached the COMPLETE state. IMPORTANT: the node prints the generic
# "Exploration complete. Final coverage: X%" banner on EVERY entry to the COMPLETE
# state, including the give-up paths — so that line is NOT proof of a clean finish.
# The specific give-up messages are authoritative and take priority; only fall
# back to coverage_complete when none of them was logged. Listed in priority order.
_STOP_REASONS = [
    ("stuck_in_inflation",
     re.compile(r"inside the inflation band.*Stopping exploration")),
    ("wedged_plans_failed",
     re.compile(r"Robot stuck:.*consecutive fully-failed plans.*Stopping")),
    ("candidate_pool_exhausted",
     re.compile(r"Candidate pool is exhausted.*stopping")),
    ("no_progress",
     re.compile(r"No-progress stop:")),
    ("coverage_complete",
     re.compile(r"Exploration complete")),
]


def analyze_stuck(log_path: Path) -> dict | None:
    """Return {human_unstuck, self_recovered, ended_stuck, episodes} or None.

    None when the log file does not exist (run not launched via run_with_log.sh),
    so callers can render 'n/a' rather than a misleading 0.
    """
    if not log_path.is_file():
        return None

    last_nav: tuple[int, int] | None = None
    episodes: list[dict] = []
    cur: dict | None = None

    def close(resume: tuple[int, int] | None, ended: bool = False) -> None:
        nonlocal cur
        if cur is None:
            return
        cur["resume"] = resume
        cur["ended"] = ended
        episodes.append(cur)
        cur = None

    for line in log_path.read_text(errors="replace").splitlines():
        nav = _NAV_RE.search(line)
        if nav:
            pose = (int(nav.group(1)), int(nav.group(2)))
            close(pose)                 # a Navigating line ends any open episode
            last_nav = pose
            continue
        if _GIVEUP_RE.search(line):
            if cur is None:             # give-up without a preceding WARN block
                cur = {"start": last_nav, "recovery": False}
            close(None, ended=True)
            continue
        stuck = _STUCK_RE.search(line)
        if stuck:
            if cur is None:
                cur = {"start": last_nav, "recovery": False}
            if _RECOVERY_RE.search(line):
                cur["recovery"] = True
            continue
        if _RECOVERY_RE.search(line) and cur is not None:
            cur["recovery"] = True

    close(None, ended=True)             # log ended mid-episode -> ended_stuck

    human = self_rec = ended = 0
    for e in episodes:
        if e.get("ended"):
            ended += 1
        elif e.get("recovery"):
            self_rec += 1
        else:
            s, r = e.get("start"), e.get("resume")
            moved = (((s[0] - r[0]) ** 2 + (s[1] - r[1]) ** 2) ** 0.5
                     if s and r else None)
            if moved is not None and moved >= _SELF_RECOVER_MOVE_PX:
                human += 1
            else:
                self_rec += 1        # resumed ~in place -> escaped on its own

    # Stop reason: take the FIRST reason (in priority order) that appears at all.
    # coverage_complete is last, so a give-up message always wins over the generic
    # "Exploration complete" banner that trails every stop. None when no
    # COMPLETE-path message was logged (Ctrl-C, or the recorder stopped first).
    text = log_path.read_text(errors="replace")
    stop_reason = None
    for name, rx in _STOP_REASONS:
        if rx.search(text):
            stop_reason = name
            break

    return {
        "human_unstuck": human,
        "self_recovered": self_rec,
        "ended_stuck": ended,
        "episodes": len(episodes),
        "stop_reason": stop_reason,
    }


def stuck_for_run(run_dir: Path) -> dict | None:
    """analyze_stuck for a run folder's logs/exploration.log."""
    return analyze_stuck(Path(run_dir) / "logs" / "exploration.log")


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(p, "->", stuck_for_run(Path(p)) or "no log")
