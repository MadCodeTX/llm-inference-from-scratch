"""A small on-disk library of lessons from finished optimisation tasks.

Concept (not code) follows Harness² (Google, Apache-2.0; see
``harness/UPSTREAM.md``): lessons live per domain and are only kept when
they state a cue, a lesson, the evidence for it, and a confidence.
This implementation stores one folder per adopted task containing a
markdown note plus the winning candidate as JSON.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .spec import Candidate

_WHITESPACE_OR_SLASH = re.compile(r"[\s/\\]+")


def slug(task_id: str) -> str:
    """Filesystem-safe folder name for a task id."""
    return _WHITESPACE_OR_SLASH.sub("__", str(task_id)).strip("_") or "task"


def domain_dir_name(domain: str) -> str:
    cleaned = _WHITESPACE_OR_SLASH.sub("__", str(domain)).strip("_")
    return cleaned or "unknown"


@dataclass(frozen=True)
class Experience:
    cue: str = ""
    lesson: str = ""
    evidence: str = ""
    confidence: str = ""
    edits: str = ""

    @property
    def is_complete(self) -> bool:
        required = (self.cue, self.lesson, self.evidence, self.confidence)
        return all(isinstance(value, str) and bool(value.strip()) for value in required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cue": self.cue,
            "lesson": self.lesson,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "edits": self.edits,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Experience":
        source = data or {}
        return cls(**{name: str(source.get(name, "")) for name in
                      ("cue", "lesson", "evidence", "confidence", "edits")})


def has_experience(experience: Experience | dict | None) -> bool:
    if isinstance(experience, Experience):
        return experience.is_complete
    if isinstance(experience, dict):
        return Experience.from_dict(experience).is_complete
    return False


def _coerce_experience(experience: Experience | dict | None) -> Experience:
    if isinstance(experience, Experience):
        return experience
    return Experience.from_dict(experience)


def _render_note(task_id: str, candidate: Candidate, exp: Experience,
                 why: str, label: str, run_id: str) -> str:
    chosen = label or candidate.name
    parts = [
        f"# Lesson from {task_id}",
        "",
        f"Adopted candidate `{chosen}` (stored candidate `{candidate.name}`"
        + (f", run `{run_id}`" if run_id else "") + ").",
        f"Settings: `{json.dumps(candidate.settings, sort_keys=True)}`",
        "",
    ]
    if why.strip():
        parts.extend(["## Why this candidate won", "", why.strip(), ""])
    if exp.edits.strip():
        parts.extend(["## What was changed", "", exp.edits.strip(), ""])
    parts.extend([
        "## Transferable lesson",
        "",
        f"- **cue**: {exp.cue.strip()}",
        f"- **lesson**: {exp.lesson.strip()}",
        f"- **evidence**: {exp.evidence.strip()}",
        f"- **confidence**: {exp.confidence.strip()}",
        "",
    ])
    return "\n".join(parts)


def _read_cue(note_path: Path) -> str:
    if not note_path.is_file():
        return ""
    marker = "- **cue**:"
    for line in note_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(marker):
            return line[len(marker):].strip()
    return ""


@dataclass
class Library:
    """Lessons for one domain, rooted at ``root`` on disk."""

    root: Path
    domain: str

    @property
    def dir(self) -> Path:
        return Path(self.root) / domain_dir_name(self.domain)

    @property
    def index(self) -> Path:
        return self.dir / "INDEX.md"

    def entry_dir(self, task_id: str) -> Path:
        return self.dir / "harnesses" / slug(task_id)

    def exists(self) -> bool:
        return self.index.is_file()

    def entries(self) -> list[str]:
        folder = self.dir / "harnesses"
        if not folder.is_dir():
            return []
        return sorted(child.name for child in folder.iterdir() if child.is_dir())

    # Friendlier alias for callers.
    def list(self) -> list[str]:
        return self.entries()

    def read(self, task_id: str) -> str | None:
        note = self.entry_dir(task_id) / "EXPERIENCE.md"
        return note.read_text(encoding="utf-8") if note.is_file() else None

    def read_experience(self, task_id: str) -> str | None:
        return self.read(task_id)

    def read_candidate(self, task_id: str) -> Candidate | None:
        snapshot = self.entry_dir(task_id) / "candidate.json"
        if not snapshot.is_file():
            return None
        return Candidate.from_dict(json.loads(snapshot.read_text(encoding="utf-8")))

    def context_dir(self) -> Path | None:
        has_note = self.exists() and any(self.dir.rglob("EXPERIENCE.md"))
        return self.dir if has_note else None

    def add(self, task_id: str, candidate: Candidate, *,
            experience: Experience | dict | None = None,
            why: str = "", label: str = "", run_id: str = "") -> Path | None:
        """Keep a lesson only when cue/lesson/evidence/confidence are all set."""
        exp = _coerce_experience(experience)
        if not exp.is_complete:
            return None
        folder = self.entry_dir(task_id)
        folder.mkdir(parents=True, exist_ok=True)
        snapshot = json.dumps(candidate.to_dict(), indent=2, sort_keys=True)
        (folder / "candidate.json").write_text(snapshot, encoding="utf-8")
        note = _render_note(task_id, candidate, exp, why, label, run_id)
        (folder / "EXPERIENCE.md").write_text(note, encoding="utf-8")
        self.rebuild_index()
        return folder

    def forget(self, task_ids: Sequence[str]) -> None:
        changed = False
        for task_id in task_ids:
            folder = self.entry_dir(task_id)
            if folder.is_dir():
                shutil.rmtree(folder, ignore_errors=True)
                changed = True
        if changed:
            self.rebuild_index()

    def rebuild_index(self) -> Path:
        lines = [
            f"# Library index — {self.domain}",
            "",
            "Lessons kept from earlier tasks in this domain. Each entry "
            "links the stored candidate to the note explaining it.",
            "",
        ]
        names = self.entries()
        if not names:
            lines.append("_(empty — no task has contributed a lesson yet)_")
            lines.append("")
        for name in names:
            folder = self.dir / "harnesses" / name
            lines.append(f"## `harnesses/{name}/`")
            lines.append(f"- candidate snapshot: `harnesses/{name}/candidate.json`")
            lines.append(f"- lesson note: `harnesses/{name}/EXPERIENCE.md`")
            cue = _read_cue(folder / "EXPERIENCE.md")
            if cue:
                lines.append(f"- cue: {cue}")
            lines.append("")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index.write_text("\n".join(lines), encoding="utf-8")
        return self.index


__all__ = ["Experience", "Library", "domain_dir_name", "has_experience", "slug"]
