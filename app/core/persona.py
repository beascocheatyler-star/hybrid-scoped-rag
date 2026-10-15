"""Persona configuration — what turns this generic engine into *a* chatbot.

The same code becomes a customer-care agent or a course TA purely by swapping
this config. Personas are stored as JSON next to the corpus so the choice
survives a restart.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Persona:
    name: str = "Assistant"
    topic: str = "the uploaded documents"
    # Free-text behavioural guidance appended to the system prompt.
    style: str = "Be clear, concise, and professional."
    greeting: str = ""
    # Shown in the UI as clickable starter questions.
    suggested_questions: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.suggested_questions is None:
            self.suggested_questions = []

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Persona":
        """Build a Persona from untrusted data, coercing bad values to defaults.

        This is a trust boundary: ``data`` may come from a hand-edited or
        corrupted ``persona.json``. Anything unusable is replaced with the field
        default rather than allowed to produce a Persona that explodes later on
        a ``.strip()`` call.
        """
        if not isinstance(data, dict):
            return cls()

        defaults = cls()
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean: Dict = {}

        for key, value in data.items():
            if key not in known:
                continue
            if key == "suggested_questions":
                if isinstance(value, list):
                    clean[key] = [str(v) for v in value if isinstance(v, (str, int))]
            elif isinstance(value, str):
                clean[key] = value
            elif value is not None:
                # Numbers and similar are salvageable as text; None is not.
                clean[key] = str(value)

        merged = {f: getattr(defaults, f) for f in known}
        merged.update(clean)
        return cls(**merged)

    def effective_greeting(self) -> str:
        if isinstance(self.greeting, str) and self.greeting.strip():
            return self.greeting
        return (
            "Hi — I'm {}. I answer questions about {}, using only the documents "
            "I've been given. What would you like to know?".format(self.name, self.topic)
        )


# Ready-made personas so the engine demonstrates both assignment examples.
PRESETS: Dict[str, Persona] = {
    "customer_care": Persona(
        name="the Northwind Support Assistant",
        topic="Northwind Outfitters orders, shipping, returns, warranties, and account help",
        style=(
            "Be warm, direct, and solution-oriented. Lead with the answer, then the "
            "detail. Quote exact figures, deadlines, and policy windows from the "
            "documentation rather than paraphrasing them loosely. If a customer needs "
            "a human, say so and point to the documented escalation path."
        ),
        suggested_questions=[
            "How long do I have to return an item?",
            "Is my item covered under warranty?",
            "How much is expedited shipping?",
            "How do I cancel an order?",
        ],
    ),
    "course_ta": Persona(
        name="the CS 1428 Teaching Assistant",
        topic="the CS 1428 syllabus, course policies, schedule, grading, and assignments",
        style=(
            "Be encouraging but precise. Students rely on you for policy facts, so "
            "quote the syllabus exactly on dates, weights, and penalties. Never guess "
            "at a deadline or a grade cutoff. For questions about course *content* "
            "that the syllabus does not cover, point the student to office hours. "
            "Never do a student's graded work for them — explain the concept instead."
        ),
        suggested_questions=[
            "How is the final grade calculated?",
            "What is the late work policy?",
            "When are office hours?",
            "What happens if I miss an exam?",
        ],
    ),
    "generic": Persona(),
}


class PersonaStore:
    """Persists the active persona to disk."""

    def __init__(self, data_dir: Path):
        self._path = Path(data_dir) / "persona.json"
        self._persona = self._load()

    def _load(self) -> Persona:
        """Load the persisted persona, falling back to defaults on any problem.

        Runs at import time, so a malformed file must never prevent startup.
        """
        if not self._path.exists():
            return Persona()
        try:
            return Persona.from_dict(json.loads(self._path.read_text("utf-8")))
        except Exception:
            logger.warning(
                "Could not read %s; falling back to the default persona.", self._path
            )
            return Persona()

    @property
    def current(self) -> Persona:
        return self._persona

    def set(self, persona: Persona) -> Persona:
        self._persona = persona
        self._path.write_text(
            json.dumps(persona.to_dict(), indent=2, ensure_ascii=False), "utf-8"
        )
        return persona

    def apply_preset(self, key: str) -> Persona:
        if key not in PRESETS:
            raise KeyError(
                "Unknown preset '{}'. Available: {}".format(
                    key, ", ".join(sorted(PRESETS))
                )
            )
        return self.set(PRESETS[key])
