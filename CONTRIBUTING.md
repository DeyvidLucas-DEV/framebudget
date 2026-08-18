# Contributing

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a pull request

```
ruff format .
ruff check .
mypy src/framebudget
pytest
```

All four have to pass. CI runs the same commands on Python 3.10 through 3.13.

## Tests

Tests build their own videos in `tests/conftest.py` rather than shipping binary
fixtures. Each one has known structure: `cut_video` has cuts at 5, 10 and 15
seconds, `static_video` has none, `motion_video` moves continuously without ever
cutting. That last one matters, because telling motion apart from a cut is the
hard part of this problem and the easy way to break it is to tune for footage
that only ever does one of the two.

If you change anything in `signals.py`, `scenes.py` or `selection.py`, add a
test that pins the behaviour you are relying on. The thresholds in this codebase
interact, and a change that looks local usually is not.

## Style

- English everywhere: code, identifiers, comments, docstrings, docs.
- Comments explain why, never what. If a line needs a comment to say what it
  does, rename the variable instead.
- Public API is fully typed and documented. Private helpers carry types but may
  skip docstrings.
- Errors are specific. Raise a package defined exception with an actionable
  message, never a bare `Exception`.
- No prints in library code. The CLI layer is the only place allowed to write to
  stdout.
- Minimum supported Python is 3.10.

## Commits

Conventional Commits. `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
