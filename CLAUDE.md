# CLAUDE.md

This repository builds the `blender-blueprint-reconstruction` agent skill: it rebuilds Blender models from front/side/back reference images.

- **Start here:** read [HANDOFF.md](HANDOFF.md) for progress, the interrupted WI-1 work, and the prioritized next steps.
- **Design for the next work items:** [docs/materials-uv-design.md](docs/materials-uv-design.md).
- **Skill package:** `blender-blueprint-reconstruction/`. Its `SKILL.md` must stay at or under 1000 words.
- **Conventions:** core code in `scripts/bbr/core/` is pure numpy with pytest tests. `scripts/bbr/blender/` uses bpy. API functions return `ok` dicts and never raise.
- **Tests:** `python3 -m pytest -q tests -p no:cacheprovider` (1.5-4 min). The Blender smoke test needs Blender 5.1+: `blender -b --factory-startup -P tests/blender/test_blender_smoke.py`.
- Clean `__pycache__` and `.pytest_cache` after test runs.
