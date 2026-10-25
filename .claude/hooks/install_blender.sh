#!/bin/bash
# SessionStart hook: provide `blender` (5.1+) in Claude Code cloud sessions.
set -euo pipefail
[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0
# bpy needs EGL for offscreen rendering
ldconfig -p | grep -q libEGL.so.1 || { apt-get update -qq && apt-get install -y -qq libegl1 libgl1 libxkbcommon0 libsm6 libxi6 libxxf86vm1 libxfixes3 libxrender1; } >/dev/null 2>&1 || true
python3 -c "import bpy, numpy, PIL, pytest" 2>/dev/null \
  || pip install -q "bpy==5.1.1" numpy Pillow pytest >/dev/null 2>&1
if ! command -v blender >/dev/null 2>&1; then
  cat > /usr/local/bin/blender <<SH
#!/bin/bash
exec python3 "$CLAUDE_PROJECT_DIR/.claude/hooks/blender_shim.py" "\$@"
SH
  chmod +x /usr/local/bin/blender
fi
