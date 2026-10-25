"""Minimal `blender` CLI emulation on top of the PyPI `bpy` module.

Cloud sessions cannot reach download.blender.org, but PyPI is reachable and
ships Blender as `bpy`. This shim accepts the subset of Blender's CLI the repo
uses: -b/--background, --factory-startup, -P/--python FILE, --python-expr EXPR,
--python-exit-code N, and `--` followed by script arguments.
"""
import runpy
import sys
import traceback


def main(argv):
    scripts, exit_code, i = [], 0, 0
    while i < len(argv):
        a = argv[i]
        if a == "--":
            break
        if a in ("-P", "--python"):
            scripts.append(("file", argv[i + 1])); i += 2; continue
        if a == "--python-expr":
            scripts.append(("expr", argv[i + 1])); i += 2; continue
        if a == "--python-exit-code":
            exit_code = int(argv[i + 1]); i += 2; continue
        if a in ("-v", "--version"):
            import bpy
            print(f"Blender {bpy.app.version_string} (bpy module)"); return 0
        i += 1  # -b, --factory-startup and other flags are implied/ignored
    sys.argv = ["blender", *argv]
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=False)
    for kind, val in scripts:
        try:
            if kind == "file":
                runpy.run_path(val, run_name="__main__")
            else:
                exec(compile(val, "<expr>", "exec"), {"__name__": "__main__"})
        except SystemExit as e:
            return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        except Exception:
            traceback.print_exc()
            if exit_code:
                return exit_code
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
