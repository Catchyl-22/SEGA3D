"""Import SEGA3D and check registrations/configs without training or loading weights."""
import argparse
import importlib
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-path", action="append", help="Check only this configuration; repeat for multiple files.")
    parser.add_argument("--native-ops", action="store_true", help="Also import the compiled pointgroup extension; requires its build environment.")
    args = parser.parse_args()
    config_paths = [Path(name) if Path(name).is_absolute() else ROOT / name for name in args.cfg_path] if args.cfg_path else sorted((ROOT / "lavis/projects/sega3d").rglob("*.yaml"))
    if not config_paths:
        parser.error("No SEGA3D configurations found.")
    for path in config_paths:
        if not path.is_file():
            parser.error(f"Configuration not found: {path}")

    print(f"Repository: {ROOT}", flush=True)
    print(f"Python: {sys.executable}", flush=True)
    try:
        if args.native_ops:
            native_ops = importlib.import_module("pointgroup_ops")
            for name in ("voxelization_idx", "voxelization"):
                if not callable(getattr(native_ops, name, None)):
                    raise RuntimeError(f"Native operation not found: {name}")
            print("[OK] compiled pointgroup extension import (kernels are not executed)")
        # Execute the real package initializers and model imports.
        # Importing classes does not construct models or download checkpoints.
        import lavis.models as models
        import lavis.datasets.builders as builders
        import lavis.tasks as tasks
        from lavis.common.config import Config
        from lavis.common.registry import registry
        from lavis.sega3d_registry import COMPONENTS, register_components

        register_components()
        methods = {"models": registry.get_model_class, "datasets": registry.get_builder_class, "tasks": registry.get_task_class}
        identities = {}
        for group, components in COMPONENTS.items():
            for key, (module_name, class_name) in components.items():
                implementation = getattr(importlib.import_module(module_name), class_name)
                if methods[group](key) is not implementation:
                    raise RuntimeError(f"Incorrect registration: {key}")
                identities[key] = implementation
                print(f"[OK] {key} -> {module_name}.{class_name}")
        if models.SEGA3D is not registry.get_model_class("sega3d_t5_nv2"):
            raise RuntimeError("lavis.models.SEGA3D does not export nv2.")
        if models.SEGA3DNV3 is not registry.get_model_class("sega3d_t5_nv3"):
            raise RuntimeError("lavis.models.SEGA3DNV3 does not export nv3.")
        if not set(COMPONENTS["models"]).issubset(models.model_zoo.model_zoo):
            raise RuntimeError("ModelZoo was created before SEGA3D model registration.")
        if not set(COMPONENTS["datasets"]).issubset(builders.dataset_zoo.get_names()):
            raise RuntimeError("DatasetZoo was created before SEGA3D dataset registration.")
        register_components()
        for group, components in COMPONENTS.items():
            for key in components:
                if methods[group](key) is not identities[key]:
                    raise RuntimeError(f"Repeated registration changed {key}.")

        for path in config_paths:
            cfg = Config(argparse.Namespace(cfg_path=str(path), options=None, replace_cfg=None))
            if registry.get_model_class(cfg.model_cfg.arch) is None:
                raise RuntimeError(f"Model not found: {cfg.model_cfg.arch}")
            for name in cfg.datasets_cfg:
                if registry.get_builder_class(name) is None:
                    raise RuntimeError(f"Dataset not found: {name}")
            task = tasks.setup_task(cfg)
            if not isinstance(task, registry.get_task_class(cfg.run_cfg.task)):
                raise RuntimeError(f"Incorrect task setup: {cfg.run_cfg.task}")
            print(f"[OK] configuration: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    except Exception:
        traceback.print_exc()
        print("[FAILED] Runtime import/configuration check. Use the SEGA3D environment in docs/installation.md.", file=sys.stderr)
        return 1

    print(f"[PASS] Real package imports, 6 component registrations, exported nv2/nv3 classes, task setup, and {len(config_paths)} configurations.")
    print("No model construction, checkpoint loading, data loading, forward pass, or training was performed.")
    if not args.native_ops:
        print("Compiled pointgroup operations were not loaded or executed. Use --native-ops to check extension import.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
