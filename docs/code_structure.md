# SEGA3D component registration

## Central registration table

All SEGA3D model, dataset, and task names are defined in **[lavis/sega3d_registry.py](../lavis/sega3d_registry.py)**. Its `COMPONENTS` table maps each public name to an implementation module and class. `register_components()` imports these classes and adds them to the shared registry.

| Group | Registered name | Implementation |
|---|---|---|
| Model | `sega3d_t5_nv2` | [sega3d_t5_nv2.py](../lavis/models/sega3d_models/sega3d_t5_nv2.py), `SEGA3DT5` |
| Model | `sega3d_t5_nv3` | [sega3d_t5_nv3.py](../lavis/models/sega3d_models/sega3d_t5_nv3.py), `SEGA3DT5` |
| Dataset | `sega3d_scanrefer` | [sega3d_builder.py](../lavis/datasets/builders/sega3d_builder.py), `SEGA3DScanReferBuilder` |
| Dataset | `sega3d_reasoning_scannet` | Same builder module, `SEGA3DScanNetBuilder` |
| Dataset | `sega3d_reasoning_matterport` | Same builder module, `SEGA3DMatterportBuilder` |
| Task | `sega3d_seg` | [sega3d_seg_task.py](../lavis/tasks/sega3d_seg_task.py), `SEGA3DSegTask` |

Each group is registered by its package initializer:

- [lavis/models/__init__.py](../lavis/models/__init__.py) calls `register_components("models")`.
- [lavis/datasets/builders/__init__.py](../lavis/datasets/builders/__init__.py) calls `register_components("datasets")`.
- [lavis/tasks/__init__.py](../lavis/tasks/__init__.py) calls `register_components("tasks")`.

[lavis/common/registry.py](../lavis/common/registry.py) implements the shared registry, including `register_model`, `register_builder`, `register_task`, and their lookup functions. The implementation files contain the classes; the central table owns their public names.

## Configuration lookup

[config.py](../lavis/common/config.py) merges model defaults, dataset defaults, and the selected task YAML. [train.py](../train.py) uses the registered components. For example, `lavis/projects/sega3d/train/sega3d-scannet-nv3.yaml` selects:

```yaml
model:
  arch: sega3d_t5_nv3
datasets:
  sega3d_reasoning_scannet:
    # Dataset-specific overrides
run:
  task: sega3d_seg
```

Each builder's `DATASET_CONFIG_DICT` points to `lavis/configs/datasets/<registered_name>/defaults.yaml`. The dataset implementations are [scanrefer_dataset.py](../lavis/datasets/datasets/scanrefer_dataset.py), [reasoning_scannet_dataset.py](../lavis/datasets/datasets/reasoning_scannet_dataset.py), and [reasoning_matterport_dataset.py](../lavis/datasets/datasets/reasoning_matterport_dataset.py).

## Configuration files

Both `lavis/projects/sega3d/train/` and `lavis/projects/sega3d/val/` contain:

- `sega3d-scanrefer-nv2.yaml`
- `sega3d-scannet-nv2.yaml`
- `sega3d-scannet-nv3.yaml`
- `sega3d-matterport-nv2.yaml`
- `sega3d-matterport-nv3.yaml`

Use `data/reasoning/train.json` and `data/reasoning/val.json` for reasoning annotations, and `data/candidates/reasoning/` for reasoning candidate inputs. Their data fields and point ordering are described in [the reasoning guide](reasoning.md). Tensor attribute names and checkpoint `state_dict` keys are unchanged by the code naming updates. Upstream sources and licenses are documented in [third-party notices](../THIRD_PARTY_NOTICES.md).
