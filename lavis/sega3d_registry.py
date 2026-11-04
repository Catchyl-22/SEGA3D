"""Central registration table for SEGA3D models, datasets, and tasks.

Edit COMPONENTS here to find or change a component's public name and implementation.
Package initializers call register_components for their own group.
"""
from importlib import import_module

COMPONENTS = {
    "models": {
        "sega3d_t5_nv2": ("lavis.models.sega3d_models.sega3d_t5_nv2", "SEGA3DT5"),
        "sega3d_t5_nv3": ("lavis.models.sega3d_models.sega3d_t5_nv3", "SEGA3DT5"),
    },
    "datasets": {
        "sega3d_scanrefer": ("lavis.datasets.builders.sega3d_builder", "SEGA3DScanReferBuilder"),
        "sega3d_reasoning_scannet": ("lavis.datasets.builders.sega3d_builder", "SEGA3DScanNetBuilder"),
        "sega3d_reasoning_matterport": ("lavis.datasets.builders.sega3d_builder", "SEGA3DMatterportBuilder"),
    },
    "tasks": {
        "sega3d_seg": ("lavis.tasks.sega3d_seg_task", "SEGA3DSegTask"),
    },
}

def register_components(group=None):
    """Import and register components; repeated calls preserve existing classes."""
    from lavis.common.registry import registry

    registration_methods = {
        "models": (registry.get_model_class, registry.register_model),
        "datasets": (registry.get_builder_class, registry.register_builder),
        "tasks": (registry.get_task_class, registry.register_task),
    }
    groups = tuple(COMPONENTS) if group is None else (group,)
    if any(name not in COMPONENTS for name in groups):
        raise ValueError(f"Unknown SEGA3D component group: {group}")
    for name in groups:
        lookup, register = registration_methods[name]
        for key, (module_name, class_name) in COMPONENTS[name].items():
            component = getattr(import_module(module_name), class_name)
            existing = lookup(key)
            if existing is None:
                register(key)(component)
            elif existing is not component:
                raise RuntimeError(f"SEGA3D component name already in use: {key}")
    return registry
