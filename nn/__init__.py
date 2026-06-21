"""Neural road-mask training (D-LinkNet) for sat2graph."""

__all__ = [
    "CUSTOM_OBJECTS",
    "bce_dice_loss",
    "create_dlinknet",
    "dice_coefficient",
    "discover_pairs",
    "load_split_arrays",
    "split_by_tile_id",
]


def __getattr__(name: str):
    if name in ("CUSTOM_OBJECTS", "bce_dice_loss", "create_dlinknet", "dice_coefficient"):
        from . import dlinknet_model

        return getattr(dlinknet_model, name)
    if name in ("discover_pairs", "load_split_arrays", "split_by_tile_id"):
        from . import data

        return getattr(data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
