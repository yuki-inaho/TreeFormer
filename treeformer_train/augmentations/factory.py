from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from treeformer_train.augmentations.transforms import (
    AlbumentationsXPhotometricTransform,
    ComposeGraphTransforms,
    ElasticGraphTransform,
    OpenCVPhotometricTransform,
    RandomAffineGraphTransform,
)


def build_graph_augmentation(config: Any) -> ComposeGraphTransforms | None:
    if config is None or not bool(_get(config, "enabled", False)):
        return None

    transforms = []
    photometric = _get(config, "photometric", None)
    if photometric is not None and bool(_get(photometric, "enabled", False)):
        backend = str(_get(photometric, "backend", "opencv")).lower()
        common = {
            "p": float(_get(photometric, "p", 0.8)),
            "brightness_contrast_p": float(_get(photometric, "brightness_contrast_p", 0.35)),
            "hsv_p": float(_get(photometric, "hsv_p", 0.25)),
            "gamma_p": float(_get(photometric, "gamma_p", 0.25)),
            "noise_p": float(_get(photometric, "noise_p", 0.2)),
            "blur_p": float(_get(photometric, "blur_p", 0.15)),
        }
        if backend in {"albumentationsx", "albumentations", "auto"}:
            transforms.append(
                AlbumentationsXPhotometricTransform(
                    **common,
                    allow_fallback=bool(_get(photometric, "allow_fallback", True)),
                )
            )
        elif backend == "opencv":
            transforms.append(OpenCVPhotometricTransform(**common))
        else:
            raise ValueError(f"unsupported photometric augmentation backend: {backend!r}")

    affine = _get(config, "affine", None)
    if affine is not None and bool(_get(affine, "enabled", False)):
        transforms.append(
            RandomAffineGraphTransform(
                p=float(_get(affine, "p", 0.35)),
                max_rotate_deg=float(_get(affine, "max_rotate_deg", 8.0)),
                max_translate_frac=float(_get(affine, "max_translate_frac", 0.035)),
                scale_range=_tuple2(_get(affine, "scale_range", (0.96, 1.04))),
                keep_all_nodes_inside=bool(_get(affine, "keep_all_nodes_inside", True)),
            )
        )

    elastic = _get(config, "elastic", None)
    if elastic is not None and bool(_get(elastic, "enabled", False)):
        transforms.append(
            ElasticGraphTransform(
                p=float(_get(elastic, "p", 0.2)),
                alpha_frac=float(_get(elastic, "alpha_frac", 0.018)),
                sigma_frac=float(_get(elastic, "sigma_frac", 0.035)),
                grid_size=int(_get(elastic, "grid_size", 4)),
                keep_all_nodes_inside=bool(_get(elastic, "keep_all_nodes_inside", True)),
            )
        )

    if not transforms:
        return None
    seed = _get(config, "seed", None)
    return ComposeGraphTransforms(transforms, seed=None if seed is None else int(seed))


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _tuple2(value: Any) -> tuple[float, float]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ValueError(f"expected two values, got {value!r}")
    return float(parts[0]), float(parts[1])
