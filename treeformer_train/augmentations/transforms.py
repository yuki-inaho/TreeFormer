from __future__ import annotations

import os
import warnings
from collections.abc import Iterable
from typing import Protocol

import cv2
import numpy as np

from treeformer_train.augmentations.sample import GraphSample, as_float_image


class SampleTransform(Protocol):
    def __call__(self, sample: GraphSample, rng: np.random.Generator) -> GraphSample: ...


class ComposeGraphTransforms:
    def __init__(self, transforms: Iterable[SampleTransform], *, seed: int | None = None) -> None:
        self.transforms = list(transforms)
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample: GraphSample) -> GraphSample:
        output = sample.copy()
        for transform in self.transforms:
            output = transform(output, self.rng)
        return output


class OpenCVPhotometricTransform:
    """Portable image-only photometric augmentation that leaves graph data unchanged."""

    def __init__(
        self,
        *,
        p: float = 0.8,
        brightness_contrast_p: float = 0.35,
        hsv_p: float = 0.25,
        gamma_p: float = 0.25,
        noise_p: float = 0.2,
        blur_p: float = 0.15,
    ) -> None:
        self.p = p
        self.brightness_contrast_p = brightness_contrast_p
        self.hsv_p = hsv_p
        self.gamma_p = gamma_p
        self.noise_p = noise_p
        self.blur_p = blur_p

    def __call__(self, sample: GraphSample, rng: np.random.Generator) -> GraphSample:
        if not _coin(self.p, rng):
            return sample

        image = as_float_image(sample.image)
        if _coin(self.brightness_contrast_p, rng):
            alpha = rng.uniform(0.82, 1.22)
            beta = rng.uniform(-0.12, 0.12)
            image = np.clip(image * alpha + beta, 0.0, 1.0)

        if image.ndim == 3 and image.shape[2] >= 3 and _coin(self.hsv_p, rng):
            rgb_uint8 = _to_uint8(image[:, :, :3])
            hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-8.0, 8.0)) % 180.0
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.85, 1.18), 0.0, 255.0)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.88, 1.16), 0.0, 255.0)
            image[:, :, :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

        if _coin(self.gamma_p, rng):
            gamma = rng.uniform(0.78, 1.35)
            image = np.clip(np.power(image, gamma), 0.0, 1.0)

        if _coin(self.noise_p, rng):
            sigma = rng.uniform(0.004, 0.025)
            image = np.clip(image + rng.normal(0.0, sigma, image.shape).astype(np.float32), 0.0, 1.0)

        if _coin(self.blur_p, rng):
            kernel_size = int(rng.choice(np.array([3, 5], dtype=np.int32)))
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)

        return GraphSample(image=as_float_image(image), nodes=sample.nodes.copy(), edges=sample.edges.copy())


class AlbumentationsXPhotometricTransform:
    """AlbumentationsX adapter isolated behind the graph transform protocol."""

    def __init__(
        self,
        *,
        p: float = 0.8,
        brightness_contrast_p: float = 0.35,
        hsv_p: float = 0.25,
        gamma_p: float = 0.25,
        noise_p: float = 0.2,
        blur_p: float = 0.15,
        allow_fallback: bool = True,
    ) -> None:
        self.p = p
        self.brightness_contrast_p = brightness_contrast_p
        self.hsv_p = hsv_p
        self.gamma_p = gamma_p
        self.noise_p = noise_p
        self.blur_p = blur_p
        self.allow_fallback = allow_fallback
        self._pipeline = None
        self._fallback: OpenCVPhotometricTransform | None = None

    def __call__(self, sample: GraphSample, rng: np.random.Generator) -> GraphSample:
        if not _coin(self.p, rng):
            return sample
        pipeline = self._get_pipeline()
        if pipeline is None:
            return self._get_fallback()(sample, rng)

        image = _to_uint8(sample.image)
        augmented = pipeline(image=image)["image"].astype(np.float32) / 255.0
        return GraphSample(image=as_float_image(augmented), nodes=sample.nodes.copy(), edges=sample.edges.copy())

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            os.environ.setdefault("ALBUMENTATIONS_NO_TELEMETRY", "1")
            import albumentations as A

            self._pipeline = A.Compose(
                [
                    A.RandomBrightnessContrast(p=self.brightness_contrast_p),
                    A.HueSaturationValue(p=self.hsv_p),
                    A.RandomGamma(p=self.gamma_p),
                    A.GaussNoise(p=self.noise_p),
                    A.OneOf([A.GaussianBlur(), A.MotionBlur()], p=self.blur_p),
                ]
            )
            return self._pipeline
        except Exception as exc:
            if not self.allow_fallback:
                raise RuntimeError(
                    "AlbumentationsX photometric augmentation was requested, but the albumentations module "
                    "could not be initialized. Install the albumentationsx package or use backend=opencv."
                ) from exc
            warnings.warn(
                f"AlbumentationsX unavailable; falling back to OpenCV photometric augmentation: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def _get_fallback(self) -> OpenCVPhotometricTransform:
        if self._fallback is None:
            self._fallback = OpenCVPhotometricTransform(
                p=1.0,
                brightness_contrast_p=self.brightness_contrast_p,
                hsv_p=self.hsv_p,
                gamma_p=self.gamma_p,
                noise_p=self.noise_p,
                blur_p=self.blur_p,
            )
        return self._fallback


class RandomAffineGraphTransform:
    """Apply a single affine transform to both image pixels and normalized graph nodes."""

    def __init__(
        self,
        *,
        p: float = 0.35,
        max_rotate_deg: float = 8.0,
        max_translate_frac: float = 0.035,
        scale_range: tuple[float, float] = (0.96, 1.04),
        keep_all_nodes_inside: bool = True,
    ) -> None:
        self.p = p
        self.max_rotate_deg = max_rotate_deg
        self.max_translate_frac = max_translate_frac
        self.scale_range = scale_range
        self.keep_all_nodes_inside = keep_all_nodes_inside

    def __call__(self, sample: GraphSample, rng: np.random.Generator) -> GraphSample:
        if not _coin(self.p, rng):
            return sample
        image = as_float_image(sample.image)
        height, width = image.shape[:2]
        angle = float(rng.uniform(-self.max_rotate_deg, self.max_rotate_deg))
        scale = float(rng.uniform(self.scale_range[0], self.scale_range[1]))
        translate_x = float(rng.uniform(-self.max_translate_frac, self.max_translate_frac) * width)
        translate_y = float(rng.uniform(-self.max_translate_frac, self.max_translate_frac) * height)
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale).astype(np.float32)
        matrix[0, 2] += translate_x
        matrix[1, 2] += translate_y

        nodes = _transform_nodes_affine(sample.nodes, matrix, width=width, height=height)
        if self.keep_all_nodes_inside and _has_outside_nodes(nodes):
            return sample

        image = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return GraphSample(image=as_float_image(image), nodes=np.clip(nodes, 0.0, 1.0), edges=sample.edges.copy())


class ElasticGraphTransform:
    """Low-frequency deformation that updates graph nodes using the same displacement field."""

    def __init__(
        self,
        *,
        p: float = 0.2,
        alpha_frac: float = 0.018,
        sigma_frac: float = 0.035,
        grid_size: int = 4,
        keep_all_nodes_inside: bool = True,
    ) -> None:
        self.p = p
        self.alpha_frac = alpha_frac
        self.sigma_frac = sigma_frac
        self.grid_size = grid_size
        self.keep_all_nodes_inside = keep_all_nodes_inside

    def __call__(self, sample: GraphSample, rng: np.random.Generator) -> GraphSample:
        if not _coin(self.p, rng):
            return sample
        image = as_float_image(sample.image)
        height, width = image.shape[:2]
        dx, dy = self._make_displacement(height, width, rng)

        nodes = _transform_nodes_displacement(sample.nodes, dx, dy, width=width, height=height)
        if self.keep_all_nodes_inside and _has_outside_nodes(nodes):
            return sample

        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        remapped = cv2.remap(
            image,
            (grid_x - dx).astype(np.float32),
            (grid_y - dy).astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return GraphSample(image=as_float_image(remapped), nodes=np.clip(nodes, 0.0, 1.0), edges=sample.edges.copy())

    def _make_displacement(
        self,
        height: int,
        width: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        grid = max(2, int(self.grid_size))
        dx = rng.uniform(-1.0, 1.0, size=(grid, grid)).astype(np.float32)
        dy = rng.uniform(-1.0, 1.0, size=(grid, grid)).astype(np.float32)
        dx = cv2.resize(dx, (width, height), interpolation=cv2.INTER_CUBIC)
        dy = cv2.resize(dy, (width, height), interpolation=cv2.INTER_CUBIC)
        sigma = max(1.0, self.sigma_frac * max(height, width))
        dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=sigma, sigmaY=sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=sigma, sigmaY=sigma)
        dx = _normalize_field(dx) * (self.alpha_frac * width)
        dy = _normalize_field(dy) * (self.alpha_frac * height)
        return dx.astype(np.float32), dy.astype(np.float32)


def _coin(probability: float, rng: np.random.Generator) -> bool:
    return float(probability) >= 1.0 or bool(rng.random() < float(probability))


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return (as_float_image(image) * 255.0).round().astype(np.uint8)


def _transform_nodes_affine(nodes: np.ndarray, matrix: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if nodes.size == 0:
        return nodes.copy()
    points = nodes.astype(np.float32) * np.array([width, height], dtype=np.float32)
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    transformed = homogeneous @ matrix.T
    return transformed / np.array([width, height], dtype=np.float32)


def _transform_nodes_displacement(
    nodes: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    if nodes.size == 0:
        return nodes.copy()
    points = nodes.astype(np.float32) * np.array([width, height], dtype=np.float32)
    offset_x = _sample_field(dx, points[:, 0], points[:, 1])
    offset_y = _sample_field(dy, points[:, 0], points[:, 1])
    transformed = points + np.stack([offset_x, offset_y], axis=1)
    return transformed / np.array([width, height], dtype=np.float32)


def _sample_field(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = field.shape
    x0 = np.floor(np.clip(x, 0, width - 1)).astype(np.int64)
    y0 = np.floor(np.clip(y, 0, height - 1)).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = np.clip(x - x0, 0.0, 1.0).astype(np.float32)
    wy = np.clip(y - y0, 0.0, 1.0).astype(np.float32)

    top = field[y0, x0] * (1.0 - wx) + field[y0, x1] * wx
    bottom = field[y1, x0] * (1.0 - wx) + field[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def _normalize_field(field: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(field)))
    if scale <= 1e-6:
        return np.zeros_like(field, dtype=np.float32)
    return (field / scale).astype(np.float32)


def _has_outside_nodes(nodes: np.ndarray) -> bool:
    return bool(nodes.size and ((nodes < 0.0).any() or (nodes > 1.0).any()))
