from pathlib import Path
import logging

import numpy as np
from PIL import Image
import yaml


LOG_PATH = Path("logs")
LOG_PATH.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str, log_file: str | Path) -> logging.Logger:
    """
    Create and return a logger writing to a file.
    Prevent duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


QUALITY_LOGGER = setup_logger("image_quality", LOG_PATH / "image_quality.log")

PREPROCESSING_LOGGER = setup_logger(
    "image_preprocessing", LOG_PATH / "image_preprocessing.log"
)


def load_image_preprocessing_config(config_path: str | Path) -> dict:
    """
    Load preprocessing configuration from YAML.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        PREPROCESSING_LOGGER.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        PREPROCESSING_LOGGER.exception(f"Failed to load config file {config_path}")
        raise


def compute_quality_report(
    image_array: np.ndarray,
    image_id: str | None = None,
) -> dict:
    """
    Compute simple image quality metrics and log issues.
    """
    gray = image_array.mean(axis=2)

    brightness = gray.mean()
    contrast = gray.std()

    quality = {
        "brightness": float(brightness),
        "contrast": float(contrast),
        "near_white": bool(brightness > 245),
        "low_contrast": bool(contrast < 15),
        "extremely_dark": bool(brightness < 20),
    }

    label = image_id if image_id else "<unknown_image>"

    if quality["near_white"]:
        QUALITY_LOGGER.warning(f"{label} near white image")

    if quality["low_contrast"]:
        QUALITY_LOGGER.warning(f"{label} low contrast")

    if quality["extremely_dark"]:
        QUALITY_LOGGER.warning(f"{label} extremely dark")

    return quality


def preprocess_image(
    image: str | Path | Image.Image,
    image_id: str | None = None,
    config_path: str | Path = "image_preprocessing_config.yaml",
) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    Deterministic preprocessing of a single image.

    Parameters
    ----------
    image : str | Path | PIL.Image.Image
        Input image path or already loaded PIL image.
    image_id : str | None
        Optional image identifier for logging.
    config_path : str | Path
        Path to YAML preprocessing config.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, dict]
        Preprocessed image array, and optionally a quality report.
    """
    try:
        config = load_image_preprocessing_config(config_path)

        preprocessing_config = config.get("preprocessing", {})
        quality_config = config.get("quality", {})

        output_size = preprocessing_config.get("output_size", [224, 224])
        convert_rgb = preprocessing_config.get("convert_rgb", True)
        normalization = preprocessing_config.get("normalization", True)
        normalize_mode = preprocessing_config.get("normalize_mode", "zero_one")
        keep_aspect_ratio = preprocessing_config.get("keep_aspect_ratio", False)
        resample_method = preprocessing_config.get("resample_method", "bilinear")
        compute_quality = quality_config.get("compute_quality_report", False)

        if not isinstance(output_size, (list, tuple)) or len(output_size) != 2:
            raise ValueError("output_size must be a list or tuple like [224, 224]")

        width, height = int(output_size[0]), int(output_size[1])

        if width <= 0 or height <= 0:
            raise ValueError("output_size values must be positive integers.")

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        if isinstance(image, (str, Path)):
            image_path = Path(image)

            if image_id is None:
                image_id = image_path.stem

            try:
                image = Image.open(image_path)
            except Exception:
                PREPROCESSING_LOGGER.exception(f"Failed to load image {image_path}")
                raise

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a path or PIL.Image.Image")

        if convert_rgb:
            image = image.convert("RGB")

        image_array = np.array(image)

        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image with shape (H, W, 3), got shape {image_array.shape}"
            )

        # ----------------------------------------------------
        # Quality report on raw image
        # ----------------------------------------------------

        quality_report = None
        if compute_quality:
            quality_report = compute_quality_report(
                image_array,
                image_id=image_id,
            )

        # ----------------------------------------------------
        # Resizing
        # ----------------------------------------------------

        resample_map = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }

        if resample_method not in resample_map:
            raise ValueError(
                f"Unsupported resample_method '{resample_method}'. "
                f"Choose from {list(resample_map.keys())}."
            )

        resample = resample_map[resample_method]

        if keep_aspect_ratio:
            image.thumbnail((width, height), resample=resample)

            background = Image.new(
                "RGB",
                (width, height),
                (255, 255, 255),
            )

            offset_x = (width - image.width) // 2
            offset_y = (height - image.height) // 2

            background.paste(image, (offset_x, offset_y))
            image = background
        else:
            image = image.resize((width, height), resample=resample)

        image_array = np.array(image)

        # ----------------------------------------------------
        # Normalization
        # ----------------------------------------------------

        if normalization:
            image_array = image_array.astype(np.float32) / 255.0

            if normalize_mode == "imagenet":
                mean = np.array(
                    [0.485, 0.456, 0.406],
                    dtype=np.float32,
                )
                std = np.array(
                    [0.229, 0.224, 0.225],
                    dtype=np.float32,
                )
                image_array = (image_array - mean) / std

            elif normalize_mode != "zero_one":
                raise ValueError("normalize_mode must be 'zero_one' or 'imagenet'")

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        if compute_quality:
            return image_array, quality_report

        return image_array

    except Exception:
        PREPROCESSING_LOGGER.exception(f"Preprocessing failed for image_id={image_id}")
        raise
