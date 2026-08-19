import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from transformers import AutoImageProcessor, AutoModel

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ============================================================
# CONFIGURATION
# ============================================================

# DINOv2 Small is a good starting point for an 8 GB RAM / 4 GB
# NVIDIA GTX laptop. It produces image-only visual embeddings.
MODEL_NAME = os.getenv(
    "DINOV2_MODEL",
    "facebook/dinov2-small",
)

# YOLO-World is used because the standard COCO YOLO model does
# not have a "shoe" class. This model can be prompted with
# footwear classes.
YOLO_WORLD_MODEL = os.getenv(
    "YOLO_WORLD_MODEL",
    "yolov8s-world.pt",
)

FOOTWEAR_CLASSES = [
    "shoe",
    "sneaker",
    "boot",
    "sandal",
    "slipper",
    "loafer",
    "formal shoe",
    "sports shoe",
]

IMAGE_SIZE = 224
YOLO_CONFIDENCE = float(
    os.getenv("YOLO_CONFIDENCE", "0.12")
)

# Keep YOLO on CPU by default to protect 4 GB VRAM.
# DINOv2 gets CUDA if available.
YOLO_DEVICE = os.getenv(
    "SHOE_YOLO_DEVICE",
    "cpu",
)

USE_YOLO = os.getenv(
    "USE_YOLO",
    "true",
).lower() not in {"0", "false", "no"}

CROP_PADDING = 0.10


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DINOv2 SHOE EMBEDDING INITIALIZATION")
print("=" * 60)
print(f"Model : {MODEL_NAME}")
print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")


# ============================================================
# DINOv2
# ============================================================

print("Loading DINOv2...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()

print("DINOv2 loaded successfully.")


# ============================================================
# YOLO-WORLD LAZY LOADING
# ============================================================

_yolo_model = None
_yolo_failed = False


def get_yolo_model():
    """
    Load YOLO-World only when an image actually needs processing.
    This keeps application startup lighter and allows a safe
    fallback if YOLO cannot be loaded.
    """

    global _yolo_model
    global _yolo_failed

    if not USE_YOLO:
        return None

    if _yolo_failed:
        return None

    if _yolo_model is not None:
        return _yolo_model

    if YOLO is None:
        print(
            "Ultralytics is not installed. "
            "YOLO cropping will use fallback."
        )
        _yolo_failed = True
        return None

    try:
        print(
            f"Loading YOLO-World: {YOLO_WORLD_MODEL}"
        )

        _yolo_model = YOLO(
            YOLO_WORLD_MODEL
        )

        # YOLO-World supports custom text classes.
        if hasattr(_yolo_model, "set_classes"):
            _yolo_model.set_classes(
                FOOTWEAR_CLASSES
            )

        print("YOLO-World loaded successfully.")

        return _yolo_model

    except Exception as error:
        print(
            f"YOLO loading failed: {error}"
        )
        print(
            "Continuing with OpenCV/original-image fallback."
        )
        _yolo_failed = True
        return None


# ============================================================
# BASIC IMAGE HELPERS
# ============================================================

def ensure_rgb(
    image: Image.Image,
) -> Image.Image:
    return image.convert("RGB")


def correct_orientation(
    image: Image.Image,
) -> Image.Image:
    try:
        return ImageOps.exif_transpose(
            image
        ).convert("RGB")
    except Exception:
        return image.convert("RGB")


def resize_for_detection(
    image: Image.Image,
    max_dimension: int = 960,
) -> Tuple[Image.Image, float]:
    """
    Resize only for detection. Returns the resized image and
    the scale used to map coordinates back to the original.
    """

    width, height = image.size

    scale = min(
        1.0,
        max_dimension / max(width, height),
    )

    if scale == 1.0:
        return image, 1.0

    resized = image.resize(
        (
            max(1, int(width * scale)),
            max(1, int(height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )

    return resized, scale


# ============================================================
# YOLO SHOE DETECTION
# ============================================================

def detect_footwear_crop(
    image: Image.Image,
) -> Optional[Image.Image]:
    """
    Detect footwear using YOLO-World.

    If detection fails or is uncertain, return None so that the
    caller can use a safe fallback instead of rejecting the query.
    """

    yolo = get_yolo_model()

    if yolo is None:
        return None

    image = correct_orientation(image)

    detection_image, scale = resize_for_detection(
        image
    )

    try:
        results = yolo.predict(
            source=np.asarray(detection_image),
            conf=YOLO_CONFIDENCE,
            iou=0.45,
            imgsz=640,
            device=YOLO_DEVICE,
            verbose=False,
        )

        if not results:
            return None

        result = results[0]

        if result.boxes is None:
            return None

        boxes = result.boxes

        if len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()

        if xyxy.size == 0:
            return None

        # Keep detections with reasonable confidence.
        keep = confs >= YOLO_CONFIDENCE

        xyxy = xyxy[keep]
        confs = confs[keep]

        if len(xyxy) == 0:
            return None

        # For a pair of shoes, combine nearby footwear detections.
        # Limiting to the strongest few avoids accidental huge crops.
        order = np.argsort(-confs)[:4]
        selected = xyxy[order]

        xmin = float(np.min(selected[:, 0]))
        ymin = float(np.min(selected[:, 1]))
        xmax = float(np.max(selected[:, 2]))
        ymax = float(np.max(selected[:, 3]))

        # Map detection coordinates back to original image.
        xmin /= scale
        ymin /= scale
        xmax /= scale
        ymax /= scale

        width, height = image.size

        box_width = xmax - xmin
        box_height = ymax - ymin

        if box_width <= 5 or box_height <= 5:
            return None

        pad_x = box_width * CROP_PADDING
        pad_y = box_height * CROP_PADDING

        xmin = max(0, int(xmin - pad_x))
        ymin = max(0, int(ymin - pad_y))
        xmax = min(
            width,
            int(xmax + pad_x),
        )
        ymax = min(
            height,
            int(ymax + pad_y),
        )

        cropped = image.crop(
            (xmin, ymin, xmax, ymax)
        )

        if (
            cropped.width < 32
            or cropped.height < 32
        ):
            return None

        print(
            "YOLO crop: "
            f"({xmin}, {ymin}, {xmax}, {ymax})"
        )

        return cropped

    except Exception as error:
        print(
            f"YOLO detection failed: {error}"
        )
        return None


# ============================================================
# OPENCV FALLBACK CROP
# ============================================================

def opencv_fallback_crop(
    image: Image.Image,
) -> Image.Image:
    """
    Lightweight fallback when YOLO cannot detect footwear.

    It tries foreground segmentation but is conservative:
    if the result looks unreliable, the original image is returned.
    """

    image = correct_orientation(image)

    if min(image.size) < 40:
        return image

    rgb = np.asarray(image)
    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    h, w = bgr.shape[:2]

    scale = min(
        1.0,
        800 / max(h, w),
    )

    if scale < 1:
        small = cv2.resize(
            bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = bgr

    sh, sw = small.shape[:2]

    mask = np.zeros(
        (sh, sw),
        np.uint8,
    )

    mx = max(2, int(sw * 0.08))
    my = max(2, int(sh * 0.08))

    rect = (
        mx,
        my,
        max(1, sw - 2 * mx),
        max(1, sh - 2 * my),
    )

    bgd = np.zeros(
        (1, 65),
        np.float64,
    )
    fgd = np.zeros(
        (1, 65),
        np.float64,
    )

    try:
        cv2.grabCut(
            small,
            mask,
            rect,
            bgd,
            fgd,
            2,
            cv2.GC_INIT_WITH_RECT,
        )

        foreground = np.where(
            (mask == cv2.GC_FGD)
            | (mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)

        kernel = np.ones(
            (5, 5),
            np.uint8,
        )

        foreground = cv2.morphologyEx(
            foreground,
            cv2.MORPH_OPEN,
            kernel,
        )

        foreground = cv2.morphologyEx(
            foreground,
            cv2.MORPH_CLOSE,
            kernel,
        )

        contours, _ = cv2.findContours(
            foreground,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return image

        image_area = sw * sh

        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area <= 0:
                continue

            ratio = area / image_area

            if ratio < 0.03 or ratio > 0.90:
                continue

            x, y, cw, ch = cv2.boundingRect(
                contour
            )

            candidates.append(
                (area, x, y, cw, ch)
            )

        if not candidates:
            return image

        candidates.sort(
            reverse=True,
            key=lambda item: item[0],
        )

        _, x, y, cw, ch = candidates[0]

        pad_x = int(cw * 0.10)
        pad_y = int(ch * 0.10)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(sw, x + cw + pad_x)
        y2 = min(sh, y + ch + pad_y)

        crop = small[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:
            return image

        crop = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB,
        )

        cropped = Image.fromarray(
            crop
        )

        if scale < 1:
            cropped = cropped.resize(
                (
                    max(
                        1,
                        int(cropped.width / scale),
                    ),
                    max(
                        1,
                        int(cropped.height / scale),
                    ),
                ),
                Image.Resampling.LANCZOS,
            )

        return cropped

    except Exception as error:
        print(
            f"OpenCV crop fallback failed: {error}"
        )
        return image


def crop_shoe(
    image: Image.Image,
) -> Image.Image:
    """
    Main crop pipeline:
    YOLO-World -> OpenCV fallback -> original image.
    """

    image = correct_orientation(image)

    detected = detect_footwear_crop(
        image
    )

    if detected is not None:
        return detected

    fallback = opencv_fallback_crop(
        image
    )

    return fallback


# ============================================================
# ROBUST IMAGE VIEWS
# ============================================================

def prepare_views(
    image: Image.Image,
) -> List[Image.Image]:
    """
    Create a small set of views that improve robustness to:
    blur, compression, brightness changes, and phone-camera
    differences.

    We keep the number of views small for the laptop.
    """

    image = correct_orientation(image)

    views = []

    # View 1: normal
    base = ImageOps.contain(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        (255, 255, 255),
    )

    x = (IMAGE_SIZE - base.width) // 2
    y = (IMAGE_SIZE - base.height) // 2

    canvas.paste(
        base,
        (x, y),
    )

    views.append(canvas)

    # View 2: mild contrast/sharpness
    enhanced = ImageEnhance.Contrast(
        canvas
    ).enhance(1.07)

    enhanced = ImageEnhance.Sharpness(
        enhanced
    ).enhance(1.10)

    views.append(enhanced)

    # View 3: mild denoising
    views.append(
        canvas.filter(
            ImageFilter.MedianFilter(3)
        )
    )

    return views


# ============================================================
# DINOv2 ENCODING
# ============================================================

def encode_images(
    images: List[Image.Image],
) -> np.ndarray:
    """
    Encode multiple views and return one normalized vector.

    DINOv2-Small has a compact embedding size, which is suitable
    for FAISS and the target laptop.
    """

    inputs = processor(
        images=[
            correct_orientation(image)
            for image in images
        ],
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model(
            **inputs
        )

        if hasattr(
            outputs,
            "pooler_output",
        ) and outputs.pooler_output is not None:

            features = outputs.pooler_output

        else:

            # CLS token
            features = (
                outputs.last_hidden_state[:, 0]
            )

        features = torch.nn.functional.normalize(
            features,
            p=2,
            dim=-1,
        )

        # Average normalized views and normalize again.
        combined = features.mean(
            dim=0,
            keepdim=True,
        )

        combined = torch.nn.functional.normalize(
            combined,
            p=2,
            dim=-1,
        )

    return (
        combined
        .cpu()
        .numpy()[0]
        .astype("float32")
    )


# ============================================================
# PUBLIC API
# ============================================================

def get_image_embedding_from_pil(
    image: Image.Image,
) -> np.ndarray:
    """
    Full image -> crop -> robust views -> DINOv2 embedding pipeline.
    """

    image = correct_orientation(
        image
    )

    cropped = crop_shoe(
        image
    )

    views = prepare_views(
        cropped
    )

    return encode_images(
        views
    )


def get_image_embedding(
    image_path: str,
) -> np.ndarray:

    if not os.path.isfile(
        image_path
    ):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(
        image_path
    ) as image:

        return get_image_embedding_from_pil(
            image
        )
