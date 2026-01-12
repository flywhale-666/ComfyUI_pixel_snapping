import os
import time

import cv2
import numpy as np
import torch

_MODEL_CACHE = {}


def _list_yolo_models():
    try:
        import folder_paths  # type: ignore

        models_dir = folder_paths.models_dir
        yolo_dir = os.path.join(models_dir, "yolo")
        if not os.path.isdir(yolo_dir):
            return []
        files = [
            f for f in os.listdir(yolo_dir)
            if os.path.isfile(os.path.join(yolo_dir, f)) and f.lower().endswith(".pt")
        ]
        return sorted(files)
    except Exception:
        return []


def _resolve_model_path(model_path: str) -> str:
    if not model_path:
        return model_path

    if os.path.isabs(model_path) and os.path.exists(model_path):
        return model_path

    try:
        import folder_paths  # type: ignore

        models_dir = folder_paths.models_dir
        candidate = os.path.join(models_dir, "yolo", model_path)
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass

    return model_path


def _load_model(model_path: str):
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise ImportError(
            "ultralytics is required for face detection. Install it with `pip install ultralytics`."
        ) from exc

    resolved_path = _resolve_model_path(model_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Model not found: {resolved_path}")

    device = 0 if torch.cuda.is_available() else "cpu"
    cache_key = (resolved_path, device)
    model = _MODEL_CACHE.get(cache_key)
    if model is None:
        model = YOLO(resolved_path)
        _MODEL_CACHE[cache_key] = model

    return model, device, resolved_path


class FaceHandCropNode:
    @classmethod
    def INPUT_TYPES(cls):
        model_choices = _list_yolo_models()
        if not model_choices:
            model_choices = ["face_yolov8m.pt"]
        return {
            "required": {
                "image": ("IMAGE",),
                "model_path": (model_choices,),
                "max_faces": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number",
                }),
                "start_face": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number",
                }),
                "padding_ratio": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                }),
                "maskpadding_ratio": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                }),
                "face_position": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                }),
                "enable_resize": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Enabled",
                    "label_off": "Disabled"
                }),
                "target_width": ("INT", {
                    "default": 512,
                    "min": 0,
                    "max": 16384,
                    "step": 1,
                    "display": "number",
                }),
                "target_height": ("INT", {
                    "default": 512,
                    "min": 0,
                    "max": 16384,
                    "step": 1,
                    "display": "number",
                }),
                "padding_mode": (["black", "white", "none"],),
                "size_multiple": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 999,
                    "step": 1,
                    "display": "number",
                }),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "CROP_INFO", "INT")
    RETURN_NAMES = ("cropped_image", "face_mask", "crop_mask_full", "crop_info", "face_count")
    FUNCTION = "detect_and_crop"
    CATEGORY = "image/transform"

    def detect_and_crop(
        self,
        image,
        model_path,
        max_faces,
        start_face,
        padding_ratio,
        maskpadding_ratio,
        face_position,
        target_width,
        target_height,
        enable_resize,
        padding_mode,
        size_multiple,
        mask=None,
    ):
        t0 = time.perf_counter()
        img = image[0].cpu().numpy()
        img_h, img_w = img.shape[:2]

        if img.ndim == 2:
            img_rgb = np.repeat(img[:, :, None], 3, axis=2)
        elif img.shape[2] == 1:
            img_rgb = np.repeat(img, 3, axis=2)
        else:
            img_rgb = img[:, :, :3]

        img_bgr = cv2.cvtColor((img_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        t1 = time.perf_counter()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available, abort!")
        model, device, _ = _load_model(model_path)
        t2 = time.perf_counter()
        torch.backends.cudnn.benchmark = False
        use_half = device != "cpu"
        t3 = time.perf_counter()
        results = model.predict(
            source=img_bgr,
            device=device,
            conf=0.25,
            half=use_half,
            verbose=False
        )
        t4 = time.perf_counter()

        boxes = None
        if results and len(results) > 0:
            boxes = results[0].boxes

        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            empty_face_mask = torch.zeros((1, img_h, img_w), dtype=torch.float32)
            full_crop_mask = torch.ones((1, img_h, img_w), dtype=torch.float32)
            crop_info = {
                "original_size": (img_w, img_h),
                "crop_region": (0, 0, img_w, img_h),
                "cropped_size": (img_w, img_h),
                "padding": {"top": 0, "bottom": 0, "left": 0, "right": 0},
            }
            print(
                "FaceHandCrop timing:",
                f"pre={(t1 - t0):.3f}s, load={(t2 - t1):.3f}s, infer={(t4 - t3):.3f}s, total={(t4 - t0):.3f}s",
            )
            return (image, empty_face_mask, full_crop_mask, crop_info, 0)

        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        order = np.argsort(-areas)
        xyxy = xyxy[order]

        max_faces = max(1, min(int(max_faces), 100))
        face_count = int(min(len(xyxy), max_faces))
        xyxy = xyxy[:face_count]

        start_face = max(1, int(start_face))
        face_index = (start_face - 1) % face_count
        x1, y1, x2, y2 = xyxy[face_index].tolist()

        x1 = float(np.clip(x1, 0, img_w - 1))
        y1 = float(np.clip(y1, 0, img_h - 1))
        x2 = float(np.clip(x2, 1, img_w))
        y2 = float(np.clip(y2, 1, img_h))

        bbox_w = max(1.0, x2 - x1)
        bbox_h = max(1.0, y2 - y1)

        padding_ratio = float(np.clip(padding_ratio, 0.5, 10.0))
        maskpadding_ratio = float(np.clip(maskpadding_ratio, 0.5, 10.0))
        face_position = float(np.clip(face_position, 0.0, 1.0))

        crop_w = max(1, int(round(bbox_w * padding_ratio)))
        crop_h = max(1, int(round(bbox_h * padding_ratio)))
        mask_w = max(1, int(round(bbox_w * maskpadding_ratio)))
        mask_h = max(1, int(round(bbox_h * maskpadding_ratio)))

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        crop_x_min = int(round(cx - 0.5 * crop_w))
        crop_y_min = int(round(cy - face_position * crop_h))
        crop_x_max = crop_x_min + crop_w
        crop_y_max = crop_y_min + crop_h

        pad_left = pad_right = pad_top = pad_bottom = 0

        if padding_mode == "none":
            if crop_w <= img_w:
                crop_x_min = min(max(crop_x_min, 0), img_w - crop_w)
                crop_x_max = crop_x_min + crop_w
            else:
                crop_x_min = 0
                crop_x_max = img_w
                crop_w = img_w

            if crop_h <= img_h:
                crop_y_min = min(max(crop_y_min, 0), img_h - crop_h)
                crop_y_max = crop_y_min + crop_h
            else:
                crop_y_min = 0
                crop_y_max = img_h
                crop_h = img_h
        crop_w = crop_x_max - crop_x_min
        crop_h = crop_y_max - crop_y_min

        # 【修复】先计算遮罩的初始位置
        mask_x_min = int(round(cx - 0.5 * mask_w))
        mask_y_min = int(round(cy - face_position * mask_h))
        mask_x_max = mask_x_min + mask_w
        mask_y_max = mask_y_min + mask_h

        if enable_resize and target_width > 0 and target_height > 0:
            target_aspect = target_width / target_height
            current_aspect = crop_w / crop_h
            if abs(target_aspect - current_aspect) > 1e-3:
                if target_aspect > current_aspect:
                    # 需要扩展宽度
                    new_crop_w = int(round(crop_h * target_aspect))
                    w_expand = new_crop_w - crop_w
                    left_expand = w_expand // 2
                    right_expand = w_expand - left_expand
                    crop_x_min -= left_expand
                    crop_x_max += right_expand
                    if padding_mode == "none":
                        crop_x_min = max(0, crop_x_min)
                        crop_x_max = min(img_w, crop_x_max)
                else:
                    # 需要扩展高度
                    new_crop_h = int(round(crop_w / target_aspect))
                    h_expand = new_crop_h - crop_h
                    top_expand = h_expand // 2
                    bottom_expand = h_expand - top_expand
                    crop_y_min -= top_expand
                    crop_y_max += bottom_expand
                    if padding_mode == "none":
                        crop_y_min = max(0, crop_y_min)
                        crop_y_max = min(img_h, crop_y_max)
            crop_w = crop_x_max - crop_x_min
            crop_h = crop_y_max - crop_y_min

        crop_x_min_raw = crop_x_min
        crop_y_min_raw = crop_y_min
        crop_x_max_raw = crop_x_max
        crop_y_max_raw = crop_y_max

        if padding_mode != "none":
            pad_left = max(0, -crop_x_min)
            pad_top = max(0, -crop_y_min)
            pad_right = max(0, crop_x_max - img_w)
            pad_bottom = max(0, crop_y_max - img_h)

            crop_x_min = max(0, crop_x_min)
            crop_y_min = max(0, crop_y_min)
            crop_x_max = min(img_w, crop_x_max)
            crop_y_max = min(img_h, crop_y_max)
        else:
            mask_x_min = max(0, min(mask_x_min, img_w))
            mask_x_max = max(0, min(mask_x_max, img_w))
            mask_y_min = max(0, min(mask_y_min, img_h))
            mask_y_max = max(0, min(mask_y_max, img_h))

        cropped = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        user_mask_full = None
        user_mask_crop = None
        if mask is not None:
            user_mask_full = mask[0].cpu().numpy().astype(np.float32, copy=False)
            user_mask_full = np.clip(user_mask_full, 0.0, 1.0)
            if user_mask_full.shape[:2] == (img_h, img_w):
                user_mask_crop = user_mask_full[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            else:
                user_mask_full = None

        if padding_mode != "none" and (pad_left or pad_right or pad_top or pad_bottom):
            pad_value = 1.0 if padding_mode == "white" else 0.0
            cropped = np.pad(
                cropped,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=pad_value,
            )
            if user_mask_crop is not None:
                user_mask_crop = np.pad(
                    user_mask_crop,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode="constant",
                    constant_values=0.0,
                )

        cropped = cropped.astype(np.float32, copy=False)
        out_h, out_w = cropped.shape[:2]

        face_mask = np.zeros((out_h, out_w), dtype=np.float32)
        if padding_mode == "none":
            mask_x1 = mask_x_min
            mask_y1 = mask_y_min
            mask_x2 = mask_x_max
            mask_y2 = mask_y_max
            base_crop_x = crop_x_min
            base_crop_y = crop_y_min
        else:
            mask_x1 = mask_x_min
            mask_y1 = mask_y_min
            mask_x2 = mask_x_max
            mask_y2 = mask_y_max
            base_crop_x = crop_x_min_raw
            base_crop_y = crop_y_min_raw

        rel_x1 = int(np.floor(mask_x1 - base_crop_x))
        rel_y1 = int(np.floor(mask_y1 - base_crop_y))
        rel_x2 = int(np.ceil(mask_x2 - base_crop_x))
        rel_y2 = int(np.ceil(mask_y2 - base_crop_y))

        rel_x1 = max(0, min(rel_x1, out_w))
        rel_x2 = max(0, min(rel_x2, out_w))
        rel_y1 = max(0, min(rel_y1, out_h))
        rel_y2 = max(0, min(rel_y2, out_h))

        if rel_x2 > rel_x1 and rel_y2 > rel_y1:
            face_mask[rel_y1:rel_y2, rel_x1:rel_x2] = 1.0

        # crop_mask_full: 裁剪区域在原图中的遮罩
        crop_mask_full = np.zeros((img_h, img_w), dtype=np.float32)
        full_x1 = max(0, min(int(crop_x_min), img_w))
        full_y1 = max(0, min(int(crop_y_min), img_h))
        full_x2 = max(0, min(int(crop_x_max), img_w))
        full_y2 = max(0, min(int(crop_y_max), img_h))

        if full_x2 > full_x1 and full_y2 > full_y1:
            crop_mask_full[full_y1:full_y2, full_x1:full_x2] = 1.0

        before_w = out_w
        before_h = out_h
        final_w = out_w
        final_h = out_h

        if enable_resize and (target_width > 0 or target_height > 0):
            if target_width == 0:
                final_w = max(1, int(round(out_w * (target_height / out_h))))
                final_h = target_height
            elif target_height == 0:
                final_w = target_width
                final_h = max(1, int(round(out_h * (target_width / out_w))))
            else:
                if padding_mode == "none":
                    scale = min(target_width / out_w, target_height / out_h)
                    final_w = max(1, int(round(out_w * scale)))
                    final_h = max(1, int(round(out_h * scale)))
                else:
                    final_w = target_width
                    final_h = target_height

            if final_w != out_w or final_h != out_h:
                cropped = cv2.resize(cropped, (final_w, final_h), interpolation=cv2.INTER_LINEAR)
                face_mask = cv2.resize(face_mask, (final_w, final_h), interpolation=cv2.INTER_LINEAR)
                if user_mask_crop is not None:
                    user_mask_crop = cv2.resize(
                        user_mask_crop,
                        (final_w, final_h),
                        interpolation=cv2.INTER_LINEAR,
                    )

                scale_w = final_w / before_w
                scale_h = final_h / before_h
                pad_left = int(round(pad_left * scale_w))
                pad_right = int(round(pad_right * scale_w))
                pad_top = int(round(pad_top * scale_h))
                pad_bottom = int(round(pad_bottom * scale_h))

        if size_multiple > 1:
            prev_w = final_w
            prev_h = final_h
            final_w = round(final_w / size_multiple) * size_multiple
            final_h = round(final_h / size_multiple) * size_multiple

            if final_w == 0:
                final_w = size_multiple
            if final_h == 0:
                final_h = size_multiple

            if final_w != cropped.shape[1] or final_h != cropped.shape[0]:
                cropped = cv2.resize(cropped, (final_w, final_h), interpolation=cv2.INTER_LINEAR)
                face_mask = cv2.resize(face_mask, (final_w, final_h), interpolation=cv2.INTER_LINEAR)
                if user_mask_crop is not None:
                    user_mask_crop = cv2.resize(
                        user_mask_crop,
                        (final_w, final_h),
                        interpolation=cv2.INTER_LINEAR,
                    )

                scale_w = final_w / prev_w if prev_w else 1.0
                scale_h = final_h / prev_h if prev_h else 1.0
                pad_left = int(round(pad_left * scale_w))
                pad_right = int(round(pad_right * scale_w))
                pad_top = int(round(pad_top * scale_h))
                pad_bottom = int(round(pad_bottom * scale_h))

        face_mask = np.clip(face_mask, 0.0, 1.0)
        if user_mask_crop is not None:
            face_mask = np.clip(face_mask * user_mask_crop, 0.0, 1.0)
        if user_mask_full is not None:
            crop_mask_full = np.clip(crop_mask_full * user_mask_full, 0.0, 1.0)

        cropped_tensor = torch.from_numpy(cropped).unsqueeze(0)
        face_mask_tensor = torch.from_numpy(face_mask).unsqueeze(0)
        crop_mask_full_tensor = torch.from_numpy(crop_mask_full).unsqueeze(0)

        crop_info = {
            "original_size": (img_w, img_h),
            "crop_region": (int(crop_x_min), int(crop_y_min), int(crop_x_max), int(crop_y_max)),
            "cropped_size": (int(final_w), int(final_h)),
            "padding": {
                "top": int(pad_top),
                "bottom": int(pad_bottom),
                "left": int(pad_left),
                "right": int(pad_right),
            },
        }
        t5 = time.perf_counter()
        print(
            "FaceHandCrop timing:",
            f"pre={(t1 - t0):.3f}s, load={(t2 - t1):.3f}s, infer={(t4 - t3):.3f}s, post={(t5 - t4):.3f}s, total={(t5 - t0):.3f}s",
        )

        return (
            cropped_tensor,
            face_mask_tensor,
            crop_mask_full_tensor,
            crop_info,
            face_count,
        )


NODE_CLASS_MAPPINGS = {
    "FaceHandCrop": FaceHandCropNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceHandCrop": "Face Hand Crop (YOLO)",
}
