from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from ali import AliOssSegmentPipeline as _AliPipeline
from app_state import OUTPUT_DIR
from modelscope_universal_matting_service import (
    MODELSCOPE_UNIVERSAL_MODEL_ID,
    ModelScopeUniversalMattingError,
    ModelScopeUniversalMattingService,
)
from subject_alpha_filter import (
    SubjectAlphaFilterConfig,
    filter_subject_alpha_with_instance,
)
from subject_edge_refine import (
    SubjectEdgeRefineConfig,
    refine_subject_edge,
    save_edge_refine_debug,
)
from subject_instance_segmentation import (
    InstanceSegmentationConfig,
    SubjectInstanceSegmenter,
)
from subject_visitor_suppression import (
    SubjectVisitorSuppressionConfig,
    apply_post_alpha_hard_clear_with_instance,
    suppress_visitors_with_instance_masks,
)


class AliSegmentError(RuntimeError):
    def __init__(self, message: str, *, provider: str = "ali_segment_body", stage: str = "segment") -> None:
        super().__init__(message)
        self.provider = provider
        self.stage = stage


def _create_pipeline(api_cfg: dict[str, Any]) -> _AliPipeline:
    """根据 matting_api 布尔开关选择管线：use_suxiaoban > use_seedream > 默认 ali。"""
    use_suxiaoban = bool(api_cfg.get("use_suxiaoban", False))
    use_seedream = bool(api_cfg.get("use_seedream", False))

    output_dir = api_cfg.get("output_dir")
    bucket = api_cfg.get("bucket")
    region = api_cfg.get("region")
    oss_endpoint = api_cfg.get("oss_endpoint")
    imageseg_endpoint = api_cfg.get("imageseg_endpoint")

    base_kwargs: dict[str, Any] = {}
    if output_dir:
        # 解析为绝对路径，避免被管线内部以自身包目录为基准拼接
        base_kwargs["output_dir"] = str(Path(output_dir).resolve())
    if bucket:
        base_kwargs["bucket"] = bucket
    if region:
        base_kwargs["region"] = region
    if oss_endpoint:
        base_kwargs["oss_endpoint"] = oss_endpoint
    if imageseg_endpoint:
        base_kwargs["imageseg_endpoint"] = imageseg_endpoint

    if use_suxiaoban:
        try:
            from ali_seedream_chinamobile.ali import AliOssSegmentPipeline as SuxiaobanPipeline
        except ImportError as exc:
            raise ImportError(
                "ali_seedream_chinamobile 依赖未安装，请执行: pip install pyyaml requests"
            ) from exc

        base_kwargs["ali_segment_max_side"] = int(api_cfg.get("max_image_edge", 2000))
        base_kwargs["use_suxiaoban_greenscreen"] = True
        base_kwargs["use_yolo_segment"] = False
        base_kwargs["use_volcengine_greenscreen"] = False

        sb_cfg = api_cfg.get("suxiaoban", {})
        if isinstance(sb_cfg, dict):
            if sb_cfg.get("api_url"):
                base_kwargs["suxiaoban_api_url"] = str(sb_cfg["api_url"])
            if sb_cfg.get("authorization"):
                base_kwargs["suxiaoban_authorization"] = str(sb_cfg["authorization"])
            if sb_cfg.get("model"):
                base_kwargs["suxiaoban_model"] = str(sb_cfg["model"])
            if sb_cfg.get("prompt"):
                base_kwargs["suxiaoban_prompt"] = str(sb_cfg["prompt"])
            if sb_cfg.get("size"):
                base_kwargs["suxiaoban_size"] = str(sb_cfg["size"])
            if sb_cfg.get("request_timeout_sec"):
                base_kwargs["suxiaoban_request_timeout_sec"] = float(sb_cfg["request_timeout_sec"])

        return _wrap_pipeline_init(SuxiaobanPipeline, base_kwargs, "Suxiaoban")  # type: ignore[arg-type]

    if use_seedream:
        try:
            from ali_seedream.ali import AliOssSegmentPipeline as SeedreamPipeline
        except ImportError as exc:
            raise ImportError(
                "ali_seedream 依赖未安装，请执行: pip install pyyaml volcengine-python-sdk[ark] ultralytics"
            ) from exc

        base_kwargs["ali_segment_max_side"] = int(api_cfg.get("max_image_edge", 2000))
        return _wrap_pipeline_init(SeedreamPipeline, base_kwargs, "Seedream")  # type: ignore[arg-type]

    base_kwargs["max_image_edge"] = int(api_cfg.get("max_image_edge", 2000))
    return _wrap_pipeline_init(_AliPipeline, base_kwargs, "Ali")


def _wrap_pipeline_init(factory: Any, kwargs: dict[str, Any], label: str) -> Any:
    try:
        return factory(**kwargs)
    except AliSegmentError:
        raise
    except Exception as exc:
        raise AliSegmentError(f"Failed to initialize {label} matting pipeline: {exc}", stage="initialize") from exc


class AliSegmentService:
    def __init__(self, config: dict[str, Any]):
        api_cfg = config.get("matting_api", {})
        self.provider = str(api_cfg.get("provider", "ali_segment_body"))
        self.modelscope_universal = None
        self.alpha_filter_config = SubjectAlphaFilterConfig.from_mapping(
            config.get("subject_alpha_filter", {})
        )
        self.visitor_suppression_config = SubjectVisitorSuppressionConfig.from_mapping(
            config.get("subject_visitor_suppression", {})
        )
        self.edge_refine_config = SubjectEdgeRefineConfig.from_mapping(
            config.get("subject_edge_refine", {})
        )
        self.last_subject_location = None
        self.last_instance_result = None
        self._instance_metrics_lock = threading.Lock()
        self._instance_metrics_by_stem: dict[str, dict[str, float | int]] = {}
        self.instance_seg_config = InstanceSegmentationConfig.from_mapping(
            config.get("online_instance_segmentation", {})
        )
        self.instance_segmenter = SubjectInstanceSegmenter(
            self.instance_seg_config,
            output_dir=OUTPUT_DIR / "subject_instance_online",
        )
        if self.provider == "modelscope_universal":
            model_id = str(api_cfg.get("modelscope_universal_model_id", MODELSCOPE_UNIVERSAL_MODEL_ID))
            try:
                self.modelscope_universal = ModelScopeUniversalMattingService(model_id=model_id)
                self.modelscope_universal.warmup()
            except ModelScopeUniversalMattingError as exc:
                raise AliSegmentError(
                    f"Failed to initialize ModelScope universal matting pipeline: {exc}",
                    provider=self.provider,
                    stage="initialize",
                ) from exc
            except Exception as exc:
                raise AliSegmentError(
                    f"Failed to initialize ModelScope universal matting pipeline: {exc}",
                    provider=self.provider,
                    stage="initialize",
                ) from exc
        else:
            try:
                self.pipeline = _create_pipeline(api_cfg)
            except AliSegmentError:
                raise
            except ImportError as exc:
                raise AliSegmentError(
                    f"Failed to import matting pipeline dependencies: {exc}",
                    stage="initialize",
                ) from exc
            except Exception as exc:
                raise AliSegmentError(f"Failed to initialize matting pipeline: {exc}", stage="initialize") from exc

    def _set_instance_metrics(self, stem: str, metrics: dict[str, float | int]) -> None:
        lock = getattr(self, "_instance_metrics_lock", None)
        store = getattr(self, "_instance_metrics_by_stem", None)
        if lock is None or store is None:
            return
        with lock:
            store[stem] = dict(metrics)

    def get_instance_metrics(self, stem: str) -> dict[str, float | int] | None:
        lock = getattr(self, "_instance_metrics_lock", None)
        store = getattr(self, "_instance_metrics_by_stem", None)
        if lock is None or store is None:
            return None
        with lock:
            value = store.get(stem)
        return dict(value) if isinstance(value, dict) else None

    def segment_image_file(self, source_path: Path, output_path: Path) -> Image.Image:
        segment_source = source_path
        location = None
        self.last_subject_location = None
        instance_t0 = time.perf_counter()
        instance_segmenter = getattr(self, "instance_segmenter", None)
        if instance_segmenter is None:
            raise AliSegmentError(
                "instance_segmentation_failed: instance segmenter not initialized",
                provider=self.provider,
                stage="instance",
            )
        self.last_instance_result = instance_segmenter.segment(source_path, output_path.stem)
        if self.last_instance_result is None:
            raise AliSegmentError(
                "instance_segmentation_failed: no suitable person instance",
                provider=self.provider,
                stage="instance",
            )
        metrics = {
            "candidates_count": int(len(self.last_instance_result.candidates)),
            "visitors_count": int(len(self.last_instance_result.visitors)),
            "selected_score": float(self.last_instance_result.selected.score or 0.0),
            "selected_mask_px": int(self.last_instance_result.selected.mask.sum()),
            "instance_elapsed_seconds": float(time.perf_counter() - instance_t0),
        }
        self._set_instance_metrics(output_path.stem, metrics)
        print(
            "[SubjectInstanceSeg][Online] "
            f"candidates={metrics['candidates_count']} "
            f"visitors={metrics['visitors_count']} "
            f"selected_score={metrics['selected_score']:.3f} "
            f"selected_mask_px={metrics['selected_mask_px']} "
            f"instance_elapsed_seconds={metrics['instance_elapsed_seconds']:.3f}"
        )

        visitor_suppression_config = getattr(
            self,
            "visitor_suppression_config",
            SubjectVisitorSuppressionConfig(enabled=False),
        )
        if visitor_suppression_config.enabled and visitor_suppression_config.pre_aliyun_enabled:
            try:
                with Image.open(source_path) as source_image:
                    suppression = suppress_visitors_with_instance_masks(
                        source_image.convert("RGB"),
                        self.last_instance_result,
                        visitor_suppression_config,
                    )
                preclean_path = output_path.parent / f"{output_path.stem}_subject_preclean.jpg"
                preclean_path.parent.mkdir(parents=True, exist_ok=True)
                suppression.image.save(preclean_path, format="JPEG", quality=95)
                segment_source = preclean_path
                print(
                    "[SubjectVisitorSuppression][Online] "
                    f"preclean_pixels={suppression.visitor_mask_pixels} "
                    f"source={segment_source}"
                )
            except Exception as exc:
                print(f"[SubjectVisitorSuppression][Online] fallback to raw image: {exc}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.provider == "modelscope_universal":
            try:
                if self.modelscope_universal is None:
                    raise AliSegmentError(
                        "ModelScope universal pipeline not initialized",
                        provider=self.provider,
                        stage="initialize",
                    )
                image = self.modelscope_universal.segment_image_file(segment_source)
            except ModelScopeUniversalMattingError as exc:
                raise AliSegmentError(
                    f"{self.provider} person segmentation failed: {exc}",
                    provider=self.provider,
                    stage="segment",
                ) from exc
            except AliSegmentError:
                raise
            except Exception as exc:
                raise AliSegmentError(
                    f"{self.provider} person segmentation failed: {exc}",
                    provider=self.provider,
                    stage="segment",
                ) from exc
        else:
            try:
                result = self.pipeline.process_and_save(
                    segment_source,
                    output_filename=output_path.name,
                )
            except AliSegmentError:
                raise
            except Exception as exc:
                raise AliSegmentError(
                    f"{self.provider} person segmentation failed: {exc}",
                    provider=self.provider,
                    stage="segment",
                ) from exc

            if result.ndim == 2:
                image = Image.fromarray(result)
            elif result.ndim == 3 and result.shape[2] == 4:
                rgba = cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)
                image = Image.fromarray(rgba)
            elif result.ndim == 3 and result.shape[2] == 3:
                rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
            else:
                raise AliSegmentError(
                    f"Unsupported segmented image shape: {result.shape}",
                    provider=self.provider,
                    stage="decode",
                )

        alpha_filter_config = getattr(self, "alpha_filter_config", SubjectAlphaFilterConfig())
        if self.last_instance_result is not None and alpha_filter_config.enabled:
            try:
                before_alpha = image.getchannel("A") if image.mode == "RGBA" else image.convert("RGBA").getchannel("A")
                before_area = before_alpha.point(lambda value: 255 if value > alpha_filter_config.alpha_threshold else 0).getbbox()
                image = filter_subject_alpha_with_instance(image, self.last_instance_result, alpha_filter_config)
                after_alpha = image.getchannel("A")
                before_pixels = 0
                after_pixels = 0
                if before_area:
                    import numpy as np

                    before_pixels = int(np.count_nonzero(np.array(before_alpha) > alpha_filter_config.alpha_threshold))
                    after_pixels = int(np.count_nonzero(np.array(after_alpha) > alpha_filter_config.alpha_threshold))
                removed = max(before_pixels - after_pixels, 0)
                print(
                    "[SubjectAlphaFilter][Online] "
                    f"visitors={len(self.last_instance_result.visitors)} "
                    f"alpha_before={before_pixels} alpha_after={after_pixels} removed={removed}"
                )
            except Exception as exc:
                print(f"[SubjectAlphaFilter][Online] fallback to raw result: {exc}")

        visitor_suppression_config = getattr(
            self,
            "visitor_suppression_config",
            SubjectVisitorSuppressionConfig(enabled=False),
        )
        if self.last_instance_result is not None and visitor_suppression_config.enabled:
            try:
                before_hard_clear = None
                if image.mode == "RGBA":
                    import numpy as np

                    before_hard_clear = int(np.count_nonzero(np.array(image.getchannel("A")) > 0))
                image = apply_post_alpha_hard_clear_with_instance(
                    image,
                    self.last_instance_result,
                    visitor_suppression_config,
                )
                if before_hard_clear is not None:
                    after_hard_clear = int(np.count_nonzero(np.array(image.getchannel("A")) > 0))
                    print(
                        "[SubjectVisitorSuppression][Online] "
                        f"post_alpha_before={before_hard_clear} "
                        f"post_alpha_after={after_hard_clear} "
                        f"removed={max(before_hard_clear - after_hard_clear, 0)} "
                        f"visitor_suppression_weak={bool(self.last_instance_result.visitors and before_hard_clear == after_hard_clear)}"
                    )
            except Exception as exc:
                print(f"[SubjectVisitorSuppression][Online] post hard-clear skipped: {exc}")

        edge_refine_config = getattr(self, "edge_refine_config", SubjectEdgeRefineConfig(enabled=False))
        if edge_refine_config.enabled:
            try:
                before_refine = image
                refine_result = refine_subject_edge(image, edge_refine_config)
                image = refine_result.image
                save_edge_refine_debug(
                    before_refine,
                    refine_result,
                    output_dir=OUTPUT_DIR / "subject_debug",
                    stem=output_path.stem,
                    config=edge_refine_config,
                )
                print(
                    "[SubjectEdgeRefine] "
                    f"removed_small_components={refine_result.removed_small_components} "
                    f"alpha_area_before={refine_result.alpha_area_before} "
                    f"alpha_area_after={refine_result.alpha_area_after} "
                    f"effective_bbox={refine_result.effective_bbox}"
                )
            except Exception as exc:
                print(f"[SubjectEdgeRefine] skipped: {exc}")

        image.save(output_path, format="PNG")
        return image
