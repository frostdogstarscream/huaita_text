# -*- coding: utf-8 -*-
"""
流水线预处理步骤：显式策略接口 + 链式执行，便于扩展与单测。

``AliOssSegmentPipeline`` 在构造时若未传入 ``preprocessors``，则根据 YAML / 参数
中的 ``use_yolo_segment``、``use_volcengine_greenscreen``、``use_suxiaoban_greenscreen``
组装默认链（顺序：YOLO 最大实例 → 火山绿幕 → 速销班图生图绿幕）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Protocol, runtime_checkable

import numpy as np

from .suxiaobanengine_segment_greenscreen import SuxiaobanImageGenerationsClient

if TYPE_CHECKING:
    from .volcengine_segment_greenscreen import VolcengineGreenscreenSegment
    from .yolo_seg import YoloLargestSegmentMask


@runtime_checkable
class SegmentPipelinePreprocessor(Protocol):
    """单步预处理：输入与输出均为 ``uint8`` 的 ``numpy.ndarray``（BGR / BGRA / 灰）。"""

    def apply(self, image: np.ndarray) -> np.ndarray:
        ...


class YoloLargestInstancePreprocessor:
    """将 YOLO 最大实例分割封装为可链接的一步。"""

    def __init__(self, impl: "YoloLargestSegmentMask") -> None:
        self._impl = impl

    def apply(self, image: np.ndarray) -> np.ndarray:
        return self._impl.process(image, save=False)


class VolcengineGreenscreenPreprocessor:
    """将火山方舟绿幕预处理封装为可链接的一步。"""

    def __init__(self, impl: "VolcengineGreenscreenSegment") -> None:
        self._impl = impl

    def apply(self, image: np.ndarray) -> np.ndarray:
        return self._impl.process(image, save=False)


class SuxiaobanGreenscreenPreprocessor:
    """将速销班 llmgate 图生图（绿幕）封装为可链接的一步。"""

    def __init__(self, impl: SuxiaobanImageGenerationsClient) -> None:
        self._impl = impl

    def apply(self, image: np.ndarray) -> np.ndarray:
        return self._impl.generate(image, save=False, verbose=False, retries=3)


def build_default_preprocessors(
    *,
    use_yolo: bool,
    use_volcengine: bool,
    use_suxiaoban: bool = False,
    yolo: Optional["YoloLargestSegmentMask"] = None,
    volcengine: Optional["VolcengineGreenscreenSegment"] = None,
    suxiaoban: Optional[SuxiaobanImageGenerationsClient] = None,
) -> List[SegmentPipelinePreprocessor]:
    """按固定顺序组装默认预处理链（YOLO → 火山 → 速销班）。"""
    steps: List[SegmentPipelinePreprocessor] = []
    if use_yolo and yolo is not None:
        steps.append(YoloLargestInstancePreprocessor(yolo))
    if use_volcengine and volcengine is not None:
        steps.append(VolcengineGreenscreenPreprocessor(volcengine))
    if use_suxiaoban and suxiaoban is not None:
        steps.append(SuxiaobanGreenscreenPreprocessor(suxiaoban))
    return steps
