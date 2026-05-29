# YOLO-seg + Track + MatAnyone2 真实短 Burst 离线验证设计

## Summary

目标是在不使用人工标注、不修改线上拍照主链路的前提下，验证一条更具泛化能力的人像主体分离路线：

```text
真实短 burst 原始帧
-> YOLO11x-seg + ByteTrack 锁定同一主体并识别游客
-> MatAnyone2 使用完整序列生成视频 alpha
-> 每帧以 tracked 主体/游客实例约束 alpha
-> 从序列抽取 4 张成片
-> 与现有离线分支对比
```

本轮优先验证质量上限：重点观察游客残留、发缘/衣袖软外溢、连续输出边缘稳定性；记录耗时，但暂不以线上 `5-10s` 预算作为淘汰条件。

## Key Decisions

- 本轮仅新增离线采集与评测能力，不接入 `capture_manager.process_capture_task()` 默认路径，不改变当前线上 `4` 张结果行为。
- 不再采用固定“右发缘”或固定方位规则作为主算法；方向区域仅可用于 debug 对照。
- 不人工标注 ground truth，不微调模型。自动指标用于筛查退化与比较相对表现，最终质量判断仍依据对比总览图。
- 主验证分支为 `tracked_matanyone`；并行生成 `tracked_yolo_seg_mask` 与 `tracked_modnet_4frame` 作为实例归属基线和稳健对照。
- Ultralytics track 使用现有 `models/yolo11x-seg.pt` 与 `tracker="bytetrack.yaml"`。官方 track API 支持 Segment 模型、`persist=True` 的连续帧追踪及 track id 输出。

## Components And Data Flow

### 1. 离线 Burst 采集

新增独立 CLI `capture_burst_eval.py`，直接复用当前摄像头驱动，只负责采样与落盘：

```text
generated/burst_eval_inputs/<session>/<scenario>/<take>/
  frames/000001.jpg ... 000016.jpg
  burst.avi
  metadata.json
```

默认配置：

```text
frame_count = 16
fps = 16
duration_seconds = 1.0
output_frame_indices = [3, 7, 10, 13]  # 约 25% / 45% / 65% / 85%
```

采集场景名称仅作样本分类，不参与算法：

```text
single_subject
visitor_behind_left
visitor_behind_right
visitor_crossing
visitor_close_contact
hair_edge_motion
```

建议每类 `2` 组，共不少于 `12` 组 burst。采集失败或少于 `12` 个有效帧时，该 take 标记失败，不进入评测。

### 2. 序列主体跟踪与实例约束

新增 `subject_instance_tracking.py`，输出每帧可直接供 matting 后处理使用的 tracking 结果：

```python
TrackedInstanceSequence:
    subject_track_id: int
    frames: list[TrackedFrameResult]
    track_switch_count: int
    track_lost_frames: int
    status: str

TrackedFrameResult:
    frame_path: Path
    selected: InstanceCandidate | None
    visitors: list[InstanceCandidate]
    sure_foreground: np.ndarray
    sure_background: np.ndarray
    unknown: np.ndarray
    track_recovered: bool
```

处理规则：

- 对帧序列顺序执行 `YOLO.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0])`，保留 person 实例 mask、bbox、置信度和 track id。
- 首帧按现有主体选择评分（居中、面积大、位置靠下）选定主角，并记录 `subject_track_id`。
- 后续帧优先选择相同 track id；所有其他 person 实例均视为游客。
- 若主体 id 临时缺失，最多连续 `2` 帧使用上一帧主体 mask 与候选 mask 的 IoU、中心距离综合恢复；恢复候选必须满足最小 IoU 阈值 `0.35`。
- 超过 `2` 帧无法恢复，或首帧没有主体，则该序列标记 `tracking_failed`，不输出 `tracked_matanyone` 结果。
- 每帧约束仍复用实例 trimap 逻辑：主体 mask 腐蚀生成 `sure_foreground`，游客 mask 膨胀并入 `sure_background`，剩余主体边缘为 `unknown`。

### 3. 视频 Matting 分支

新增离线 runner `run_tracked_video_matting_eval.py`，对每个有效 burst 生成三条可比较分支：

```text
tracked_yolo_seg_mask
tracked_modnet_4frame
tracked_matanyone
```

`tracked_matanyone` 数据流：

```text
16 帧原始序列
-> 首帧 tracked 主体实例 mask 作为 MatAnyone2 初始 prompt
-> MatAnyone2 输出每帧 raw alpha
-> 对每帧应用其自己的 tracked 约束：
     sure_background / visitor mask 区域 alpha = 0
     sure_foreground 区域 alpha = 255
     unknown 区域保留 MatAnyone2 alpha
-> 轻量 edge refine
-> 再次强制 sure_background = 0 与 sure_foreground = 255
-> 选定 4 帧合成 final
```

`tracked_modnet_4frame` 只对相同的 4 个输出帧运行 MODNet，并应用同一套 tracked 实例约束与 edge refine，作为更低复杂度参考线。

`tracked_yolo_seg_mask` 直接使用 tracked 主体实例 mask 输出 cutout，用于观察错误首先来自实例归属还是 matting 边缘。

### 4. 输出与调试

每个 take 输出：

```text
generated/tracked_video_matting_eval/<timestamp>/<scenario>/<take>/
  tracking_debug/
  tracked_yolo_seg_mask/cutouts|final/
  tracked_modnet_4frame/cutouts|final/
  tracked_matanyone/cutouts|final/
  sheets/cutout_sheet.jpg
  sheets/final_sheet.jpg
  summary_metrics.json
```

必须输出的 debug：

- 带 track id、主体绿色 mask、游客红色 mask 的逐帧预览图或视频。
- 每帧 `sure_foreground`、`sure_background` 与 `unknown`。
- MatAnyone2 raw alpha 与约束后 alpha。
- Track 丢失、恢复与切换事件日志。

## Metrics And Acceptance

自动指标按 burst 与全数据集汇总：

```text
visitor_track_alpha_ratio
subject_core_missing_ratio
outside_subject_soft_alpha_ratio
edge_temporal_jitter
track_switch_count
track_lost_frames
elapsed_seconds
```

其中：

- `visitor_track_alpha_ratio`：所有游客实例 mask 区域内最终有效 alpha 比例；用于筛查游客残留。
- `subject_core_missing_ratio`：主体 `sure_foreground` 区域内 alpha 缺失比例；用于筛查误删。
- `outside_subject_soft_alpha_ratio`：主体实例 mask 外的有效软 alpha 比例；用于筛查软晕扩张。
- `edge_temporal_jitter`：将相邻输出 alpha 按主体 bbox/位移配准后，边缘带差异比例的均值；用于筛查闪边。

第一轮通过条件：

- `tracked_matanyone` 在肉眼对比中不出现可辨认的游客脸部或躯干残留。
- 与现有 `4` 帧 MatAnyone constrained 输出相比，头发、衣袖和肩线周边软晕/闪边明显减少。
- `subject_core_missing_ratio` 不出现明显上升；主体脸部、眼镜、肩线不发生可见缺损。
- 若 `tracked_matanyone` 整体观感不优于 `tracked_modnet_4frame`，停止推进 MatAnyone 上线设计，后续优先研究 tracked constraints + MODNet。
- 本轮只记录 `elapsed_seconds`，不因超过线上时延预算直接否决质量验证。

## Failure Handling And Tests

- 采集层：相机不可用、写帧失败、有效帧不足时报告 take 失败，并保留已采集日志。
- Tracking 层：首帧无主体、track id 长时间丢失、mask 为空时仅跳过该分支该 take，不影响其他 take 的评测。
- Matting 层：MatAnyone2 或 MODNet 不可用时，在 summary 中记录 `not_available`，仍保留可运行的 mask/其他分支。

测试要求：

- 单元测试覆盖 track id 保持、短暂丢失恢复、游客列表生成、trimap 约束映射、输出帧索引选择及自动指标计算。
- 集成测试使用 fake detector / fake matting backend 验证 `16` 帧输入到 `4` 张输出的完整数据流与失败回退。
- 现场离线验证采集不少于 `12` 组真实 burst，并输出全局总览和按场景汇总 JSON。

## Assumptions

- 当前机器 GPU 足以进行本轮离线质量验证；推理时长仅记录，不做上线承诺。
- 用户允许重新采集真实短 burst，但不接受人工 alpha 标注或针对场景的模型微调。
- 最终线上产品契约仍为一次拍摄产生 `4` 张可选成片；本轮不修改线上交互。
- 参考：Ultralytics Track 官方文档说明 tracking 支持 Segment 模型、ByteTrack 以及 `persist=True` 顺序帧跟踪：https://docs.ultralytics.com/modes/track/
