# YOLO-seg + MODNet 离线验证进展与下一步计划

更新日期：2026-05-22

## 1. 当前目标

当前线上主链路仍保持不变：

```text
4 张 capture
-> YOLO bbox 主体定位
-> ROI 裁切 / 侧向收边
-> 游客预抑制
-> 阿里云 SegmentBody
-> 主体 alpha 过滤 / 游客 hard-clear
-> 边缘净化
-> 4 帧时序融合
-> 背景合成
```

这一阶段的目标不是替换线上链路，而是新增一条离线验证链路：

```text
capture.jpg
-> YOLO11x-seg 实例分割
-> 主角 / 游客实例归属
-> 生成 sure foreground / sure background / unknown
-> MODNet 生成精细 alpha
-> 用实例约束修正 alpha
-> 边缘净化与 4 帧时序融合
-> 输出 cutout / final / 指标
```

核心验证问题：

1. YOLO-seg 是否能把右后方游客和主体分成独立实例。
2. MODNet 是否能在保留发丝、肩线、衣袖边缘的同时，减少当前阿里云链路中的毛刺。
3. 用游客实例 mask 作为 sure background 后，是否能比当前 hard-clear 更稳定地删除右后方游客。

## 2. 已完成进展

### 2.1 YOLO11x-seg 实例归属验证

已新增并验证：

- `subject_instance_segmentation.py`
- `run_yolo_seg_instance_eval.py`
- `tests/test_subject_instance_segmentation.py`
- `tests/test_run_yolo_seg_instance_eval.py`

已下载本地权重：

- `models/yolo11x-seg.pt`
- `models/yolo11n-seg.pt`

这两个权重文件不提交 git。

在两组 8 张 hard case 上，YOLO11x-seg 能稳定检测到：

- 每张图 2 个 person 实例
- 1 个主体
- 1 个游客

上一轮实例验证输出目录：

```text
generated/yolo_seg_instance_eval/20260522_161046/
```

初步结论：

- YOLO-seg 的实例归属是可用的。
- 它比单纯 bbox 更适合给后续 matting 提供“主角属于前景、游客属于背景”的约束。
- 这一步解决的是“谁是谁”的问题，不直接解决边缘 alpha 精细度。

### 2.2 MODNet 离线适配层

已新增：

- `modnet_matting_service.py`

主要能力：

- 提供 `ModnetMattingService`。
- 输入原图与 `InstanceSegmentationResult`。
- 输出 `AlphaConstraintResult`，包含：
  - MODNet raw alpha
  - 实例约束后的 constrained alpha
  - RGBA cutout
  - forced foreground / forced background / unknown 像素统计
- 推理策略：
  - 优先尝试 CUDA
  - CUDA 初始化失败后回退 CPU
  - repo 或 checkpoint 缺失时抛出清晰的 `ModnetMattingError`
- 不接入线上 `APP_STATE["matting_service"]`。
- 不影响当前拍照服务启动和线上抠图。

实例约束逻辑：

```text
MODNet raw alpha
-> sure_foreground 区域 alpha 强制为 255
-> sure_background 区域 alpha 强制为 0
-> unknown 区域保留 MODNet alpha
-> 输出 constrained alpha cutout
```

这样做的目的：

- 主体核心不被 MODNet 误删。
- 游客区域不依赖后处理猜测，而是直接归为确定背景。
- 只有头发、肩膀、衣袖边界进入 matting 的不确定区域。

### 2.3 YOLO-seg + MODNet 离线 runner

已新增：

- `run_yolo_seg_matting_eval.py`

默认输入两组 hard case：

```text
generated/captures/8032532334c940d28cf78782fc2d43b3_1.jpg
...
generated/captures/8032532334c940d28cf78782fc2d43b3_4.jpg

generated/captures/9595dd5a6d504901a8f6911a9a951353_1.jpg
...
generated/captures/9595dd5a6d504901a8f6911a9a951353_4.jpg
```

runner 输出目录：

```text
generated/yolo_seg_matting_eval/<timestamp>/
```

输出内容：

- `debug/*_seg_instance_debug.jpg`
- `debug/*_subject_mask.png`
- `debug/*_visitor_mask.png`
- `debug/*_trimap.png`
- `yolo_seg_mask/cutouts/<group>/*.png`
- `yolo_seg_mask/final/<group>/*.jpg`
- `*_yolo_seg_mask_cutout_sheet.jpg`
- `*_yolo_seg_mask_final_sheet.jpg`
- `summary_metrics.json`

如果 MODNet 可用，还会额外输出：

- `debug/*_modnet_raw_alpha.png`
- `debug/*_modnet_constrained_alpha.png`
- `debug/*_modnet_edge_refine_before.png`
- `debug/*_modnet_edge_refine_after.png`
- `yolo_seg_modnet/cutouts/<group>/*.png`
- `yolo_seg_modnet/final/<group>/*.jpg`
- `*_yolo_seg_modnet_cutout_sheet.jpg`
- `*_yolo_seg_modnet_final_sheet.jpg`

### 2.4 当前实际运行结果

已执行命令：

```powershell
python run_yolo_seg_matting_eval.py --groups 8032532334c940d28cf78782fc2d43b3 9595dd5a6d504901a8f6911a9a951353
```

本次输出目录：

```text
generated/yolo_seg_matting_eval/20260522_163225/
```

本次状态：

```text
matting_not_available: MODNet repo not found: D:\code_copy\huaita_text\MODNet
```

说明：

- 当前本机还没有 `D:\code_copy\huaita_text\MODNet`。
- 当前本机还没有 `models\modnet_photographic_portrait_matting.ckpt`。
- 因此 MODNet 分支没有实际推理。
- runner 按设计没有中断，仍完整产出了 YOLO-seg mask baseline。

本次 summary 中的关键结果：

第一组 `8032532334c940d28cf78782fc2d43b3`：

- 4 张全部检测到 2 个 person 实例。
- `visitor_residual_ratio` 约为 `0.0011 - 0.0020`。
- `subject_core_missing_ratio` 为 `0.0`。
- `fragment_count` 为 `1`。
- 4 帧时序融合成功：`alignment_success_count = 4`。

第二组 `9595dd5a6d504901a8f6911a9a951353`：

- 4 张全部检测到 2 个 person 实例。
- `visitor_residual_ratio` 约为 `0.0029 - 0.0103`。
- `subject_core_missing_ratio` 为 `0.0`。
- `fragment_count` 为 `1`。
- 4 帧时序融合成功：`alignment_success_count = 4`。

初步结论：

- YOLO-seg baseline 对“游客区域 alpha 残留”压得很低。
- 但是纯 YOLO-seg mask 的边缘会偏硬，不一定能满足发丝、衣领、肩线质量。
- MODNet 的价值在于补“精细 alpha”，而不是重新判断游客归属。

## 3. 已完成测试

已新增并通过：

```powershell
python -m pytest tests\test_modnet_matting_service.py tests\test_run_yolo_seg_matting_eval.py tests\test_subject_instance_segmentation.py tests\test_run_yolo_seg_instance_eval.py -q
```

结果：

```text
12 passed
```

覆盖点：

- MODNet fake backend 可运行。
- CUDA 初始化失败时回退 CPU。
- MODNet 输出空 alpha 时抛错。
- sure foreground 强制保留。
- sure background 强制清除。
- unknown 保留 raw matting alpha。
- 游客残留率、主体核心误删率、碎片数指标计算正确。
- YOLO-seg 实例归属基础测试仍通过。

## 4. 当前代码状态

本阶段新增文件：

```text
modnet_matting_service.py
run_yolo_seg_matting_eval.py
tests/test_modnet_matting_service.py
tests/test_run_yolo_seg_matting_eval.py
docs/yolo_seg_modnet_progress_plan.md
```

依赖但来自上一阶段的文件：

```text
subject_instance_segmentation.py
run_yolo_seg_instance_eval.py
subject_edge_refine.py
subject_temporal_fusion.py
```

注意：

- 当前仓库里还有较多历史未提交/未跟踪文件。
- 本阶段没有改线上 FastAPI 路由、拍照流程、阿里云主链路配置。
- `models/*.pt`、MODNet repo、MODNet checkpoint 都不应提交 git。

## 5. 当前主要卡点

### 5.1 MODNet repo 与权重缺失

当前 runner 已经具备 MODNet 接入点，但本机缺少：

```text
D:\code_copy\huaita_text\MODNet
D:\code_copy\huaita_text\models\modnet_photographic_portrait_matting.ckpt
```

因此现在只能评测：

```text
YOLO-seg mask baseline
```

还不能评测：

```text
YOLO-seg + MODNet constrained alpha
```

### 5.2 MODNet 对“只保留主角”的能力不能单独依赖

MODNet 是人像 matting，不是实例分割模型。

因此不能让它直接决定：

```text
这个像素属于主体还是游客
```

必须继续依赖 YOLO-seg 提供：

- 主体实例 mask
- 游客实例 mask
- sure foreground
- sure background
- unknown band

MODNet 只负责 unknown band 的精细 alpha。

### 5.3 如果游客和主体实例 mask 本身粘连，MODNet 也救不了

若 YOLO-seg 未把游客和主体分成两个实例，后续 matting 约束会缺少 sure background。

这类情况下一步要补：

- YOLO-pose
- face detector
- 短 burst tracking

而不是继续调 MODNet alpha。

## 6. 下一步详细计划

### 阶段 A：补齐 MODNet 本地推理环境

目标：让 `yolo_seg_modnet` 分支真正跑出结果。

计划：

1. 获取 MODNet 官方代码到：

```text
D:\code_copy\huaita_text\MODNet
```

2. 获取 portrait matting checkpoint 到：

```text
D:\code_copy\huaita_text\models\modnet_photographic_portrait_matting.ckpt
```

3. 确认依赖：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import cv2, numpy, PIL; print('ok')"
```

4. 重新执行：

```powershell
python run_yolo_seg_matting_eval.py
```

5. 检查 `summary_metrics.json` 中：

```text
matting_status = available
```

通过标准：

- `yolo_seg_modnet/cutouts/` 有 8 张 PNG。
- `yolo_seg_modnet/final/` 有 8 张 JPG。
- `debug/*_modnet_raw_alpha.png` 存在。
- `debug/*_modnet_constrained_alpha.png` 存在。

### 阶段 B：与 YOLO-seg mask baseline 做 A/B

目标：判断 MODNet 是否真的改善毛刺和边缘。

对比对象：

```text
A: yolo_seg_mask
B: yolo_seg_modnet
```

重点看：

- 第 3、4 张右上游客区域。
- 左下角毛刺。
- 头发轮廓。
- 眼镜边缘。
- 肩线和衣袖。
- 主体是否被啃。

指标优先级：

1. 游客残留率 `visitor_residual_ratio`
2. 主体核心误删率 `subject_core_missing_ratio`
3. 小碎片数 `fragment_count`
4. 有效前景面积 `foreground_px`
5. 单张耗时和 4 张总耗时

通过标准：

- MODNet 结果的游客残留率不高于 YOLO-seg mask baseline。
- MODNet 结果的主体核心误删率接近 0。
- MODNet 结果边缘明显比 YOLO-seg mask 柔和。
- 毛刺少于当前阿里云链路或 YOLO-seg mask baseline。

阶段 B 已执行结果（2026-05-22 17:05）：

```text
generated/yolo_seg_matting_eval/20260522_170525/
```

状态：

- `matting_status = available`。
- `yolo_seg_mask` 与 `yolo_seg_modnet` 的 cutout/final sheet 均已生成。
- 额外生成深色棋盘格 A/B 图，便于观察白衣主体边缘：

```text
generated/yolo_seg_matting_eval/20260522_170525/stage_b_ab/
```

指标对比：

| 组 | 分支 | 游客残留率均值 | 主体核心误删率均值 | 碎片数均值 | 前景面积均值 |
|---|---|---:|---:|---:|---:|
| `8032532334c940d28cf78782fc2d43b3` | `yolo_seg_mask` | 0.001590 | 0.000000 | 1.00 | 152552.75 |
| `8032532334c940d28cf78782fc2d43b3` | `yolo_seg_modnet` | 0.000000 | 0.010939 | 1.25 | 156268.50 |
| `9595dd5a6d504901a8f6911a9a951353` | `yolo_seg_mask` | 0.006702 | 0.000000 | 1.00 | 163892.75 |
| `9595dd5a6d504901a8f6911a9a951353` | `yolo_seg_modnet` | 0.000000 | 0.012984 | 1.00 | 166589.50 |

耗时：

- 第一组：YOLO 实例分割合计约 2.130s，MODNet 合计约 1.414s，合计约 3.545s。
- 第二组：YOLO 实例分割合计约 0.330s，MODNet 合计约 1.270s，合计约 1.600s。
- 两组 4 帧融合均成功，`alignment_success_count = 4`。

阶段 B 结论：

- MODNet 约束分支把两组游客残留率都压到 `0.0`，优于 YOLO-seg mask baseline。
- 棋盘格 A/B 图显示 MODNet 边缘更柔和，纯 YOLO-seg mask 边缘更硬。
- MODNet 分支的有效前景面积更大，说明 unknown band 中保留了更多软边缘/衣物边缘信息。
- 当前主要风险是 `subject_core_missing_ratio` 从 `0.0` 上升到约 `1.1% - 1.3%`，说明主体核心有少量低 alpha 像素。
- 下一步应进入阶段 C，优先调 `sure_fg_erode_px` 与 `subject_unknown_dilate_px`，目标是在保留 MODNet 柔和边缘的同时，把主体核心误删率压回接近 0。

### 阶段 C：优化 trimap / 约束生成

如果 MODNet 结果出现主体边缘被啃或游客残留，需要优先调 trimap，而不是调全局 alpha 阈值。

可调项：

```text
sure_fg_erode_px
subject_unknown_dilate_px
visitor_bg_dilate_px
```

建议方向：

1. 主体被啃：
   - 减小 `sure_fg_erode_px`
   - 增大 `subject_unknown_dilate_px`
   - 对头发上方区域单独扩大 unknown

2. 游客残留：
   - 增大 `visitor_bg_dilate_px`
   - 游客 bbox 与 mask 联合扩张为 sure background
   - 对右后方游客方向使用更强 sure background

3. 边缘仍硬：
   - 减小 sure foreground 面积
   - 增大 unknown band
   - 让 MODNet 在边缘带发挥作用

4. 边缘过虚：
   - 缩小 unknown band
   - 加强 `SubjectEdgeRefine`
   - 限制低 alpha 远端碎片参与有效 bbox

阶段 C 已执行结果（2026-05-22 17:29）：

工具与输出：

- 新增 `run_yolo_seg_trimap_sweep.py`，支持 24 组 trimap 参数扫描。
- 第一轮扫描输出：

```text
generated/yolo_seg_matting_eval/trimap_sweep/20260522_172927/
```

- 扫描汇总：`sweep_summary.json`、`sweep_summary.csv`。
- 排名前 3 组棋盘格 A/B 图：

```text
generated/yolo_seg_matting_eval/trimap_sweep/20260522_172927/fg10_unk12_vis18/stage_c_checker_ab_sheet.jpg
generated/yolo_seg_matting_eval/trimap_sweep/20260522_172927/fg10_unk18_vis18/stage_c_checker_ab_sheet.jpg
generated/yolo_seg_matting_eval/trimap_sweep/20260522_172927/fg10_unk24_vis18/stage_c_checker_ab_sheet.jpg
```

第一轮扫描结论（仅调 trimap，edge refine 后未回补 sure foreground）：

- 24 组参数全部满足 `visitor_residual_ratio = 0.0`。
- 但严格候选阈值下 `candidate_count = 0`。
- `subject_core_missing_ratio` 在多数组合上稳定在约 `1.1% - 1.3%`，几乎不随 `sure_fg_erode_px` / `subject_unknown_dilate_px` 变化。
- 说明主体核心低 alpha 主要来自 `SubjectEdgeRefine` 在 MODNet 约束之后继续修改 alpha，而不是 trimap 本身失效。

阶段 C 修复（edge refine 后回补 sure foreground）：

- 在 [run_yolo_seg_matting_eval.py](d:/code_copy/huaita_text/run_yolo_seg_matting_eval.py) 增加 `reapply_sure_foreground_alpha()`。
- MODNet 分支在 `SubjectEdgeRefine` 之后，再次把 `sure_foreground` 区域 alpha 强制为 `255`。

验证参数（推荐离线默认值）：

```text
sure_fg_erode_px = 0
subject_unknown_dilate_px = 18
visitor_bg_dilate_px = 18
```

验证输出：

```text
generated/yolo_seg_matting_eval/20260522_173843/
```

验证指标（8 帧聚合）：

| 指标 | 结果 |
|---|---:|
| `visitor_residual_ratio_avg` | 0.000000 |
| `subject_core_missing_ratio_avg` | 0.000000 |
| `subject_core_missing_ratio_max` | 0.000000 |
| `fragment_count_avg` | 1.125 |
| `foreground_px_avg` | 167155.38 |

阶段 C 结论：

- trimap 扫描证明：游客清除对参数不敏感，MODNet + 实例约束已经稳定。
- 主体核心误删不是 trimap 单独能解决的，必须在 edge refine 后回补 `sure_foreground`。
- 推荐离线参数：`fg0_unk18_vis18`。
- 可以进入阶段 D，与 `current_aliyun` 做三方对比；若三方指标仍优，再考虑是否进入线上可切换设计。

### 阶段 D：把当前阿里云链路纳入同一 runner

目标：做真正三方对比。

当前 runner 已能比较：

```text
yolo_seg_mask
yolo_seg_modnet
```

下一步建议加入：

```text
current_aliyun
```

最终对比表：

```text
current_aliyun
yolo_seg_mask
yolo_seg_modnet
```

每组输出：

- cutout sheet
- final sheet
- 指标 JSON
- 单张耗时
- 4 张总耗时

这样才能回答：

```text
新链路是否比当前线上链路更好？
```

而不是只回答：

```text
MODNet 是否比 YOLO-seg mask 更好？
```

阶段 D 已执行结果（2026-05-25）：

新增能力：

- `run_yolo_seg_matting_eval.py` 支持 `--include-current-aliyun`。
- 同一批样本输出三分支：

```text
current_aliyun
yolo_seg_mask
yolo_seg_modnet
```

- `summary_metrics.json` 新增 `aggregate_metrics`。
- 每组新增三方 cutout/final 总览：

```text
*_three_way_cutout_sheet.jpg
*_three_way_final_sheet.jpg
```

实测期间修复了两个会影响结论的问题：

1. 阿里云输出是 ROI 坐标，YOLO-seg 的评测 mask 是原图坐标。已在指标计算时按阿里云实际 ROI 映射 mask，避免阿里云指标缺失或错算。
2. `TemporalSubjectFusion` 曾把 `0..255` alpha 再乘 `255` 引起 `uint8` 溢出，并用边缘权重削弱稳定内部投票，导致 final 中人物几乎透明。已修复 alpha 回映和稳定性公式，并限制贴着主体的偶发软区域扩散。

最新三方输出：

```text
generated/yolo_seg_matting_eval/20260525_082605_fusion_fixed/
```

说明：该目录复用了同轮已成功获取的阿里云 cutout，只重新执行修复后的本地时序融合与 final 合成，避免重复产生云端调用。

指标口径：

- 下表为**融合前单帧 cutout** 指标，用于比较分割引擎与实例约束本身。
- `*_three_way_final_sheet.jpg` 为**修复后时序融合**的实际合成视觉结果。

8 帧聚合指标：

| 分支 | 游客残留率均值 | 游客残留率最大值 | 主体核心误删率均值 | 碎片数均值 |
|---|---:|---:|---:|---:|
| `current_aliyun` | 0.099180 | 0.733398 | 0.078060 | 1.125 |
| `yolo_seg_mask` | 0.004146 | 0.010324 | 0.026101 | 1.000 |
| `yolo_seg_modnet` | 0.000000 | 0.000000 | 0.000000 | 1.125 |

阶段 D 结论：

- 在当前 8 张贴身游客 hard case 中，`YOLO-seg + MODNet` 的单帧游客残留与主体核心误删均优于当前阿里云基线。
- `YOLO-seg mask` 已能大幅压低游客，但精细 alpha 的视觉上限低于 MODNet 分支。
- 修复后的时序融合 final 图已可用于视觉验收，人物不会再因 alpha 溢出而消失。
- 该结论仍是离线样本结论；进入线上可切换设计前，应扩大测试集并测量 4 张现场总延迟。

### 阶段 E：YOLO-seg 精确游客抑制 + 阿里云边缘精修验证

目标：验证阿里云仅负责 unknown 边缘 alpha 时，能否在保持实例约束收益的同时，比 MODNet 输出更自然的发丝、肩线和衣袖边缘。

新增离线分支：

```text
capture.jpg
-> YOLO11x-seg 主体/游客实例分割
-> 主体 bbox 扩边裁自然 RGB ROI
-> visitor instance mask 精确 inpaint
-> 阿里云 SegmentBody
-> visitor mask 清 alpha + sure_foreground 回补
-> SubjectEdgeRefine + TemporalSubjectFusion
-> final
```

新增实现：

- `yolo_seg_aliyun_service.py`：生成自然 RGB ROI、精确游客 inpaint、阿里云结果 alpha 约束和调试文件。
- `run_yolo_seg_matting_eval.py --include-yolo-seg-aliyun`：同一实例分割结果下运行第四分支并生成四方对比。
- `tests/test_yolo_seg_aliyun_service.py`：覆盖 ROI/mask 映射、inpaint、输出缩放以及 alpha 约束。

固定验证参数：

```text
sure_fg_erode_px = 0
subject_unknown_dilate_px = 18
visitor_bg_dilate_px = 18
roi_expand_ratio = 0.12
visitor_mask_dilate_px = 8
inpaint_radius = 9
```

四方实测输出（2026-05-25）：

```text
generated/yolo_seg_matting_eval/20260525_085231/
```

8 帧融合前单帧聚合指标：

| 分支 | 游客残留率均值 | 游客残留率最大值 | 主体核心误删率均值 | 平均前景像素 |
|---|---:|---:|---:|---:|
| `current_aliyun` | 0.099180 | 0.733398 | 0.078060 | 171256.0 |
| `yolo_seg_mask` | 0.004146 | 0.010324 | 0.026101 | 158222.8 |
| `yolo_seg_modnet` | 0.000000 | 0.000000 | 0.000000 | 167155.4 |
| `yolo_seg_aliyun` | 0.000000 | 0.000000 | 0.000000 | 169352.1 |

运行记录：

- `yolo_seg_aliyun` 8 张均完成，无分支错误。
- 新阿里云分支累计耗时约 `45.326s`，平均约 `5.666s/张`；这是离线串行测量，不等价于线上 4 张并行耗时。
- `yolo_seg_aliyun` 在数值指标上达到与 `yolo_seg_modnet` 相同的游客残留和主体核心保护水平。

阶段 E 结论：

- 精确游客 mask 确实解决了阿里云原链路中右后方游客残留问题；实例归属是有效关键能力。
- 四方 cutout/final 总览中，`yolo_seg_aliyun` 的衣袖外缘可见明显黑色硬边，视觉边界弱于 `yolo_seg_modnet`，没有证明阿里云边缘精修更优。
- 当前不建议把 `yolo_seg_aliyun` 接入线上默认链路；更值得扩大样本验证 `yolo_seg_modnet` 的泛化和现场速度。

### 阶段 F：若 MODNet 效果不足，切换 PP-MattingV2 验证

触发条件：

- MODNet 边缘不如阿里云。
- MODNet 对白衣、玻璃、强光背景不稳定。
- MODNet 在 CPU/GPU 上速度不满足 4 张 5-10 秒预算。

PP-MattingV2 验证方式：

```text
YOLO-seg 实例归属
-> sure foreground / sure background / unknown
-> PP-MattingV2
-> 同一套 metrics / sheet / final 输出
```

注意：

- PP-MattingV2 可能依赖 Paddle 环境。
- 不应直接接线上。
- 仍先作为离线 runner 的第二分支。

### 阶段 G：如果实例归属失败，补 pose / face / tracking

触发条件：

- YOLO-seg 把主体和游客粘成同一个实例。
- 游客只露半张脸/半个肩膀，YOLO-seg 没检测到。
- 主体和游客重叠且 4 帧都稳定存在。

优先补充：

1. YOLO-pose：
   - 主体通常姿态更完整。
   - 游客可能只有上半身或半张脸。

2. face detector：
   - 主体脸更居中、更大、更稳定。
   - 可用于校正主体选择。

3. burst tracking：
   - 8-16 帧短 burst 时，主体位置更稳定。
   - 游客探头/移动更容易被时序排除。

这一步的目标不是提升 alpha，而是提升：

```text
主角归属稳定性
```

## 7. 推荐验收流程

每次改动后按以下顺序验收：

1. 单元测试：

```powershell
python -m pytest tests\test_modnet_matting_service.py tests\test_run_yolo_seg_matting_eval.py tests\test_subject_instance_segmentation.py tests\test_run_yolo_seg_instance_eval.py -q
```

2. 离线 runner：

```powershell
python run_yolo_seg_matting_eval.py
```

3. 打开输出目录：

```text
generated/yolo_seg_matting_eval/<timestamp>/
```

4. 优先查看：

```text
*_four_way_cutout_sheet.jpg
*_four_way_final_sheet.jpg
summary_metrics.json
```

5. 按优先级判断：

```text
游客残留最低
> 主体完整度
> 毛刺和边缘自然度
> 速度
```

## 8. 当前建议结论

当前已经证明：

- YOLO11x-seg 可以稳定区分样例中的主体和右后方游客。
- YOLO-seg mask baseline 能显著压低游客残留。
- `YOLO-seg + MODNet` 在现有 8 张 hard case 中达到游客残留与主体核心误删均为 `0`。
- `YOLO-seg + 阿里云` 第四分支也达到上述数值指标，但当前总览图中出现明显黑色硬边，视觉不优于 MODNet。
- 离线 runner 已具备四方 A/B、失败兜底、指标汇总与对比总览输出。
- 线上主链路未被改变。

当前还没有证明：

- MODNet 在更大规模现场样本、不同衣物和不同光照下是否保持优势。
- MODNet 是否满足现场 4 张 5-10 秒总耗时目标。

下一步最有价值的动作：

```text
扩充真实 hard case 样本集并人工复核边缘
-> 测量 YOLO-seg + MODNet 在 4 张并行流程中的实际耗时
-> 如泛化与耗时均达标，设计可切换的线上 MODNet 分支
-> 若 MODNet 在特定场景边缘失效，再离线验证 PP-MattingV2
```
