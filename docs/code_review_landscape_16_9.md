# 代码审查：feature/landscape-16-9-support 分支

**审查日期**: 2026-05-27
**审查范围**: 相对于工作树的未提交变更（22 个文件，约 2500 行变更）
**主要变更**: 横屏 16:9 背景支持、时序融合修复、离线评估脚本扩展、前端页面适配

---

## 问题汇总

| # | 严重度 | 文件 | 问题 |
|---|--------|------|------|
| 1 | 中高 | `run_yolo_seg_matting_eval.py:861` | frame_bboxes 与分支 cutout 列表错位 |
| 2 | 中 | `run_yolo_seg_matting_eval.py:244` | 评估脚本 `_compose_final` 硬编码竖屏尺寸 |
| 3 | 中 | `subject_temporal_fusion.py:193` | 稳定性公式从加权混合改为叠加，语义变更 |
| 4 | 中 | `capture_manager.py:94` | 兜底图使用全局输出尺寸，不区分横竖屏 |
| 5 | 低中 | `run_yolo_seg_matting_eval.py:835` | MatAnyone elapsed_seconds 累计而非单帧 |
| 6 | 低中 | `run_yolo_seg_matting_eval.py:239` | 聚合指标无条件输出哨兵值 |
| 7 | 低中 | `subject_temporal_fusion.py:122` | 膨胀核替换连通域分析，截断远处软边缘 |
| 8 | 低 | `config.json:626` | provider 切换导致远程 4090 代码路径不可达 |
| 9 | 低 | `html-page/app.js:1` | BOM 字符注入 |
| 10 | 低 | `html-page/download.html:17` | 移除 img 尺寸属性导致布局偏移 |
| 11 | 低 | `modnet_matting_service.py:75` | torch.load weights_only=False 安全风险 |

---

## 详细分析

### 1. frame_bboxes 与分支 cutout 列表错位（中高严重度）

**文件**: `run_yolo_seg_matting_eval.py:861`

**问题**: 当 4 帧实例分割全部成功，但某个 matting 分支（MODNet、aliyun、vitmatte）对某一帧失败时，该分支的 cutout 列表比 `frame_bboxes` 短。

`frame_bboxes` 通过 `.append()` 在分割成功时添加（第 548 行），4 帧分割全成功则有 4 个元素。但 `modnet_cutouts` 等分支列表仅在 matting 成功时 append（第 575 行），失败时仅记录 `modnet_error`（第 587 行），不添加元素。

```python
# 第 858-861 行
for i, subject in enumerate(fused, start=1):  # i = 1..3（fused 有 3 个元素）
    if subject is None:
        continue
    bbox = frame_bboxes[i - 1]  # frame_bboxes 有 4 个元素，索引错位
```

**触发场景**: 分割全部成功 + MODNet 对第 2 帧失败 → `modnet_cutouts = [f1, f3, f4]`（3 个），`frame_bboxes = [bb1, bb2, bb3, bb4]`（4 个）→ fused 第 2 项是第 3 帧结果，但 `frame_bboxes[1]` 是第 2 帧的 bbox。

**修复建议**: 使用字典映射 `frame_index → bbox` 而非列表索引，或在 matting 失败时也填充 fallback 到 cutout 列表以保持对齐。

---

### 2. 评估脚本 `_compose_final` 硬编码竖屏尺寸（中严重度）

**文件**: `run_yolo_seg_matting_eval.py:244-246`

**问题**: 生产路径 `image_composer.compose_single_variant` 已改用 `resolve_output_size(background_item)` 按 orientation 返回尺寸，但评估脚本的 `_compose_final` 仍硬编码全局配置：

```python
def _compose_final(...):
    output_cfg = APP_STATE["config"]["output"]
    target_size = (int(output_cfg["width"]), int(output_cfg["height"]))  # 始终 1080x1920
```

**影响**: 当前因 `get_background_items()[0]` 恰好是竖屏背景而无问题。若切换为横屏背景，评估合成结果会变形为 1080x1920，与生产路径的 1920x1080 不一致。

**修复建议**: 改为调用 `resolve_output_size(background_item)`。

---

### 3. 时序融合稳定性公式语义变更（中严重度）

**文件**: `subject_temporal_fusion.py:193`

**问题**: 公式从加权混合改为叠加：

```python
# 旧公式（加权混合）
stability = vote_ratio * (1.0 - edge_w) + edge_consistency * edge_w

# 新公式（叠加）
stability = np.clip(vote_ratio + edge_consistency * edge_w, 0.0, 1.0)
```

旧公式中 `edge_consistency_weight` 是混合权重（0 表示纯投票，1 表示纯边缘）。新公式中它变成了增益倍率——边缘一致性只会抬高稳定性，不会降低。

**数值对比**:

| vote_ratio | edge_consistency | edge_w | 旧值 | 新值 |
|---:|---:|---:|---:|---:|
| 0.3 | 0.8 | 0.5 | 0.55 | 0.70 |
| 0.5 | 0.9 | 0.3 | 0.62 | 0.77 |
| 0.6 | 0.7 | 0.42 | 0.64 | 0.89 |

**影响**: 边缘一致性高的像素（即使帧间投票率低）现在更容易通过核心阈值，可能扩大前景 mask，降低时序去噪效果。

**建议**: 如果这是有意的行为变更，应重命名配置键（如改为 `edge_consistency_boost`）以避免误解；如果无意，恢复旧公式。

---

### 4. 兜底图使用全局输出尺寸，不区分横竖屏（中严重度）

**文件**: `capture_manager.py:94-95, 322-324`

**问题**: 抠图失败时的兜底黑图始终从全局配置读取尺寸：

```python
output_cfg = APP_STATE["config"].get("output", {})
fallback = Image.new("RGBA", (int(output_cfg.get("width", 1080)), int(output_cfg.get("height", 1920))), ...)
```

**影响**: 横屏背景（1920x1080）下，兜底图是 1080x1920 竖屏，与合成目标尺寸不匹配。`_place_subject_on_background` 会缩放裁剪但人物比例会变形。

**修复建议**: 将背景 item 的 orientation 传入抠图流程，或在合成阶段根据 `resolve_output_size()` 创建兜底图。

---

### 5. MatAnyone 分支 elapsed_seconds 累计而非单帧（低中严重度）

**文件**: `run_yolo_seg_matting_eval.py:766, 835`

**问题**: `mt0` 在第 766 行（`process_video` 之前）设置一次，第 835 行每帧赋值 `time.perf_counter() - mt0`。第 4 帧的 elapsed_seconds 包含了 `process_video` + 前 3 帧的处理时间。

其他分支（current_aliyun、modnet、vitmatte）在帧循环内设置 `t0`，是真正的单帧耗时。

**修复建议**: 在 MatAnyone 帧循环内每帧重置 `t0`。

---

### 6. 聚合指标无条件输出哨兵值（低中严重度）

**文件**: `run_yolo_seg_matting_eval.py:239-240`

**问题**: `aggregate_comparison_metrics()` 无条件输出以下键：

```python
metrics["gpupixel_color_beauty_eval"] = aggregate_branch_metrics(...)  # 始终输出
metrics["gpupixel_color_beauty_detail"] = aggregate_gpupixel_metrics(...)  # 始终输出
metrics["matanyone_edge_detail"] = aggregate_matanyone_detail_metrics(...)  # 始终输出
```

未请求这些分支时，输出 `visitor_residual_ratio_avg=1.0`、`frame_count=0.0` 等哨兵值。

**修复建议**: 仅在对应分支有数据时输出，或添加 `"status": "not_requested"` 标记。

---

### 7. 膨胀核替换连通域分析（低中严重度）

**文件**: `subject_temporal_fusion.py:122`

**问题**: `_keep_soft_connected_to_core` 从连通域分析改为 7x7 椭圆膨胀：

```python
# 旧代码：保留与核心 8-连通的所有软区域（不限距离）
# 新代码：仅保留核心 3px 半径内的软区域
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
near_core = cv2.dilate(core_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
return core_mask | (soft_mask & near_core)
```

**影响**: 超过 3px 的细长软边缘（发丝、围巾尾端、飘动衣物）会被截断，即使它们通过其他软像素与核心连通。7px 核大小硬编码，不可配置。

---

### 8. 配置切换导致远程代码路径不可达（低严重度）

**文件**: `config.json:626, 644`

**问题**: `provider` 从 `remote_tracked_matanyone` 改为 `tracked_matanyone`，`remote_matting.enabled` 从 `true` 改为 `false`。`capture_manager.py` 中 `_is_remote_tracked_matting_enabled()` 的条件不再满足，远程 4090 matting 代码路径变为死代码。

**影响**: 若后续重新启用 `remote_matting.enabled=true` 但未改回 provider，系统会静默使用本地路径而非远程 4090 服务，无任何警告。

---

### 9. BOM 字符注入（低严重度）

**文件**: `html-page/app.js:1, html-page/index.html:1`

UTF-8 BOM（`\xEF\xBB\xBF` / U+FEFF）被添加到两个文件开头，可能是 Windows 编辑器产物。现代浏览器能容忍，但：
- 增加 3 字节无用开销
- 脚本拼接工具可能不剥离 BOM
- `index.html` 的 BOM 在 `<!DOCTYPE html>` 前产生空行，可能触发旧版 IE 怪异模式

---

### 10. download.html 移除尺寸属性导致布局偏移（低严重度）

**文件**: `html-page/download.html:17`

`<img>` 标签的 `width="800" height="1066"` 被移除，但未用 CSS `aspect-ratio` 替代。慢速网络下图片加载前布局高度坍缩为 0，加载后产生 CLS（累积布局偏移）。

**修复建议**: 添加 CSS `aspect-ratio: 3 / 4` 或保留 HTML 尺寸属性。

---

### 11. torch.load 使用 weights_only=False（低严重度）

**文件**: `modnet_matting_service.py:75`

```python
state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
```

允许任意 pickle 反序列化。在受控的一体机环境下风险较低，但违反 PyTorch >= 2.0 的安全建议。若 checkpoint 文件被替换为恶意文件，可执行任意代码。

---

## 其他观察（非问题，仅供参考）

- **上传功能移除**: `index.html` 和 `app.js` 中的图片上传分割 UI 被移除，后端 `/api/upload-segment` 端点已不存在，属于有意的死代码清理。
- **`_warp_mask_to_frame` 修复**: 旧代码对非二值 uint8 mask 做 `mask * 255` 会溢出（如 128*255=32640 mod 256=0），新代码正确处理。这是 bug 修复，不是回归。
- **`weights_only=False` + `module.` 前缀剥离**: 同时添加了 DataParallel/DistributedDataParallel checkpoint 的前缀清理，这是正确的改进。
- **横屏文本区域**: 横屏背景的 `text_region` 为 1920×0.88=1690 宽、1080×0.22=238 高，比竖屏的 994×403 小很多。长标语在横屏下字号会大幅缩小，需验证可读性。
