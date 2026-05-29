# YOLO + 阿里云抠图链路说明（游客抑制 + 毛刺治理）

本文档用于说明当前仓库内“YOLO 主体定位 + 阿里云分割”的真实执行链路，重点解释：

1. 如何减少后排/右后方游客残留  
2. 如何减少毛刺、碎边、锯齿  
3. 发生“半个人”“游客删不掉”时应看哪些指标和调哪些参数

---

## 1. 总体流程（当前实现）

代码入口：`D:\code_copy\huaita_text\capture_manager.py`

拍照任务三阶段：

1. 串行采集 4 张 `capture`
2. 并行抠图（每张走 `build_subject_cutout`）
3. 合成背景与标语

其中抠图主链路在：
- `D:\code_copy\huaita_text\ali_segment_service.py`

当前单张图处理顺序是：

```text
capture.jpg
→ SubjectLocator(YOLO person bbox)
→ 裁 ROI（可选左右收边）
→ pre-Aliyun 游客抑制（ROI 局部 inpaint/blur）
→ 阿里云 SegmentBody（输入是自然 JPG）
→ SubjectAlphaFilter（连通域主体验证）
→ post hard-clear（游客区 alpha 强制置0）
→ SubjectEdgeRefine（去碎边/羽化/有效bbox）
→ cutout.png
```

随后在 4 张 cutout 级别，新增了时序融合：

```text
4 张 cutout
→ ECC 平移配准
→ 稳定性投票融合（core+soft）
→ 小连通域清理 + 轻羽化
→ 融合后的 4 张 subject
→ 进入合成
```

代码：`D:\code_copy\huaita_text\subject_temporal_fusion.py`

---

## 2. YOLO 主体定位（不做 mask，只做 bbox）

代码：`D:\code_copy\huaita_text\subject_locator.py`

### 2.1 检测与筛选
- 使用 Ultralytics YOLO 检测 `person`（`cls_id == 0`）
- 过滤：
  - 置信度低于 `min_confidence`
  - 人体高度低于 `min_person_height_ratio * 图高`

### 2.2 主体评分（选“前景主角”）
综合分：
- 居中性（center）
- 面积（large）
- 底部更低（lower）

权重由配置控制：
- `prefer_center_weight`
- `prefer_large_weight`
- `prefer_lower_weight`

### 2.3 ROI 裁切与侧向收边
先按 `roi_expand_ratio` 扩边，再可选做“方向感知收边”：
- 若游客在主体右侧且重叠小，收紧 ROI 右边界
- 若游客在主体左侧且重叠小，收紧 ROI 左边界

目的：在阿里云输入阶段减少把旁人送进去。

---

## 3. 阿里云前游客抑制（preclean）

代码：`D:\code_copy\huaita_text\subject_visitor_suppression.py`

函数：`suppress_visitors_in_roi(...)`

核心做法：
1. 在 ROI 坐标系里构建游客遮罩（`visitor_preclean_expand_ratio` 扩边）
2. 构建主体保护区（`subject_protect_expand_ratio`）
3. 最终遮罩 = 游客区 - 主体保护区
4. 在遮罩区做填充（当前默认 `inpaint`）
5. 输出 `*_subject_cleaned_roi.jpg` 给阿里云

这一步很关键：让阿里云“尽量看不到游客”。

---

## 4. 阿里云后处理（alpha 级游客清理）

### 4.1 SubjectAlphaFilter：主连通域选择

代码：`D:\code_copy\huaita_text\subject_alpha_filter.py`

作用：阿里云返回 RGBA 后，从多个前景连通域中尽量只保留主体。

关键点：
- 先把主体/游客 bbox 映射到阿里云输出坐标（考虑 ROI 与输出尺寸差异）
- 在游客框区域降权
- 连通域分析时选与主体框重叠最大的主连通域
- `keep_nearby_component_px` 允许保留与主体靠近的小组件（防止手臂/衣角断裂）

### 4.2 post hard-clear：游客区强制置0

代码：`apply_post_alpha_hard_clear(...)`（同文件 `subject_visitor_suppression.py`）

作用：即使游客和主体粘连，也对游客删除区做硬清理：
- `alpha[x,y] = 0`（游客区）
- 主体保护区不删

这一步是“强排游客”策略的最后保险。

---

## 5. 边缘净化（毛刺治理）

代码：`D:\code_copy\huaita_text\subject_edge_refine.py`

函数：`refine_subject_edge(...)`

顺序：
1. 删除小连通域（孤岛/碎片）
2. 轻开运算去尖刺
3. 轻羽化减锯齿
4. 计算有效 alpha bbox（`effective_bbox_alpha_threshold`）

说明：
- 合成时不是直接 `getbbox()`，而是用有效 alpha bbox。
- 这样远离主体的小毛刺不会放大缩放框，减少“左下角碎边被放大”的问题。

---

## 6. 多帧时序融合（游客 + 毛刺双治理）

代码：`D:\code_copy\huaita_text\subject_temporal_fusion.py`

接入点：`capture_manager.py` 中 `_segment_captures_parallel` 之后。

### 6.1 算法
1. 选锚帧（alpha 面积最大）
2. 其余帧做 ECC 平移配准（translation only）
3. 计算稳定性图：
   - `vote_ratio`（alpha 出现频率）
   - `edge_consistency`（边缘一致性）
4. `core + soft` 双阈值：
   - `core_mask = stability >= core_thr`
   - `soft_mask = stability >= soft_thr`
   - 仅保留与 core 连通的 soft（避免主体被啃）
5. 小连通域清理 + 轻羽化
6. 将融合 alpha 映回每帧，限制每帧 alpha 上限

### 6.2 价值
- 只在少数帧出现的游客边角更容易被压制
- 单帧偶发毛刺会被“时序不稳定”淘汰
- 相比单阈值硬切，`core+soft` 更不容易把主体削坏

---

## 7. 当前关键配置（建议对照 `config.json`）

配置文件：`D:\code_copy\huaita_text\config.json`

### 7.1 主体定位
- `subject_locator.roi_expand_ratio`
- `subject_locator.roi_side_trim_enabled`
- `subject_locator.roi_side_trim_margin_ratio`
- `subject_locator.roi_side_trim_max_overlap_ratio`

### 7.2 游客抑制
- `subject_visitor_suppression.visitor_preclean_expand_ratio`
- `subject_visitor_suppression.subject_protect_expand_ratio`
- `subject_visitor_suppression.fill_mode`（当前常用 `inpaint`）
- `subject_visitor_suppression.inpaint_radius`

### 7.3 alpha 过滤
- `subject_alpha_filter.subject_box_expand_ratio`
- `subject_alpha_filter.visitor_box_expand_ratio`
- `subject_alpha_filter.keep_nearby_component_px`
- `subject_alpha_filter.alpha_threshold`

### 7.4 边缘净化
- `subject_edge_refine.min_component_area_ratio`
- `subject_edge_refine.open_kernel_px`
- `subject_edge_refine.feather_radius_px`
- `subject_edge_refine.effective_bbox_alpha_threshold`

### 7.5 时序融合
- `temporal_subject_fusion.enabled`
- `temporal_subject_fusion.min_frames`
- `temporal_subject_fusion.alpha_vote_threshold`
- `temporal_subject_fusion.edge_consistency_weight`
- `temporal_subject_fusion.noise_component_min_area_ratio`

---

## 8. Debug 产物与日志定位

目录：`D:\code_copy\huaita_text\generated\subject_debug\`

常用文件：
- `*_locator.jpg`：YOLO 候选框、主体框、ROI 框
- `*_visitor_mask.png`：游客遮罩与主体保护区
- `*_alpha_filter.png`：alpha 过滤后的框叠加
- `*_edge_refine_before.png / after.png / mask.png`
- `*_fusion_stability_map.png`、`*_fusion_alpha_after.png`

关键日志字段：
- `[SubjectLocator] candidates / score / other_people_in_roi / visitor_overlap_ratio / roi_side_trim`
- `[SubjectVisitorSuppression] preclean_pixels`
- `[SubjectAlphaFilter] alpha_before / alpha_after / removed`
- `[SubjectVisitorSuppression] post_alpha_before / post_alpha_after / removed / visitor_suppression_weak`
- `[SubjectEdgeRefine] removed_small_components / effective_bbox`
- `[TemporalFusion] alignment_success_count / alpha_stable_ratio / removed_temporal_noise_px / fallback_reason`

---

## 9. 常见问题对照

### 9.1 “为什么还有右上游客？”
优先检查：
1. `*_locator.jpg`：是否被纳入 ROI
2. `preclean_pixels` 是否接近 0
3. `visitor_suppression_weak=true` 是否频繁出现
4. `fusion` 是否 fallback（`fallback_reason` 非空）

### 9.2 “为什么出现半个人？”
常见原因：
- ROI 过窄或侧向收边过激进
- 主体保护区太小 + hard-clear 误伤
- 融合阈值过高导致 core 过小

### 9.3 “为什么左下角毛刺明显？”
优先检查：
- `removed_small_components` 是否过低
- `effective_bbox` 是否被小碎片拉大
- `fusion_stability_map` 对应区域是否低稳定但未被清除

---

## 10. 设计边界（当前方案的真实约束）

1. 不使用 YOLO segmentation mask 作为最终 alpha。  
2. 阿里云输入始终是自然 JPG（原图或 ROI 或 cleaned ROI）。  
3. 当主体与游客严重重叠且 4 帧都稳定重叠时，算法仍可能残留。  
4. 当前策略偏“强排游客优先”，允许主体边缘轻微损失。  

---

## 11. 建议调参顺序（实战）

1. 先看 `locator` 是否选对人、ROI 是否包含游客  
2. 再调 `visitor_preclean_expand_ratio` 与 `subject_protect_expand_ratio`  
3. 再调 `subject_alpha_filter`（连通域保留逻辑）  
4. 再调 `edge_refine` 去毛刺  
5. 最后调 `temporal_subject_fusion`（避免直接把阈值拉太高）  

推荐原则：每次只改一组参数，保留同批样本对比图。

