# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

本文件为 Claude Code (claude.ai/code) 在当前仓库中工作时提供指引。

## 项目概述

淮海战役纪念馆互动拍照一体机。基于 FastAPI 的 Web 服务，驱动 摄像头 → 人像分割 → 背景合成 → 标语叠加 流水线，输出 1080×1920 竖版留念海报。一体机循环展示一组红色宣传风格标语和四张可配置背景。选配 SDBM-60 激光测距传感器通过串口（CH340 UART）接入，实现人员进入范围后自动触发拍照。

## 启动应用

```powershell
python run_app.py           # 仅启动 FastAPI 服务（无 GUI 壳）
python gui_app.py           # PySide6 全屏一体机模式（EXE 打包的实际入口）
python gui_app.py --windowed  # 窗口模式调试
# 或双击 run.bat
```

访问地址 `http://127.0.0.1:10051/`。控制台页面为 `/`，拍照/一体机页面为 `/camera.html`。

**激光诊断**（独立运行，不需要主服务）：

```powershell
python laser_diagnostics.py --port COM3 --duration 10
python laser_diagnostics.py --port COM3 --max-frames 20
```

## 测试

```powershell
python -m pytest tests/ -v                     # 全部测试（当前 373 个）
python -m pytest tests/test_capture_manager.py -v  # 单文件
python -m pytest tests/test_capture_manager.py::test_fn_name -v  # 单个测试
python -m pytest tests/ -k "keyword" -v        # 按关键字筛选
```

## 打包（PyInstaller EXE）

```powershell
.\build_exe.ps1 -PythonExe python
```

输出到 `deploy/huaita_text/`。spec 文件为 `huaita_text.spec`。

## 依赖

| 文件 | 用途 |
|---|---|
| `requirements.txt` | 运行时依赖 |
| `requirements-build.txt` | PyInstaller 打包依赖 |
| `requirements-rmbg-eval.txt` | 离线评估独立依赖（不影响生产环境） |

## 架构

```
main.py                  # FastAPI 应用，路由、lifespan、服务入口（约 364 行，薄层）
run_app.py               # 启动入口，仅调用 main.run_server()
gui_app.py               # PySide6 全屏 GUI 壳，嵌入 Uvicorn + QWebEngineView（PyInstaller EXE 入口）
app_state.py             # 全局共享 APP_STATE 字典、路径常量、目录初始化、羊皮纸背景
config_manager.py        # DEFAULT_CONFIG、深度合并、乱码修复、配置读写
slogan_manager.py        # 标语展示/排版/轮播/微调解析
background_manager.py    # 背景项管理、人物/文字布局合并、轮播选择
text_renderer.py         # 字体加载、颜色解析、金色多层渲染、自动换行布局
image_composer.py        # 人物裁剪/缩放、背景合成、多版本生成
capture_manager.py       # 拍照任务状态机、互斥锁、手动/激光工作流、激光触发循环、二维码
camera_driver.py         # OpenCV 摄像头：多索引/多后端自动检测、MJPEG 推流
laser_driver.py          # SDBM-60 激光测距 UART 驱动——连续 20 Hz、触发状态机
ali_segment_service.py   # ali/ 流水线的薄封装层
baidu_segment_service.py # 百度云备选分割方案（当前未使用）
laser_diagnostics.py     # 独立 CLI 工具，用于解码 SDBM-60 UART 数据帧
runtime_paths.py         # 路径解析：开发模式 vs 冻结/PyInstaller 模式
startup_manager.py       # Windows 自启动支持（启动文件夹 / 任务计划程序）
subject_locator.py       # YOLO 人物边界框检测
subject_alpha_filter.py  # Alpha 通道过滤，优化抠图边缘
subject_visitor_suppression.py  # 合成图中路人/旁观者抑制
package_self_test.py     # 打包后 EXE 自检
subject_instance_segmentation.py  # YOLO11x-seg 实例分割，区分主体/游客实例归属
subject_instance_tracking.py  # 离线主体实例跟踪，跨连拍帧维持 track ID
subject_edge_refine.py   # Alpha 边缘净化：小连通域清理、羽化、有效 bbox 收紧
modnet_matting_service.py  # MODNet 人像 matting，实例约束 alpha（sure fg/bg + unknown）
maggie_matting_service.py  # MaGGIe 人像 matting（MODNet 替代方案，基于 ViT）
matanyone_service.py     # MatAnyone 人像 matting，约束式 alpha 抠图，支持遮挡冲突策略（visitor_priority / selected_subject_priority）
vitmatte_service.py      # ViTMatte 人像 matting（MODNet 替代方案）
rmbg_segment_service.py  # RMBG-2.0 纯本地背景移除（离线评估用，不接入线上）
yolo_seg_aliyun_service.py  # YOLO-seg + 阿里云混合管线：精确游客 inpaint + SegmentBody
subject_temporal_fusion.py  # 4 帧时序融合：ECC 配准 + 稳定性投票 + 去噪
video_recorder.py        # 带预触发缓冲区的视频录制器
capture_burst_eval.py    # 离线连拍采集工具（摄像头 → 视频 + 关键帧）
run_yolo_seg_matting_eval.py    # 离线四方评估主入口（mask / modnet / aliyun / current_aliyun）
run_yolo_seg_trimap_sweep.py    # Trimap 参数网格扫描（sure_fg / unknown / visitor_bg 膨胀量）
run_yolo_seg_instance_eval.py   # YOLO-seg 实例分割独立验证
run_rmbg_local_eval.py          # RMBG-2.0 本地背景移除效果评估
run_rmbg_ab_eval.py             # RMBG A/B 对比评估
run_tracked_video_matting_eval.py  # 跟踪式连拍视频 matting 离线评估（YOLO-seg + 多种 matting 分支）
face_beauty_service.py           # 美颜处理（磨皮、美白、锐化）
gpupixel_beauty_service.py       # GPUPixel 美颜服务封装
remote_matting_client.py         # 远程 4090 matting 服务客户端（HTTP 上传/轮询）
tracked_matting_service.py       # 跟踪式连拍 matting 服务（组合实例跟踪 + 多种 matting 后端）
run_gpupixel_background_sweep.py # GPUPixel 背景参数网格扫描评估
run_gpupixel_dered_tracked_eval.py  # GPUPixel 去红 + 跟踪式 matting 离线评估
```

### 依赖关系（无循环）

```
config_manager  ←  runtime_paths（无内部依赖）
    ↑
app_state  ←  config_manager
    ↑
slogan_manager  ←  app_state + config_manager
background_manager  ←  app_state + config_manager
    ↑        ↑
text_renderer  ←  slogan_manager + background_manager + app_state
    ↑
image_composer  ←  text_renderer + background_manager + app_state
    ↑
capture_manager  ←  image_composer + slogan_manager + app_state + remote_matting_client + tracked_matting_service
    ↑
main.py  ←  capture_manager + 所有其他模块（仅路由，不含业务逻辑）
gui_app.py  ←  main（PySide6 壳，启动 Uvicorn 后打开 QWebEngineView）
```

`subject_locator.py`、`subject_alpha_filter.py`、`subject_visitor_suppression.py` 被 `image_composer.py` 在抠图/合成阶段调用，不改变上述依赖方向。

### 离线评估子系统（不接入线上链路）

```
subject_instance_segmentation  ←  YOLO11x-seg 权重（models/*.pt，不提交 git）
         ↓
subject_instance_tracking  ←  跨连拍帧维持主体 track ID（recovery / re-id）
         ↓
modnet_matting_service  ←  MODNet 权重（models/modnet_*.ckpt，不提交 git）
maggie_matting_service  ←  MaGGIe 权重（基于 ViT，MODNet 替代方案）
matanyone_service       ←  MatAnyone 权重（约束式 alpha 抠图）
vitmatte_service        ←  ViTMatte 权重（MODNet 替代方案）
         ↓
subject_edge_refine  ←  subject_instance_segmentation（获取有效 bbox）
         ↓
subject_temporal_fusion  ←  4 帧 cutout（ECC 配准 + 稳定性投票）
         ↓
run_yolo_seg_matting_eval  ←  统筹四方对比 + 指标汇总 + sheet 输出
         ↑
yolo_seg_aliyun_service  ←  YOLO-seg 实例归属 + 阿里云 SegmentBody（第四分支）
rmbg_segment_service  ←  RMBG-2.0（transformers pipeline，纯本地，独立评估用）
         ↓
run_tracked_video_matting_eval  ←  跟踪式连拍视频 matting 评估
         ↑
capture_burst_eval  ←  离线连拍采集（摄像头 → MP4 + 关键帧）
video_recorder      ←  带预触发缓冲区的视频录制器

face_beauty_service  ←  美颜处理（无内部依赖）
gpupixel_beauty_service  ←  GPUPixel 美颜封装
         ↓
run_gpupixel_background_sweep  ←  GPUPixel 背景参数网格扫描
run_gpupixel_dered_tracked_eval  ←  GPUPixel 去红 + 跟踪式 matting 评估
```

离线评估入口脚本均通过 `--groups` 参数指定测试样本组（`generated/captures/<group_id>_*.jpg`），输出到 `generated/<eval_name>/<timestamp>/`。所有评估脚本不影响 `APP_STATE` 和线上拍照流程。

四方对比分支：`current_aliyun`（线上链路）→ `yolo_seg_mask`（纯 YOLO-seg mask）→ `yolo_seg_modnet`（YOLO-seg + MODNet 约束 alpha）→ `yolo_seg_aliyun`（YOLO-seg + 精确游客 inpaint + 阿里云）

跟踪式连拍评估（`run_tracked_video_matting_eval.py`）支持多种 matting 分支对比：`yolo_seg_modnet`、`yolo_seg_maggie`、`yolo_seg_matanyone`、`yolo_seg_vitmatte`，通过 `--branches` 指定。

**MatAnyone 遮挡冲突策略**：`matanyone_service.py` 的 `MatAnyoneConstraintConfig` 有两个关键开关：

| 参数 | 默认值 | 作用 |
|---|---|---|
| `occlusion_conflict_policy` | `"visitor_priority"` | 冲突区归属策略 |
| `contact_subject_priority_enabled` | `False` | 是否启用主体优先接触区修复 |

当设为 `selected_subject_priority` + `True` 时（即 `tracked_matanyone_subject_priority` 分支），遮挡冲突区逻辑翻转：
- 游客清理不再覆盖主体区域（`visitor_visible_clear = visitor_clear & ~subject`）
- 冲突区分为 `contact_core`（主体腐蚀核，强制还原为前景）和 `contact_edge`（过渡带，alpha 下限 ≥160）
- 避免游客清理把主体身体部位抠掉

两个 MatAnyone 分支对比：`tracked_matanyone_current`（visitor_priority）vs `tracked_matanyone_subject_priority`（主体优先），后者通过 `_subject_priority_config()` 创建专用配置，输出额外指标：`contact_conflict_px`、`contact_core_restored_px`、`contact_edge_floor_applied_px`、`visitor_visible_residual_ratio`、`subject_contact_missing_ratio`。

### 配置

`config.json` 深度合并到 `config_manager.py` 中的 `DEFAULT_CONFIG` 之上。首次运行无配置文件时，默认配置会被写出。配置文件驱动一切：摄像头选择模式、轮播标语及时序、四张背景及其独立的人物/文字布局覆盖、激光触发参数（`laser_trigger`）、文字样式（字号、金色渐变调色板、阴影/高光效果），以及通过 `text_tuning.by_slogan` 实现的逐标语文字微调。

**路径行为**：`runtime_paths.py` 解析所有目录。开发模式下，所有路径相对于仓库根目录。冻结/PyInstaller 模式下，`base_dir` 和 `resource_dir` 均指向 exe 所在目录，因此 `config.json`、`html-page/`、`fonts/`、`generated/` 均位于 exe 旁边。

**关键配置节**：

| 节 | 作用 | 关键字段 |
|---|---|---|
| `matting_api` | 抠图管线选择 | `provider`、`use_seedream`、`use_suxiaoban`、`suxiaoban`、`max_image_edge` |
| `remote_matting` | 远程 4090 matting 服务 | `enabled`、`base_url`、`upload_mode`、超时/轮询参数、`host`/`port` |
| `tracked_matting` | 跟踪式连拍 matting | `enabled`、`input_frame_count`、`output_frame_indices`、`subject_priority_enabled` |
| `subject_edge_refine` | Alpha 边缘净化 | `enabled`、`min_component_area_ratio`、`feather_radius_px` |
| `output` | 输出尺寸/质量 | `width`、`height`、`jpeg_quality` |
| `ui` | 前端超时参数 | `kiosk_idle_return_seconds`、`select_background_rotate_seconds` |
| `laser_trigger` | 激光触发参数 | 距离阈值、倒计时、冷却等 |
| `compose` | 合成叠加参数 | `top_overlay_height`、`overlay_opacity` |

### 图片处理流水线

1. `capture_manager.start_capture_task(source)` —— 获取互斥锁，启动守护线程
2. `capture_manager.process_capture_task(task_id)` —— 摄像头帧 → `image_composer.build_subject_cutout`（上传 OSS → 阿里云 SegmentBody → 下载 RGBA）→ `image_composer.compose_single_variant`（缩放人物、贴入背景、渲染标语文字）→ 保存 1080×1920 JPEG
3. **手动/激光统一三阶段并发管线**：
   - **阶段 1 — 串行采集**：`burst_count`（默认 4）帧依次拍摄，间隔 `burst_interval_seconds`（默认 0.5s）
   - **阶段 2 — 并行抠图**：`ThreadPoolExecutor(max_workers=burst_count)` → 每个槽位调用 `_cutout_with_fallback()`（3 次重试 + NotFoundFace 跳过 + 原图 RGBA 兜底），**保证每个槽位必定产出 subject，始终 4 张结果**
   - **阶段 3 — 串行合成**：逐个 `compose_single_variant` + `draw_slogan`，兜底项打 `"error": True` 标记
   - **容错**：`ali_segment_greenscreen_util.py` 下载 5xx 自动重试 3 次；`_cutout_with_fallback` 重试耗尽用原图兜底；极端失败用纯黑 1080×1920 RGBA 占位

前景人物按边界框裁剪，缩放至输出高度的 `person_layout.target_height_ratio`，通过 `center_x_ratio` + `bottom_margin` + `center_y_offset` 定位。每个背景项可覆盖所有上述参数。

### 标语渲染

`text_renderer.draw_slogan` 处理多行布局、字号适配和装饰性文字样式。关键行为：

- 支持 1–3 行布局，按行数可配置字号缩放（`row2_font_scale`、`row3_font_scale`）
- `gold_layered` 样式模式渲染多层金属质感金色文字（阴影 → 深色描边 → 渐变填充 → 内发光 → 高光带 → 高亮）
- 每个背景的 `text_layout` 可定义 `text_region`（基于比例的边界框），或回退到传统的全宽顶部横幅
- `slogan_manager.resolve_text_tuning` 将特定标语文字映射到覆盖参数（font_scale、y_offset、preferred_lines）。`by_slogan.forced_lines` 条目被显式剥离——所有标语使用统一的 1→2→3 行自动布局
- 中文字体回退链：`fonts/default.ttf` → Windows 系统字体（msyh、simhei、simsun）
- 乱码防御：`config_manager.normalize_mojibake_text` 修复 UTF-8 字节被错误按 Latin-1 解码的问题

### 激光触发状态机（`laser_driver.py`）

状态：`MANUAL_ONLY` → `IDLE` → `COUNTDOWN` → `TRIGGERED`/`COOLDOWN` → 回到 `IDLE`

- 发送 `AA0000200001000627` 启动连续 20 Hz 测距模式
- 解析 13 字节结果帧：`AA0000220003` + 4 字节距离值（mm，大端序）+ 2 字节质量值 + 校验和
- `capture_manager.laser_trigger_loop` 以 10 Hz 频率轮询 `laser_driver.tick()` 和 `consume_trigger()`
- 自动检测 CH340 USB 串口（`_is_ch340_port`），通过 `app_state.persist_laser_serial_port` 将发现的端口持久化回 `config.json`
- `require_leave_before_retrigger` 确保人员离开超过 `leave_min_cm` 后才能开始新的倒计时

### 前端

静态文件从 `html-page/` 通过 FastAPI `StaticFiles` 挂载提供。主要页面：
- `index.html` —— 控制台，含实时预览、手动拍照按钮、模板轮播显示
- `camera.html` —— 面向一体机的拍照页面，通过心跳（`/api/camera-page-active`）控制激光触发开关
- `kiosk-wait.html` —— 一体机待机等待页，激光触发进入相机页
- `select.html` —— 四幅合成结果选择页，底部操作提示 + 右下角金色 btn-home 返回按钮
- `view.html` —— 扫码下载页，顶部标题、中间预览图、底部 QR 码卡片 + 返回按钮 + 倒计时
- `download.html` —— 背景选择和结果浏览

**页面跳转逻辑**（均由前端 JS 驱动，依赖激光传感器）：

| 方向 | 触发条件 | 检测机制 | 时长 |
|------|----------|----------|------|
| 等待页 → camera | 人进入激光检测范围（100-180cm） | `initKioskWaitPage()` 每 200ms 轮询 `/api/laser-status`，`person_in_range` 连续 ≥3 次 | ≥600ms |
| camera → 等待页 | 人离开激光检测范围 | `pollFullscreenState()` 每 500ms 轮询，`person_in_range` 连续为 false ≥8 次 | ≥4s |
| camera → select | 激光倒计时触发拍照完成 | `syncLaserTaskTransitionByLatestTask()` 每 500ms 轮询 `/api/latest-task`，`status === "completed"` 或 `"timeout"` | — |
| select → 等待页 | 30s 无操作 | `createIdleReturnController` 监听点击/触摸事件，超时跳转 `/kiosk-wait.html` | 30s |
| view → select | 30s 无操作或点击返回按钮 | 同上 + 底部返回按钮跳转 `select.html` | 30s |

**缓存策略**：`main.py` 中 `_NoCacheStaticFiles` 子类为所有静态资源响应添加 `Cache-Control: no-cache` 头，`serve_page()` 同样添加。修改 CSS/JS 后无需手动更新版本号，浏览器自动重新验证。

**view.html 布局**（1080×1920 竖屏）：
- 顶部 pill 形 hero（标题"扫码保存照片"）
- 中间图片预览（`max-height: 54dvh`）
- QR 码卡片（暖色渐变背景，金色边框，二维码 180-300px）
- 底部 pill 形 footer（"返回选图"按钮 + 倒计时文字并排居中）
- 各区域阴影均已收紧，gap 增大至 1.8rem，不再跨元素重叠

### 抠图管线

三条独立管线通过 `config.json` → `matting_api` 布尔开关切换，优先级：`use_suxiaoban` > `use_seedream` > 默认 ali。

| 管线 | 开关 | 路径 | 流程 |
|---|---|---|---|
| **原版** | 默认（均 false） | `ali/` | 输入图 → OSS 上传 → SegmentBody → 下载 RGBA |
| **Seedream** | `use_seedream: true` | `ali_seedream/ali/` | 输入图 → [YOLO 本地抠图(关)] → Seedream 绿幕 → 缩放 1999px → OSS 上传 → SegmentBody → 下载 RGBA |
| **速销班** | `use_suxiaoban: true` | `ali_seedream_chinamobile/ali/` | 输入图 → 速销班网关图生图绿幕 → 缩放 → OSS 上传 → SegmentBody → 下载 RGBA |

三条管线共享 `AliOssSegmentPipeline` 的预处理链模式。`ali_segment_service.py:_create_pipeline()` 根据开关选择管线，其余代码无感知。

Seedream 管线依赖 `pyyaml`、`volcengine-python-sdk[ark]`、`ultralytics`；速销班管线依赖 `SuxiaobanImageGenerationsClient`。各模块在 `__init__` 中按需延迟导入，未启用时不加载。

密钥：`ali/` 和 `ali_seedream/ali/` 各自硬编码 AccessKey。Seedream 的 ARK API Key 在 `ali_seedream/ali/volcengine_segment_greenscreen.py:49`。速销班 API 配置在 `config.json` → `matting_api.suxiaoban`。详细文档见 `ali/README.md`。

## 重要说明

- 运行测试见上方「测试」章节
- 仓库根目录的 `config.json` 是**当前部署的实际生效配置**（不是模板）——包含完整的标语集、背景路径和激光端口
- `runtime_paths.py` 处理开发模式与冻结模式的区别；任何新增文件路径都应通过 `get_app_paths()` 获取
- 摄像头自动检测会遍历多个索引和后端（DSHOW、ANY、MSMF），通过 `preferred_indices` 优先选择外接摄像头
- `CameraDriver.get_frame()` 返回 numpy 数组的**副本**——调用方无需自行拷贝
- `APP_STATE` 是 `app_state.py` 中唯一的全局共享字典；所有模块直接导入使用，而非依赖注入
- 新增背景模板时，只需修改 `config.json`（在 `background_set.items` 中添加一项，可选覆盖 `person_layout` 和 `text_layout`）
- 新增标语时，在 `config.json` 的 `rotation.slogans` 中添加条目；逐标语微调可选配置在 `text_tuning.by_slogan` 中

## 标语统计

`config.json` 中 `rotation.slogans` 共 75 条标语，按 `row` 字段分为三类：

| row | 含义 | 数量 |
|---|---|---|
| 1 | 单行标语 | 33 条 |
| 2 | 双行标语 | 40 条 |
| 3 | 三行标语 | 2 条 |

每个 slogans 条目结构：

```json
{
  "content": "标语原文，用 \\n 分隔各行",
  "row": 1  // 1 / 2 / 3
}
```

标语行数决定后端渲染时的字号和间距策略：
- row=1：单行，字号最大
- row=2：双行，`row2_font_scale: 0.94` 控制字号缩放
- row=3：三行，`row3_font_scale: 0.78` 控制字号缩放，可配合 `row3_width_ratio` 调整宽度

`text_tuning.by_slogan` 中的 `forced_lines`（显式指定每行断字位置）和 `preferred_lines`（偏好行数）会覆盖默认的自动换行布局。在 `slogan_manager.py` 中，`forced_lines` 被解析后实际拆分内容并移除原始换行符，剩余标语均由统一的 1→2→3 行自动布局处理。

## 隐形文本框（text_region）

四幅背景图的 `text_region` 配置相同，输出画布为 1080×1920：

| 参数 | 配置值 | 像素值 |
|---|---|---|
| 上边距 | `margin_top_ratio: 0.04` | 1920 × 0.04 = **77 px** |
| 宽度 | `width_ratio: 0.92` | 1080 × 0.92 = **994 px** |
| 高度 | `height_ratio: 0.21` | 1920 × 0.21 = **403 px** |
| 内边距 | `inner_padding_px: 0` | **0 px** |

文本框尺寸 **994 × 403 像素**，位于画面顶部下方 77px 处，水平居中（左右各约 43px 边距）。目前四幅图完全一致，如需逐背景差异化，可在各 `background_set.items[].text_layout.text_region` 中覆盖。

## 逐标语字号上限（max_font_size）

每条标语在 `text_tuning.by_slogan` 中有一个 `max_font_size` 字段（整数，px），作为该标语在 994px 文本框内的字号绝对值上限：

- 短标语（≤8 字/行）→ 上限 ≥124，不影响当前渲染（系统 font_size_max=124 已是更紧约束）
- 长标语（9–10 字/行）→ 上限 90–110，在 row 缩放后生效，防止文字溢出文本框

`text_renderer.py:635-638` 负责应用此上限：在所有 row 缩放之后，若 tuning 中有 `max_font_size`，则 `font_size_max = min(font_size_max, max_font_size)`。

## 批量标语预览

`text_region_preview/batch_render.py` 可批量渲染所有标语到指定背景，用于逐条检查排版效果：

```powershell
python text_region_preview/batch_render.py
```

生成 `text_region_preview/1/` ~ `4/` 四个子目录，每个目录含 75 张该背景下的标语渲染图。

## 离线评估命令

```powershell
python run_yolo_seg_matting_eval.py                                    # 四方对比（默认两组 hard case）
python run_yolo_seg_matting_eval.py --include-current-aliyun           # 含阿里云当前链路
python run_yolo_seg_matting_eval.py --include-yolo-seg-aliyun          # 含第四分支（YOLO-seg + 阿里云）
python run_yolo_seg_trimap_sweep.py                                    # Trimap 参数网格扫描
python run_yolo_seg_instance_eval.py                                   # YOLO-seg 实例归属独立验证
python run_rmbg_local_eval.py                                          # RMBG-2.0 原始效果评估
python run_rmbg_local_eval.py --cpu                                    # CPU 模式
python run_tracked_video_matting_eval.py                               # 跟踪式连拍 matting 评估
python run_tracked_video_matting_eval.py --branches yolo_seg_modnet yolo_seg_matanyone  # 指定分支
python capture_burst_eval.py                                           # 离线连拍采集（16 帧 → video + 4 个关键帧）
python run_gpupixel_background_sweep.py                                # GPUPixel 背景参数网格扫描
python run_gpupixel_dered_tracked_eval.py                              # GPUPixel 去红 + 跟踪式 matting 评估
```

离线评估独立依赖（不影响生产环境）：

```powershell
python -m pip install -r requirements-rmbg-eval.txt
```

## Git

远程仓库：`https://github.com/frostdogstarscream/huaita_text.git`

活跃分支（部分）：
- `codex/yolo-seg-modnet-eval` — YOLO-seg + MODNet 离线评估 + 跟踪式连拍 matting（当前分支）
- `feature/test-aliyun-seedream-cutout` — 接入 Seedream 绿幕预处理
- `codex/GUI` — GUI 壳相关
- `codex/adapt-new-laser-sensor` — 新激光传感器适配
- `codex/fast-subject-extraction` — 快速主体提取
- `feature/concurrent-upload-api-speedup` — 并发上传加速
- `feature/suxiaoban-integration` — 速销班网关集成

## 异常处理

### 自定义异常（19 个）

| 异常类 | 文件:行 | 基类 | 用途 |
|---|---|---|---|
| `CaptureBusyError` | `capture_manager.py:27` | `RuntimeError` | 拍照互斥，已有拍照进行中 |
| `CameraUnavailableError` | `camera_driver.py:14` | `RuntimeError` | 摄像头未启动或不可用 |
| `FrameUnavailableError` | `camera_driver.py:18` | `RuntimeError` | 帧为 None 或编码失败 |
| `CameraFocusUnsupportedError` | `camera_driver.py:22` | `RuntimeError` | 摄像头不支持自动对焦 |
| `LaserUnavailableError` | `laser_driver.py:17` | `RuntimeError` | 激光传感器不可用 |
| `AliSegmentError` | `ali_segment_service.py:29` | `RuntimeError` | 阿里云人像分割失败 |
| `BaiduSegmentError` | `baidu_segment_service.py:12` | `RuntimeError` | 百度云人像分割失败 |
| `ConfigError` | `config_manager.py:238` | `RuntimeError` | 配置错误基类 |
| `ConfigLoadError` | `config_manager.py:242` | `ConfigError` | 配置加载失败 |
| `ConfigSaveError` | `config_manager.py:246` | `ConfigError` | 配置保存失败 |
| `AutostartError` | `startup_manager.py:20` | `RuntimeError` | Windows 自启动设置失败 |
| `ModnetMattingError` | `modnet_matting_service.py:16` | `RuntimeError` | MODNet 推理失败（含 stage 标记） |
| `RmbgSegmentError` | `rmbg_segment_service.py:11` | `RuntimeError` | RMBG-2.0 分割失败（含 stage 标记） |
| `MaggieError` | `maggie_matting_service.py:16` | `RuntimeError` | MaGGIe 推理失败（含 stage 标记） |
| `MatAnyoneError` | `matanyone_service.py:14` | `RuntimeError` | MatAnyone 推理失败（含 stage 标记） |
| `VitmatteError` | `vitmatte_service.py:16` | `RuntimeError` | ViTMatte 推理失败（含 stage 标记） |
| `RemoteMattingError` | `remote_matting_client.py:16` | `RuntimeError` | 远程 matting 服务调用失败 |
| `TrackedMattingError` | `tracked_matting_service.py:25` | `RuntimeError` | 跟踪式连拍 matting 失败 |
| `VideoRecorderError` | `video_recorder.py:13` | `RuntimeError` | 视频录制失败 |

### FastAPI 路由错误响应

| 路由 | 状态码 | 异常 | 位置 |
|---|---|---|---|
| 页面路由 | 404 | 文件不存在 | `main.py:82` |
| `POST /api/sync-time` | 400 | `TypeError/ValueError`（参数非法） | `main.py:202-207` |
| `POST /api/capture` | 409 | `CaptureBusyError` | `main.py:226-227` |
| `GET /api/task/{id}` | 404 | 任务不存在 | `main.py:236` |
| `GET /api/camera/frame` | 503 | `CameraUnavailableError/FrameUnavailableError` | `main.py:255-256` |
| `GET /api/qr` | 500 | 任意 Exception（过宽） | `main.py:272-273` |

**缺失**：无全局异常处理器，未预期异常直接返回 Starlette 原生 500。

### 好的实践

**重试与自动恢复**：
- 摄像头断连后自动重探测所有索引和后端（`camera_driver.py:120-139`），间隔 1s
- 摄像头 DSHOW 缓冲清理：`_reader_loop` 每次读前 `grab()×3` 丢弃缓冲旧帧，再 `retrieve()` 取最新帧，避免连拍得到相同画面
- 激光串口断连后重探测 CH340 端口（`laser_driver.py:390-417`），间隔 1s
- 前端 MJPEG 流 2.2s 超时后降级为 250ms 单帧轮询（`app.js:182-206`）
- 字体加载 5 级回退：`default.ttf → msyh → simhei → simsun → PIL default`（`text_renderer.py:18-33`）
- 前端 `resolveSelectTask()` 3 级降级：URL 参数 → sessionStorage → API（`app.js:823-844`）

**渲染降级**：
- `draw_slogan` 金属字渲染异常时降级为普通 `draw.text()`（`text_renderer.py:797-804`）
- 配置乱码自动修复：`normalize_mojibake_text`（`config_manager.py:146`）

**资源清理**：
- `lifespan` finally 块确保停止激光、摄像头（`main.py:57-65`）
- `process_capture_task` finally 释放互斥锁（`capture_manager.py:162-163`）
- `_NoCacheStaticFiles` 子类为所有静态资源添加 `Cache-Control: no-cache`，修改 CSS/JS 后浏览器自动重验证，无需手动维护版本号

### 已知缺口（按严重度排序）

1. **阿里云全链路无保护** — `ali/` 流水线（OSS 上传、预签名、SegmentBody、下载）零 try/except。网络超时/5xx/认证失败直接穿透。外部已有 `_cutout_with_fallback` 重试+兜底和下载 5xx 重试，但 OSS 上传、预签名、SegmentBody API 自身无超时配置。
2. **config 读写无保护** — `load_config()/save_config()`（`config_manager.py:179-197`）无异常处理，JSON 损坏或磁盘满直接崩溃。
3. **`process_capture_task` 异常过宽** — `except Exception` 不区分可重试错误（网络超时）和致命错误（内存不足），手动模式单张出错整批丢弃。
4. **图片 I/O 无保护** — `Image.open()/save()` 在合成路径中无 try/except（`image_composer.py:61,91`）。
5. **前端无熔断** — 摄像头轮询 250ms/次，失败无退避，持续产生无效请求。
6. **无全局 FastAPI 异常处理器** — 未预期异常直接暴 500 + 堆栈。
7. **激光触发循环** — `except Exception` 写错误文本后不尝试恢复，守护线程静默停止。
