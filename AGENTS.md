# AGENTS.md

本文件为 OpenCode / Codex 在当前仓库中工作时提供指引。

## 项目概述

淮海战役纪念馆互动拍照一体机。基于 FastAPI 的 Web 服务，驱动 **摄像头 → 人像分割 → 背景合成 → 标语叠加** 流水线，输出 1080×1920 竖版留念海报。选配 SDBM-60 激光测距传感器通过串口（CH340 UART）接入，实现人员进入范围后自动触发拍照。一体机通过 PySide6 全屏 GUI 壳（QWebEngineView）运行。

## 启动应用

```powershell
python gui_app.py                  # PySide6 全屏一体机模式（也是 EXE 打包入口）
python gui_app.py --windowed       # 窗口模式调试
python run_app.py                  # 仅启动 FastAPI 服务（无 GUI 壳）
# 或双击 run.bat
```

访问地址 `http://127.0.0.1:10051/`。控制台页面为 `/`，一体机入口为 `/kiosk-wait.html`。

**激光诊断**（独立运行，不需要主服务）：

```powershell
python laser_diagnostics.py --port COM3 --duration 10
python laser_diagnostics.py --port COM3 --max-frames 20
```

## 测试

```powershell
python -m pytest tests/ -v                       # 全部测试（当前 359 个）
python -m pytest tests/test_capture_manager.py -v # 单文件
python -m pytest tests/test_capture_manager.py::test_fn_name -v  # 单个测试
python -m pytest tests/ -k "keyword" -v            # 按关键字筛选
```

无 `pytest.ini` 或 `pyproject.toml` 配置——pytest 完全依赖 `tests/conftest.py` 中的共享 fixtures（`temp_dir`、`mock_config`、`patched_app_state`）。

## 打包（PyInstaller EXE）

```powershell
.\build_exe.ps1 -PythonExe python
```

输出到 `deploy/huaita_text/`。spec 文件为 `huaita_text.spec`。打包入口是 `gui_app.py`（含 `--self-test` 自检模式）。打包后目录结构：`huaita_text.exe` + `_internal/` + `config.json` + `html-page/` + `fonts/` + `generated/`。

## 架构

```
main.py                  # FastAPI 应用，路由、lifespan、服务入口（约 369 行，薄层）
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
ali_segment_service.py   # 抠图管线薄封装层（ali / seedream / chinamobile 三选一）
baidu_segment_service.py # 百度云备选分割方案（当前未使用）
laser_diagnostics.py     # 独立 CLI 工具，用于解码 SDBM-60 UART 数据帧
runtime_paths.py         # 路径解析：开发模式 vs 冻结/PyInstaller 模式
startup_manager.py       # Windows 自启动支持（启动文件夹 / 任务计划程序）
package_self_test.py     # 打包后 EXE 自检

# --- 人像处理管线（生产链路） ---
subject_locator.py       # YOLO 人物边界框检测
subject_alpha_filter.py  # Alpha 通道过滤，优化抠图边缘
subject_edge_refine.py   # Alpha 边缘净化：小连通域清理、羽化、有效 bbox 收紧
subject_visitor_suppression.py  # 合成图中路人/旁观者抑制

# --- 离线评估子系统（不接入线上链路） ---
subject_instance_segmentation.py  # YOLO11x-seg 实例分割，区分主体/游客实例归属
subject_instance_tracking.py      # 离线主体实例跟踪，跨连拍帧维持 track ID
subject_temporal_fusion.py        # 4 帧时序融合：ECC 配准 + 稳定性投票 + 去噪
modnet_matting_service.py         # MODNet 人像 matting（实例约束 alpha）
maggie_matting_service.py         # MaGGIe 人像 matting（基于 ViT，MODNet 替代方案）
matanyone_service.py              # MatAnyone 人像 matting，约束式 alpha 抠图
vitmatte_service.py               # ViTMatte 人像 matting（MODNet 替代方案）
rmbg_segment_service.py           # RMBG-2.0 纯本地背景移除（离线评估用）
yolo_seg_aliyun_service.py        # YOLO-seg + 阿里云混合管线
video_recorder.py                 # 带预触发缓冲区的视频录制器
capture_burst_eval.py             # 离线连拍采集工具（摄像头 → 视频 + 关键帧）

# --- 离线评估入口脚本 ---
run_yolo_seg_matting_eval.py      # 四方对比主入口（mask/modnet/aliyun/current_aliyun）
run_yolo_seg_trimap_sweep.py      # Trimap 参数网格扫描
run_yolo_seg_instance_eval.py     # YOLO-seg 实例分割独立验证
run_rmbg_local_eval.py            # RMBG-2.0 本地背景移除效果评估
run_rmbg_ab_eval.py               # RMBG A/B 对比评估
run_tracked_video_matting_eval.py # 跟踪式连拍视频 matting 离线评估
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
capture_manager  ←  image_composer + slogan_manager + app_state
    ↑
main.py  ←  capture_manager + 所有其他模块（仅路由，不含业务逻辑）
gui_app.py  ←  main（PySide6 壳，启动 Uvicorn 后打开 QWebEngineView）
```

`subject_locator.py`、`subject_alpha_filter.py`、`subject_edge_refine.py`、`subject_visitor_suppression.py` 被 `image_composer.py` 和 `ali_segment_service.py` 在抠图/合成阶段调用，不改变上述依赖方向。

### 配置

`config.json` 深度合并到 `config_manager.py` 中的 `DEFAULT_CONFIG` 之上。首次运行无配置文件时，默认配置会被写出。配置文件驱动一切：摄像头选择模式、轮播标语及时序、四张背景及其独立的人物/文字布局覆盖、激光触发参数、文字样式（字号、金色渐变调色板、阴影/高光效果），以及通过 `text_tuning.by_slogan` 实现的逐标语文字微调。

**路径行为**：`runtime_paths.py` 解析所有目录。开发模式下，所有路径相对于仓库根目录。冻结/PyInstaller 模式下，`base_dir` 和 `resource_dir` 均指向 exe 所在目录，因此 `config.json`、`html-page/`、`fonts/`、`generated/` 均位于 exe 旁边。

### GUI 壳（`gui_app.py`）

PySide6 全屏一体机外壳。在单独线程中启动 Uvicorn，主线程持有 Qt 事件循环，通过 QWebEngineView 渲染前端页面。关键行为：

- 默认全屏；`--windowed` 切换为可缩放窗口调试
- 启动时轮询 `http://127.0.0.1:{port}/` 直到服务就绪（默认 30s 超时），然后加载 `/kiosk-wait.html`
- 支持 `--url` 覆盖起始页面路径
- `--autostart apply|install|uninstall|status` 管理 Windows 自启动（调用 `startup_manager.py`）

### 自启动（`startup_manager.py`）

通过 `config.json` → `autostart` 节控制：

- `autostart.enabled: true` 启用
- `autostart.method`: `"startup_folder"`（默认，免管理员权限）或 `"task_scheduler"`（需管理员权限）
- 打包后的 `install_autostart.bat` / `uninstall_autostart.bat` / `autostart_status.bat` 直接操作

### 图片处理流水线

1. `capture_manager.start_capture_task(source)` —— 获取互斥锁，启动守护线程
2. `capture_manager.process_capture_task(task_id)` —— 摄像头帧 → `image_composer.build_subject_cutout`（上传 OSS → 阿里云 SegmentBody → 下载 RGBA）→ `image_composer.compose_single_variant`（缩放人物、贴入背景、渲染标语文字）→ 保存 1080×1920 JPEG
3. **手动/激光统一三阶段并发管线**：
   - **阶段 1 — 串行采集**：`burst_count`（默认 4）帧依次拍摄，间隔 `burst_interval_seconds`（默认 0.5s）
   - **阶段 2 — 并行抠图**：`ThreadPoolExecutor(max_workers=burst_count)` → 每个槽位调用 `_cutout_with_fallback()`（3 次重试 + NotFoundFace 跳过 + 原图 RGBA 兜底），**保证每个槽位必定产出 subject，始终 4 张结果**
   - **阶段 3 — 串行合成**：逐个 `compose_single_variant` + `draw_slogan`，兜底项打 `"error": True` 标记
   - **容错**：`ali_segment_greenscreen_util.py` 下载 5xx 自动重试 3 次；`_cutout_with_fallback` 重试耗尽用原图兜底；极端失败用纯黑 1080×1920 RGBA 占位

前景人物按边界框裁剪，缩放至输出高度的 `person_layout.target_height_ratio`，通过 `center_x_ratio` + `bottom_margin` + `center_y_offset` 定位。每个背景项可覆盖所有上述参数。

### 抠图管线

三套管线通过 `config.json` → `matting_api` 开关选择：

| 管线 | 路径 | 流程 |
|---|---|---|
| **原版 Ali** | `ali/` | 输入图 → OSS 上传 → SegmentBody → 下载 RGBA |
| **Seedream** | `ali_seedream/ali/` | 输入图 → Seedream 绿幕 → 缩放 1999px → OSS 上传 → SegmentBody → RGBA |
| **速销班** | `ali_seedream_chinamobile/ali/` | 输入图 → 速销班网关图生图 API → OSS 上传 → SegmentBody → RGBA |

`ali_segment_service.py:_create_pipeline()` 优先级：`use_suxiaoban` > `use_seedream` > 默认 ali。Seedream 管线依赖 `pyyaml`、`volcengine-python-sdk[ark]`、`ultralytics`，按需延迟导入，未启用时不加载。

**密钥位置**：`ali/` 和 `ali_seedream/ali/` 各自硬编码 AccessKey。Seedream ARK API Key 在 `ali_seedream/ali/volcengine_segment_greenscreen.py`。速销班 API Key 在 `ali_seedream_chinamobile/ali/suxiaobanengine_segment_greenscreen.py`。详细文档见 `ali/README.md`。

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
|---|---|---|---|
| 等待页 → camera | 人进入激光检测范围 | `initKioskWaitPage()` 每 200ms 轮询，连续 ≥3 次 | ≥600ms |
| camera → 等待页 | 人离开激光检测范围 | `pollFullscreenState()` 每 500ms 轮询，连续 false ≥8 次 | ≥4s |
| camera → select | 激光倒计时触发拍照完成 | `syncLaserTaskTransitionByLatestTask()` 每 500ms 轮询，status completed/timeout | — |
| select → 等待页 | 30s 无操作 | `createIdleReturnController` 监听点击/触摸事件，超时 | 30s |
| view → select | 30s 无操作或点击返回按钮 | 同上 + 底部返回按钮 | 30s |

**缓存策略**：`main.py` 中 `_NoCacheStaticFiles` 子类为所有静态资源响应添加 `Cache-Control: no-cache` 头。修改 CSS/JS 后无需手动更新版本号，浏览器自动重新验证。

### 离线评估子系统

离线评估脚本均通过 `--groups` 参数指定测试样本组（`generated/captures/<group_id>_*.jpg`），输出到 `generated/<eval_name>/<timestamp>/`。所有评估脚本不影响 `APP_STATE` 和线上拍照流程。

```powershell
python run_yolo_seg_matting_eval.py                              # 四方对比（默认两组 hard case）
python run_yolo_seg_matting_eval.py --include-current-aliyun     # 含阿里云当前链路
python run_yolo_seg_matting_eval.py --include-yolo-seg-aliyun    # 含第四分支
python run_yolo_seg_trimap_sweep.py                              # Trimap 参数网格扫描
python run_yolo_seg_instance_eval.py                             # YOLO-seg 实例归属独立验证
python run_rmbg_local_eval.py                                    # RMBG-2.0 原始效果评估
python run_rmbg_local_eval.py --cpu                              # CPU 模式
python run_tracked_video_matting_eval.py                         # 跟踪式连拍 matting 评估
python run_tracked_video_matting_eval.py --branches yolo_seg_modnet yolo_seg_matanyone
python capture_burst_eval.py                                     # 离线连拍采集（16 帧 → video + 4 关键帧）
```

四方对比分支：`current_aliyun` → `yolo_seg_mask` → `yolo_seg_modnet` → `yolo_seg_aliyun`

离线评估独立依赖：`python -m pip install -r requirements-rmbg-eval.txt`

**MatAnyone 遮挡冲突策略**（`matanyone_service.py`）：

| 参数 | 默认值 | 作用 |
|---|---|---|
| `occlusion_conflict_policy` | `"visitor_priority"` | 冲突区归属策略 |
| `contact_subject_priority_enabled` | `False` | 是否启用主体优先接触区修复 |

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
  "row": 1
}
```

标语行数决定后端渲染时的字号和间距策略：
- row=1：单行，字号最大
- row=2：双行，`row2_font_scale: 0.94` 控制字号缩放
- row=3：三行，`row3_font_scale: 0.78` 控制字号缩放，可配合 `row3_width_ratio` 调整宽度

## 隐形文本框（text_region）

四幅背景图的 `text_region` 配置相同，输出画布为 1080×1920：

| 参数 | 配置值 | 像素值 |
|---|---|---|
| 上边距 | `margin_top_ratio: 0.04` | 1920 × 0.04 = **77 px** |
| 宽度 | `width_ratio: 0.92` | 1080 × 0.92 = **994 px** |
| 高度 | `height_ratio: 0.21` | 1920 × 0.21 = **403 px** |
| 内边距 | `inner_padding_px: 0` | **0 px** |

文本框尺寸 **994 × 403 像素**，位于画面顶部下方 77px 处，水平居中。

## 逐标语字号上限（max_font_size）

每条标语在 `text_tuning.by_slogan` 中有一个 `max_font_size` 字段（整数，px），作为该标语在 994px 文本框内的字号绝对值上限。`text_renderer.py` 在所有 row 缩放之后应用此上限：`font_size_max = min(font_size_max, max_font_size)`。

## 批量标语预览

```powershell
python text_region_preview/batch_render.py
```

生成 `text_region_preview/1/` ~ `4/` 四个子目录，每个目录含 75 张该背景下的标语渲染图。

## 重要说明

- 运行测试：`python -m pytest tests/ -v`（当前 359 个测试）
- 仓库根目录的 `config.json` 是**当前部署的实际生效配置**（不是模板）——包含完整的标语集、背景路径和激光端口
- `runtime_paths.py` 处理开发模式与冻结模式的区别；任何新增文件路径都应通过 `get_app_paths()` 获取
- 摄像头自动检测会遍历多个索引和后端（DSHOW、ANY、MSMF），通过 `preferred_indices` 优先选择外接摄像头
- `CameraDriver.get_frame()` 返回 numpy 数组的**副本**——调用方无需自行拷贝
- `APP_STATE` 是 `app_state.py` 中唯一的全局共享字典；所有模块直接导入使用，而非依赖注入
- 新增背景模板时，只需修改 `config.json`（在 `background_set.items` 中添加一项，可选覆盖 `person_layout` 和 `text_layout`）
- 新增标语时，在 `config.json` 的 `rotation.slogans` 中添加条目；逐标语微调可选配置在 `text_tuning.by_slogan` 中
- **EXE 打包入口是 `gui_app.py`**，不是 `run_app.py`。打包后用 `huaita_text.exe --self-test` 自检
- **`run.bat` 调用 `gui_app.py`**，不是 `run_app.py`

## 异常处理

### 自定义异常（17 个）

| 异常类 | 文件 | 基类 | 用途 |
|---|---|---|---|
| `CaptureBusyError` | `capture_manager.py` | `RuntimeError` | 拍照互斥，已有拍照进行中 |
| `CameraUnavailableError` | `camera_driver.py` | `RuntimeError` | 摄像头未启动或不可用 |
| `FrameUnavailableError` | `camera_driver.py` | `RuntimeError` | 帧为 None 或编码失败 |
| `CameraFocusUnsupportedError` | `camera_driver.py` | `RuntimeError` | 摄像头不支持手动对焦 |
| `LaserUnavailableError` | `laser_driver.py` | `RuntimeError` | 激光传感器不可用 |
| `AliSegmentError` | `ali_segment_service.py` | `RuntimeError` | 阿里云人像分割失败（带 `provider`/`stage` 属性） |
| `BaiduSegmentError` | `baidu_segment_service.py` | `RuntimeError` | 百度云人像分割失败 |
| `ConfigError` | `config_manager.py` | `RuntimeError` | 配置操作失败（基类） |
| `ConfigLoadError` | `config_manager.py` | `ConfigError` | 配置加载失败 |
| `ConfigSaveError` | `config_manager.py` | `ConfigError` | 配置保存失败 |
| `AutostartError` | `startup_manager.py` | `RuntimeError` | Windows 自启动操作失败 |
| `ModnetMattingError` | `modnet_matting_service.py` | `RuntimeError` | MODNet 推理失败（带 `stage` 属性） |
| `RmbgSegmentError` | `rmbg_segment_service.py` | `RuntimeError` | RMBG-2.0 分割失败（带 `stage` 属性） |
| `MaggieError` | `maggie_matting_service.py` | `RuntimeError` | MaGGIe 推理失败（带 `stage` 属性） |
| `MatAnyoneError` | `matanyone_service.py` | `RuntimeError` | MatAnyone 推理失败（带 `stage` 属性） |
| `VitmatteError` | `vitmatte_service.py` | `RuntimeError` | ViTMatte 推理失败（带 `stage` 属性） |
| `VideoRecorderError` | `video_recorder.py` | `RuntimeError` | 视频录制失败 |

**异常层级**：仅 `ConfigError` 有子类（`ConfigLoadError`、`ConfigSaveError`）。6 个 matting/segment 异常类带有 `stage` 属性标记失败阶段。无全局 `@app.exception_handler`，各路由自行 catch 后转 HTTPException。

### 好的实践

**重试与自动恢复**：
- 摄像头断连后自动重探测所有索引和后端（`camera_driver.py`），间隔 1s
- 摄像头 DSHOW 缓冲清理：`_reader_loop` 每次读前 `grab()×3` 丢弃缓冲旧帧，再 `retrieve()` 取最新帧
- 激光串口断连后重探测 CH340 端口（`laser_driver.py`），间隔 1s
- 前端 MJPEG 流 2.2s 超时后降级为 250ms 单帧轮询
- 字体加载 5 级回退：`default.ttf → msyh → simhei → simsun → PIL default`
- 前端 `resolveSelectTask()` 3 级降级：URL 参数 → sessionStorage → API

**渲染降级**：
- `draw_slogan` 金属字渲染异常时降级为普通 `draw.text()`
- 配置乱码自动修复：`normalize_mojibake_text`

**资源清理**：
- `lifespan` finally 块确保停止激光、摄像头
- `process_capture_task` finally 释放互斥锁
- `_NoCacheStaticFiles` 为所有静态资源添加 `Cache-Control: no-cache`

### 已知缺口（按严重度排序）

1. **阿里云全链路无保护** — `ali/` 流水线（OSS 上传、预签名、SegmentBody、下载）零 try/except。网络超时/5xx/认证失败直接穿透。外部已有 `_cutout_with_fallback` 重试+兜底和下载 5xx 重试，但 OSS 上传、预签名、SegmentBody API 自身无超时配置。
2. **config 读写无保护** — `load_config()/save_config()` 无异常处理，JSON 损坏或磁盘满直接崩溃。
3. **`process_capture_task` 异常过宽** — `except Exception` 不区分可重试错误和致命错误。
4. **图片 I/O 无保护** — `Image.open()/save()` 在合成路径中无 try/except。
5. **前端无熔断** — 摄像头轮询 250ms/次，失败无退避。
6. **无全局 FastAPI 异常处理器** — 未预期异常直接暴 500 + 堆栈。
7. **激光触发循环** — `except Exception` 写错误文本后不尝试恢复，守护线程静默停止。

## Git

远程仓库：`https://github.com/frostdogstarscream/huaita_text.git`

活跃分支：
- `codex/yolo-seg-modnet-eval` — YOLO-seg + MODNet/MatAnyone/MaGGIe/ViTMatte 离线评估 + 跟踪式连拍 matting（当前分支）
- `feature/test-aliyun-seedream-cutout` — 接入 Seedream 绿幕预处理

## 依赖注意事项

- **运行时**：`requirements.txt`（FastAPI + OpenCV + Pillow + 阿里云 SDK + PySide6 + pyserial）
- **打包构建**：`requirements-build.txt`（运行时依赖 + PyInstaller + 完整的阿里云 SDK 子包）
- **离线评估额外依赖**：`requirements-rmbg-eval.txt`（RMBG-2.0、transformers、torch 等——不影响生产环境）
- 模型权重（`models/*.pt`、`models/*.ckpt`）不提交 git（`.gitignore` 已排除）
- `generated/` 目录不提交 git，生产环境运行时自动创建
