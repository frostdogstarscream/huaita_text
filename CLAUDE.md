# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在当前仓库中工作时提供指引。

## 项目概述

淮海战役纪念馆互动拍照一体机。基于 FastAPI 的 Web 服务，驱动 摄像头 → 人像分割 → 背景合成 → 标语叠加 流水线，输出 1080×1920 竖版留念海报。一体机循环展示一组红色宣传风格标语和四张可配置背景。选配 SDBM-60 激光测距传感器通过串口（CH340 UART）接入，实现人员进入范围后自动触发拍照。

## 启动应用

```powershell
python run_app.py
# 或双击 run.bat
```

访问地址 `http://127.0.0.1:10051/`。控制台页面为 `/`，拍照/一体机页面为 `/camera.html`。

**激光诊断**（独立运行，不需要主服务）：

```powershell
python laser_diagnostics.py --port COM3 --duration 10
python laser_diagnostics.py --port COM3 --max-frames 20
```

## 打包（PyInstaller EXE）

```powershell
.\build_exe.ps1 -PythonExe python
```

输出到 `deploy/huaita_text/`。spec 文件为 `huaita_text.spec`。

## 架构

```
main.py                  # FastAPI 应用，路由、lifespan、服务入口（约 290 行，薄层）
run_app.py               # 启动入口，仅调用 main.run_server()
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
```

### 配置

`config.json` 深度合并到 `config_manager.py` 中的 `DEFAULT_CONFIG` 之上。首次运行无配置文件时，默认配置会被写出。配置文件驱动一切：摄像头选择模式、轮播标语及时序、四张背景及其独立的人物/文字布局覆盖、激光触发参数、文字样式（字号、金色渐变调色板、阴影/高光效果），以及通过 `text_tuning.by_slogan` 实现的逐标语文字微调。

**路径行为**：`runtime_paths.py` 解析所有目录。开发模式下，所有路径相对于仓库根目录。冻结/PyInstaller 模式下，`base_dir` 和 `resource_dir` 均指向 exe 所在目录，因此 `config.json`、`html-page/`、`fonts/`、`generated/` 均位于 exe 旁边。

### 图片处理流水线

1. `capture_manager.start_capture_task(source)` —— 获取互斥锁，启动守护线程
2. `capture_manager.process_capture_task(task_id)` —— 摄像头帧 → `image_composer.build_subject_cutout`（上传 OSS → 阿里云 SegmentBody → 下载 RGBA）→ `image_composer.compose_single_variant`（缩放人物、贴入背景、渲染标语文字）→ 保存 1080×1920 JPEG
3. 激光触发拍照：连拍模式（`burst_count` 张，间隔 `burst_interval_seconds` 秒）
4. 手动（按钮）拍照：单帧，同样使用轮播背景

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
- `select.html`、`view.html`、`download.html` —— 背景选择和结果浏览

### 阿里云流水线（`ali/`）

`AliOssSegmentPipeline` 分片上传至 OSS → 获取预签名 URL → 调用 `SegmentBody` API → 下载 RGBA 结果。密钥从环境变量 `ALI_ACCESS_KEY_ID` / `ALI_ACCESS_KEY_SECRET` 读取，详见 `ali/ali_oss_multipart_upload_util.py:25-26`。`config.json` 配置 bucket/region/endpoint。

## 重要说明

- 运行测试：`python -m pytest tests/ -v`（当前 38 个测试）
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

每条标语在 `text_tuning.by_slogan` 中有一个 `max_font_size` 字段（整数，px），作为该标语在 994px 文本框内的字号绝对值上限。值由 `calc_max_font.py` 脚本通过二分查找和实际字体测量得出：

- 短标语（≤8 字/行）→ 上限 ≥124，不影响当前渲染（系统 font_size_max=124 已是更紧约束）
- 长标语（9–10 字/行）→ 上限 90–110，在 row 缩放后生效，防止文字溢出文本框

`text_renderer.py:635-638` 负责应用此上限：在所有 row 缩放之后，若 tuning 中有 `max_font_size`，则 `font_size_max = min(font_size_max, max_font_size)`。

## 批量标语预览

`text_region_preview/batch_render.py` 可批量渲染所有标语到指定背景，用于逐条检查排版效果：

```powershell
python text_region_preview/batch_render.py
```

生成 `text_region_preview/1/` ~ `4/` 四个子目录，每个目录含 75 张该背景下的标语渲染图。

## Git

远程仓库：`https://github.com/frostdogstarscream/huaita_text.git`
