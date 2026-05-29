# 4090 远程 Matting 服务部署指南

## 架构概览

一体机（客户端）通过 HTTP 调用 4090 机器上的 FastAPI 服务：

```
一体机 (capture_manager)
  → POST /api/remote/jobs         (上传 zip：burst.avi + 16帧jpg)
  → GET  /api/remote/jobs/{id}    (轮询状态)
  → GET  /api/remote/jobs/{id}/final/{idx}  (下载4张合成图)

4090 服务 (remote_gpu_matting_service.py)
  → TrackedMattingService.segment_sequence()
    → YOLO11x-seg 实例分割 + 跨帧跟踪
    → MatAnyone2 视频 matting
    → alpha 约束 + 边缘净化
  → compose_single_variant() 合成最终海报
```

## 1. 环境准备

```powershell
# 创建 conda 环境（推荐 Python 3.10）
conda create -n huaita-gpu python=3.10 -y
conda activate huaita-gpu

# PyTorch + CUDA（4090 需要 CUDA 12.x）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# MatAnyone2（从 GitHub 安装）
pip install -e MatAnyone2/

# 项目依赖（不需要 PySide6 和 pyserial，但 requirements.txt 里有）
pip install fastapi uvicorn opencv-python numpy Pillow requests pyyaml
pip install ultralytics
```

## 2. 模型文件

需要放到 4090 机器的 `models/` 目录下：

| 文件 | 大小 | 来源 |
|---|---|---|
| `models/yolo11x-seg.pt` | ~125 MB | Ultralytics 自动下载或从一体机拷贝 |
| MatAnyone2 权重 | ~数百 MB | 首次运行自动从 HuggingFace (`PeiqingYang/MatAnyone2`) 下载 |

## 3. 需要拷贝的源码文件

从仓库拷贝以下文件到 4090 机器（保持目录结构）：

### 服务核心（必须）

- `run_remote_service.py` — 入口脚本
- `remote_gpu_matting_service.py` — FastAPI 服务端
- `tracked_matting_service.py` — 跟踪式 matting 核心
- `matanyone_service.py` — MatAnyone 封装
- `subject_instance_segmentation.py` — YOLO 实例分割
- `subject_instance_tracking.py` — 跨帧跟踪
- `subject_edge_refine.py` — 边缘净化
- `image_composer.py` — 最终合成
- `app_state.py` — 全局状态
- `config_manager.py` — 配置加载
- `runtime_paths.py` — 路径解析
- `text_renderer.py` — 标语渲染
- `slogan_manager.py` — 标语管理
- `background_manager.py` — 背景管理
- `run_tracked_video_matting_eval.py` — 含 `select_output_frames` 函数
- `config.json` — 配置文件

### 资源目录（必须）

- `fonts/` — 中文字体（`default.ttf` 等）
- `html-page/assets/` — 背景图片（`photos/1.jpg` ~ `4.jpg`）
- `models/` — YOLO 权重

## 4. config.json 配置

4090 机器上的 `config.json` 需要确保以下节正确：

```json
{
  "remote_matting": {
    "host": "0.0.0.0",
    "port": 18080
  },
  "tracked_matting": {
    "enabled": true,
    "input_frame_count": 16,
    "output_frame_indices": [3, 7, 10, 13],
    "subject_priority_enabled": true
  }
}
```

## 5. 启动服务

```powershell
python run_remote_service.py
```

服务监听 `0.0.0.0:18080`。可以用 `curl` 测试：

```powershell
# 健康检查
curl http://localhost:18080/api/health
# 期望返回: {"ok": true, "service": "remote_tracked_matanyone", "gpu_ready": true}
```

## 6. 一体机端配置

一体机的 `config.json` 中将远程服务地址指向 4090 机器 IP：

```json
{
  "remote_matting": {
    "enabled": true,
    "base_url": "http://192.168.x.x:18080"
  }
}
```

## 7. 当前缺失（可后续补充）

- **没有 Dockerfile / docker-compose** — 目前是裸跑 Python
- **没有 systemd 服务文件** — 开机自启动需要手动配置
- **没有 GPU 显存监控** — 健康检查返回 `gpu_ready: True` 是硬编码的
- **任务是内存存储** — 服务重启后任务状态丢失
