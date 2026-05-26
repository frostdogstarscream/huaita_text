# ali — 阿里云 OSS 上传 + 人像分割工具包

将本地图片上传至阿里云 OSS，调用 SegmentBody 接口进行人像分割，合成绿幕背景。

## 环境

- Python 3.8+
- 阿里云 AccessKey（已内置在源码中）

## 安装

```bash
pip install alibabacloud_oss_v2 alibabacloud_imageseg20191230 alibabacloud_credentials opencv-python Pillow numpy
```

## 项目结构

```
ali/
├── __init__.py                            # 包入口，导出三个公开类
├── ali_oss_multipart_upload_util.py       # OSS 分片上传（路径 / cv2 / PIL）
├── ali_segment_greenscreen_util.py        # SegmentBody 人像分割 + 绿幕合成
├── ali_oss_segment_pipeline_util.py       # 上传 → 分割 → 保存 一体化流水线
├── ali_person_front.py                    # 原始参考实现（同步+异步）
├── ali_oss_test.py                        # OSS 上传+下载 独立示例
├── example_oss_segment_pipeline.py        # Pipeline 三种输入模式示例
├── test_upload_oss_then_segment_greenscreen.py      # 端到端测试（路径输入）
├── test_upload_oss_then_segment_greenscreen_cv2.py  # 端到端测试（cv2 输入）
├── test_upload_oss_then_segment_greenscreen_pil.py  # 端到端测试（PIL 输入）
├── AccessKey.csv                          # 凭证文件
└── resource/
    └── person_front/                      # 测试图片目录
output/                                    # 分割结果输出目录
```

## 快速开始

### 一行调用（推荐）

```python
import sys
sys.path.insert(0, r'D:\Code\homewor')

from ali import AliOssSegmentPipeline

pipeline = AliOssSegmentPipeline()
result = pipeline.process_and_save("你的图片.jpg")
# 返回 numpy.ndarray (BGR 或 BGRA)，同时结果已保存到 output/
```

### 分步调用

```python
from ali import AliOssMultipartUploader, AliSegmentBodyGreenscreen
import alibabacloud_oss_v2 as oss
from datetime import timedelta

uploader = AliOssMultipartUploader()
segmenter = AliSegmentBodyGreenscreen()

# 1. 上传到 OSS
url = uploader.upload_local_file("图片.jpg", "person_front/my_image.jpg")

# 2. 生成预签名 URL（私有桶需要）
presigned = uploader.client.presign(
    oss.GetObjectRequest(bucket=uploader.bucket, key="person_front/my_image.jpg"),
    expires=timedelta(hours=1),
)

# 3. 人像分割
result_bgr = segmenter.segment_oss_url_to_bgr(presigned.url)

# 4. 保存结果
segmenter.save_bgr(result_bgr, "output/result.png")
```

## API

### AliOssMultipartUploader

OSS 分片上传，支持三种输入格式。

```python
uploader = AliOssMultipartUploader(
    access_key_id="...",      # 可选，有默认值
    access_key_secret="...",  # 可选，有默认值
    bucket="huaita-person-img",
    region="cn-shanghai",
    endpoint="https://oss-cn-shanghai.aliyuncs.com",
    part_size=100 * 1024,     # 分片大小
)

# 上传：支持 文件路径 / cv2 ndarray / PIL Image
url = uploader.upload_local_file(source, "object_key.jpg")
```

| 输入类型 | 说明 |
|---|---|
| `str` / `Path` | 本地文件路径 |
| `numpy.ndarray` | OpenCV 图像（BGR/BGRA/灰度，uint8） |
| `PIL.Image.Image` | Pillow 图像对象 |

### AliSegmentBodyGreenscreen

调用阿里云 SegmentBody 接口，下载分割结果并处理。

```python
segmenter = AliSegmentBodyGreenscreen(
    access_key_id="...",
    access_key_secret="...",
    endpoint="imageseg.cn-shanghai.aliyuncs.com",
)

# 传入公网/预签名 URL，返回 ndarray
bgr = segmenter.segment_oss_url_to_bgr(image_url)

# 异步版本
bgr = await segmenter.segment_oss_url_to_bgr_async(image_url)

# 分割并直接保存到文件
segmenter.segment_oss_url_to_file(image_url, "output/result.png")

# 手动保存 ndarray
path = segmenter.save_bgr(bgr, "output/result.png")
```

**分割结果说明：**
- 返回 4 通道图像（BGRA）：API 返回了透明通道，直接保留
- 返回 3 通道图像（BGR）：将浅色背景（≥245）替换为纯绿 `(0, 255, 0)`

### AliOssSegmentPipeline

串联上传 + 分割 + 保存的完整流水线。

```python
pipeline = AliOssSegmentPipeline(
    access_key_id="...",
    access_key_secret="...",
    bucket="huaita-person-img",
    region="cn-shanghai",
    oss_endpoint="https://oss-cn-shanghai.aliyuncs.com",
    imageseg_endpoint=None,           # 默认 cn-shanghai
    output_dir="output",              # 默认 ../output
    presign_expires=timedelta(hours=1),
)

result = pipeline.process_and_save(
    image,                            # 路径 / ndarray / PIL Image
    oss_object_key="person_front/xxx.jpg",   # 可选
    output_filename="result.png",     # 可选
    upload_verbose=False,             # 是否打印 OSS 上传明细
)
```

## 运行示例脚本

```bash
# Pipeline 三种输入模式
python ali/example_oss_segment_pipeline.py --mode all
python ali/example_oss_segment_pipeline.py --mode path
python ali/example_oss_segment_pipeline.py --mode cv2
python ali/example_oss_segment_pipeline.py --mode pil

# 端到端测试
python ali/test_upload_oss_then_segment_greenscreen.py
python ali/test_upload_oss_then_segment_greenscreen_cv2.py
python ali/test_upload_oss_then_segment_greenscreen_pil.py

# OSS 上传 + 下载
python ali/ali_oss_test.py
```

## 配置

| 项 | 默认值 |
|---|---|
| OSS Bucket | `huaita-person-img` |
| 区域 | `cn-shanghai` |
| OSS Endpoint | `https://oss-cn-shanghai.aliyuncs.com` |
| ImageSeg Endpoint | `imageseg.cn-shanghai.aliyuncs.com` |
| 上传路径前缀 | `person_front/` |
| 输出目录 | `../output/`（相对 ali 包） |
| 分片大小 | 100 KB |

## 注意事项

- 代码中 AccessKey 为明文硬编码，生产环境应改用环境变量或 STS 临时凭证
- Windows 下 `cv2.imread` 不支持中文路径，用 `np.fromfile(path, dtype=np.uint8)` + `cv2.imdecode(raw, cv2.IMREAD_COLOR)` 替代
- SegmentBody 要求图片中的人脸/人体区域不小于 64×64 像素

## 项目集成

### 调用链

```
capture_manager.py          ← 拍照任务调度
  └─ image_composer.py      ← build_subject_cutout()
       └─ ali_segment_service.py  ← AliSegmentService（薄封装层）
            └─ ali/ali_oss_segment_pipeline_util.py  ← AliOssSegmentPipeline
                 ├── ali_oss_multipart_upload_util.py  → OSS 分片上传
                 └── ali_segment_greenscreen_util.py   → SegmentBody API
```

### 串行 → 并发改造（`capture_manager.py` 激光模式）

原始逻辑为单循环内依次执行：拍照 → OSS上传 → SegmentBody → 下载 → 合成，4 张照片完全串行。

改造后激光模式拆为三阶段：

```
阶段 1 — 串行采集（相机独占）
阶段 2 — 并行抠图（ThreadPoolExecutor, max_workers=burst_count）
         4 张照片的 OSS 上传 + SegmentBody 同时发起
阶段 3 — 串行合成（PIL 本地操作）
```

总耗时预期从 4×单次抠图时间 降至接近单次抠图时间。

### 线程安全性

整个 `AliOssSegmentPipeline` 实例可在多线程中安全共享：

| 属性 | 共享方式 | 安全性 |
|---|---|---|
| `self._uploader` | 只读引用，每次调用用独立 OSS key | 安全 |
| `self._segmenter` | 只读引用，每次调用独立请求 | 安全 |
| `self._output_dir` | 不可变 Path | 安全 |
| `oss.Client` | urllib3 连接池（线程安全） | 安全 |
| `ImageSegClient` | Tea DSL HTTP 管道（线程安全） | 安全 |
| `requests.get()` | 模块级函数，连接池线程安全 | 安全 |

每次 `process_and_save()` 调用使用 `uuid.uuid4().hex` 生成唯一的 OSS key 和输出文件名，各线程间无数据交叉。
