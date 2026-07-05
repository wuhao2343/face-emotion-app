# Face Emotion App

> **本地人脸情绪识别** · YOLOv8-face 检测 + ViT 7 类情绪分类 · 暗色玻璃拟态 UI · 完整本地 API

---



## ✨ 功能

- 图片识别:拖入或上传图片,返回带人脸框 + 情绪标签的图,并统计情绪分布
- 视频处理:上传视频,后台逐帧推理,完成后下载带标注的 MP4
- 摄像头实时:选择摄像头,浏览器看带标注的实时画面 (MJPEG 流)
- 本地 API:完整的 REST API + Swagger 文档,供其他程序调用
- 暗色玻璃拟态 UI:深色主题、毛玻璃 + 渐变 + 动画
- 纯 ONNX 推理:无 torch / 无 ultralytics / 无 transformers

---

## 🏗️ 架构

```
face-emotion-app/
├── app/                  # FastAPI 后端
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── models/
│   │   ├── detector.py   # YOLOv8-face ONNX (akanametov/yolo-face)
│   │   ├── classifier.py # ViT ONNX (trpakov/vit-face-expression)
│   │   └── pipeline.py
│   ├── services/
│   │   ├── video_processor.py
│   │   └── camera_service.py
│   └── routers/          # health / image / detect / video / camera
├── static/               # 暗色玻璃拟态前端 (无 emoji,纯 SVG 图标)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── models/               # yolov8n-face.onnx + trpakov-vit-face.onnx
├── outputs/              # 生成的 MP4
├── .venv/                # 项目本地虚拟环境 (自动创建)
├── requirements.txt
├── check_env.py
├── run.py                # 一键启动
└── README.md
```

---

## 🚀 快速开始

### 1. 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.10 ~ 3.13 |
| 内存 | ≥ 8 GB |
| 系统 | Windows / macOS / Linux |

### 2. 一键启动

```bash
cd face-emotion-app
python run.py
```

`run.py` 会自动:
1. 找到系统里的 Python 3.10+ (优先 3.12)
2. 在 `.venv/` 下创建虚拟环境
3. 把 `requirements.txt` 装进 venv
4. 下载两个 ONNX 模型到 `models/`
5. 启动 uvicorn + 自动打开浏览器

如果想用 GPU 加速(推荐):

```bash
.venv\Scripts\pip install onnxruntime-directml
```

然后启动,默认会优先使用 `DmlExecutionProvider`。

### 3. 手动启动(可选)

```bash
# 创建 venv
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 装依赖
pip install -r requirements.txt

# 跑环境检查
python check_env.py

# 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

打开 <http://localhost:8000>。

---

## 🤖 模型与下载

| 模型 | 用途 | 大小 | 来源 |
|------|------|------|------|
| `yolov8n-face.onnx` | 人脸检测 | ~12 MB | [akanametov/yolo-face v1.0.0](https://github.com/akanametov/yolo-face/releases/tag/1.0.0) |
| `trpakov-vit-face.onnx` | 7 类情绪分类 | ~327 MB | [trpakov/vit-face-expression onnx/](https://huggingface.co/trpakov/vit-face-expression/tree/main/onnx) |

模型在首次运行时自动下载。**自带断点续传** — 如果公司网络 SSL 频繁断,会自动从断点续下,最多 8 次重试。

**手动下载**(如网络不通):

```bash
# YOLO
curl -L -o models/yolov8n-face.onnx \
  https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.onnx

# ViT
curl -L -o models/trpakov-vit-face.onnx \
  https://huggingface.co/trpakov/vit-face-expression/resolve/main/onnx/model.onnx
```

---

## 🔌 API 参考

Swagger 文档: <http://localhost:8000/docs>

### GET `/api/health`

```json
{
  "status": "ok",
  "version": "0.1.0",
  "device": "DmlExecutionProvider",
  "cuda_available": false,
  "gpu_name": "DirectML adapter (Windows GPU)",
  "models_loaded": {"detector": true, "classifier": true}
}
```

### POST `/api/detect/image`

上传图片,返回带框 + 情绪标签的 JPEG。

```bash
curl -F "file=@face.jpg" http://localhost:8000/api/detect/image -o result.jpg
```

### POST `/api/detect/frame` ⭐ **程序调用首选**

```bash
curl -F "file=@face.jpg" http://localhost:8000/api/detect/frame | jq
```

```json
{
  "count": 1,
  "inference_ms": 35.2,
  "device": "DmlExecutionProvider",
  "detections": [
    {
      "bbox": {"x1": 120, "y1": 80, "x2": 280, "y2": 320},
      "emotion": "happy",
      "score": 0.92,
      "all_scores": {"angry": 0.01, "disgust": 0.0, "fear": 0.01,
                     "happy": 0.92, "sad": 0.02, "surprise": 0.03, "neutral": 0.01}
    }
  ]
}
```

### POST `/api/detect/video`

```bash
curl -F "file=@clip.mp4" http://localhost:8000/api/detect/video
# {"task_id":"abc123","status":"queued"}
curl http://localhost:8000/api/detect/video/abc123/status
curl -o out.mp4 http://localhost:8000/api/detect/video/abc123/download
```

### GET `/api/camera/list` / `/api/camera/{id}/stream`

```html
<img src="http://localhost:8000/api/camera/0/stream" />
```

---

## ⚙️ 配置 (`.env` 或环境变量)

```bash
# ORT 执行 provider,按顺序尝试
ORT_PROVIDERS=DmlExecutionProvider,CPUExecutionProvider
# macOS 上没有 DML,会自动落到 CPU;Linux 可用 CUDAExecutionProvider

# 限制
# 4 GB 容纳常见 demo 视频;视频接口流式写盘,内存恒定 ≈1MiB;图片接口
# 仍需把整张图读进内存喂 cv2,故实际部署时根据机器内存酌情下调。
MAX_UPLOAD_MB=4096
MAX_VIDEO_SIDE=1280
CAMERA_PROBE_RANGE=4
```

---

## 🐛 故障排查

| 现象                                 | 解决 |
|------------------------------------|------|
| 启动报 `onnxruntime not found`        | 确认激活了 `.venv`,或直接 `python run.py` 让它自建 |
| 启动报 `Failed to download ...`       | 手动下载两个 ONNX 到 `models/`,见上文 |
| 大文件下载 SSL 中断                       | 已自动断点续传 8 次;若仍失败,看终端日志 |
| 摄像头黑屏                              | Windows 设置 → 隐私 → 摄像头 → 允许桌面应用访问 |
| GPU 加速                             | `pip install onnxruntime-directml`,然后启动 |

---

## 🧪 Python 调用示例

```python
import requests

with open("face.jpg", "rb") as f:
    r = requests.post(
        "http://localhost:8000/api/detect/frame",
        files={"file": ("face.jpg", f, "image/jpeg")},
        timeout=30,
    )
data = r.json()
for det in data["detections"]:
    print(f"{det['emotion']} ({det['score']:.0%}) @ {det['bbox']}")
```

---
