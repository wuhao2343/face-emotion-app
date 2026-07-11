# 测试模型效果
from __future__ import annotations

import sys
from pathlib import Path

import cv2

from app.models.pipeline import Pipeline

# 测试图片路径
IMAGE_PATH = "../images/test.jpg"

if __name__ == "__main__":
    image_path = Path(IMAGE_PATH)

    # 加载 Pipeline
    pipe = Pipeline()
    pipe.ensure_loaded()

    if pipe.detector.active_provider and pipe.classifier.active_provider:
        print("模型加载成功")
    else:
        print("模型加载失败")
        sys.exit(1)

    # 读取图片
    frame_bgr = cv2.imread(str(image_path))
    if frame_bgr is None:
        print(f"无法读取图片: {image_path}")
        sys.exit(1)

    h, w = frame_bgr.shape[:2]
    print(f"[图片] 尺寸: {w}x{h}")

    # 检测
    annotated, results = pipe.detect_and_draw(frame_bgr)

    # 打印结果
    print(f"\n检测到 {len(results)} 张人脸:\n")
    for i, r in enumerate(results):
        print(f"  人脸 [{i}] {r.emotion}: {r.score * 100:.1f}%  "
              f"位置=({r.bbox[0]},{r.bbox[1]},{r.bbox[2]},{r.bbox[3]})")
        print(f"         情绪分布: " +
              ", ".join(f"{emo}={s * 100:.1f}%"
                        for emo, s in sorted(r.all_scores.items(), key=lambda x: -x[1])))

    # 保存标注结果
    output_path = image_path.with_stem(f"{image_path.stem}_annotated")
    cv2.imwrite(str(output_path), annotated)
    print(f"\n标注结果已保存到: {output_path}")
