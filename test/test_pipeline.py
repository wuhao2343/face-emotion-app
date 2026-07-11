"""Pipeline 模型效果测试"""
from __future__ import annotations

import gc
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.models.pipeline import DetectionResult, Pipeline

# 已知的 7 类情绪标签
EMOTION_LABELS = {"angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"}

# 测试图片路径（相对于本文件）
TEST_IMAGE_REL = "../images/test.jpg"


@unittest.skipUnless(
    Path(__file__).resolve().parent.parent.joinpath("models").exists(),
    "模型文件不存在，跳过测试",
)
class TestPipeline(unittest.TestCase):
    """Pipeline 集成测试"""

    _pipe: Pipeline | None = None

    @classmethod
    def setUpClass(cls) -> None:
        """加载 Pipeline（所有测试共享一个实例）"""
        cls._pipe = Pipeline()
        cls._pipe.ensure_loaded()

    @classmethod
    def tearDownClass(cls) -> None:
        """释放资源"""
        cls._pipe = None
        gc.collect()

    # ---- 辅助方法 ----

    @staticmethod
    def _load_test_image() -> np.ndarray:
        image_path = Path(__file__).resolve().parent / TEST_IMAGE_REL
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        return frame

    # ---- 测试用例 ----

    def test_01_pipeline_loaded(self) -> None:
        """Pipeline 模型加载成功"""
        self.assertIsNotNone(self._pipe)
        self.assertTrue(
            self._pipe.detector.active_provider,
            "人脸检测器未能成功初始化执行提供者",
        )
        self.assertTrue(
            self._pipe.classifier.active_provider,
            "情绪分类器未能成功初始化执行提供者",
        )

    def test_02_detect_and_draw_returns_correct_types(self) -> None:
        """detect_and_draw 返回类型正确"""
        frame = self._load_test_image()
        annotated, results = self._pipe.detect_and_draw(frame)

        self.assertIsInstance(annotated, np.ndarray, "annotated 应为 ndarray")
        self.assertIsInstance(results, list, "results 应为 list")

    def test_03_annotated_frame_same_dimensions(self) -> None:
        """标注后的帧尺寸与原图一致"""
        frame = self._load_test_image()
        annotated, _ = self._pipe.detect_and_draw(frame)

        self.assertEqual(
            annotated.shape, frame.shape,
            f"标注帧尺寸 {annotated.shape} 应与原图 {frame.shape} 一致",
        )

    def test_04_results_are_detection_result_instances(self) -> None:
        """结果列表中的每个元素都是 DetectionResult 实例"""
        frame = self._load_test_image()
        _, results = self._pipe.detect_and_draw(frame)

        for r in results:
            self.assertIsInstance(
                r, DetectionResult, f"结果应为 DetectionResult，实际为 {type(r)}"
            )

    def test_05_detection_result_bbox_valid(self) -> None:
        """DetectionResult 的 bbox 坐标格式正确"""
        frame = self._load_test_image()
        h, w = frame.shape[:2]
        _, results = self._pipe.detect_and_draw(frame)

        for r in results:
            self.assertEqual(len(r.bbox), 4, "bbox 应为 4 元组")
            x1, y1, x2, y2 = r.bbox
            self.assertTrue(0 <= x1 < x2 <= w, f"bbox x 坐标越界: {r.bbox}")
            self.assertTrue(0 <= y1 < y2 <= h, f"bbox y 坐标越界: {r.bbox}")

    def test_06_detection_result_score_in_range(self) -> None:
        """DetectionResult 的 score 在 [0, 1] 范围内"""
        frame = self._load_test_image()
        _, results = self._pipe.detect_and_draw(frame)

        for r in results:
            self.assertGreaterEqual(r.score, 0.0, f"score {r.score} 应 >= 0")
            self.assertLessEqual(r.score, 1.0, f"score {r.score} 应 <= 1")

    def test_07_detection_result_emotion_valid(self) -> None:
        """DetectionResult 的 emotion 为已知的 7 种标签之一"""
        frame = self._load_test_image()
        _, results = self._pipe.detect_and_draw(frame)

        for r in results:
            self.assertIn(
                r.emotion, EMOTION_LABELS,
                f"情绪标签 '{r.emotion}' 不在已知标签 {EMOTION_LABELS} 中",
            )

    def test_08_all_scores_keys_match_emotion_labels(self) -> None:
        """DetectionResult.all_scores 的键与 7 类情绪一致"""
        frame = self._load_test_image()
        _, results = self._pipe.detect_and_draw(frame)

        for r in results:
            self.assertEqual(
                set(r.all_scores.keys()), EMOTION_LABELS,
                "all_scores 的键应与 7 类情绪标签一致",
            )

    def test_09_all_scores_sum_approx_one(self) -> None:
        """DetectionResult.all_scores 概率之和约为 1（softmax 输出）"""
        frame = self._load_test_image()
        _, results = self._pipe.detect_and_draw(frame)

        for r in results:
            total = sum(r.all_scores.values())
            self.assertAlmostEqual(
                total, 1.0, delta=1e-5,
                msg=f"all_scores 概率之和 {total} 应约为 1.0",
            )

    def test_10_top_emotion_matches_max_score(self) -> None:
        """r.emotion 与 all_scores 中最大值对应"""
        frame = self._load_test_image()
        _, results = self._pipe.detect_and_draw(frame)

        for r in results:
            max_emo = max(r.all_scores, key=lambda k: r.all_scores[k])
            self.assertEqual(
                r.emotion, max_emo,
                f"r.emotion='{r.emotion}' 但 all_scores 最高分是 '{max_emo}'={r.all_scores[max_emo]:.4f}",
            )
            self.assertAlmostEqual(
                r.score, r.all_scores[max_emo], delta=1e-5,
                msg=f"r.score={r.score} 与最高分 {r.all_scores[max_emo]} 不一致",
            )

    def test_11_detect_only_json_returns_tuple(self) -> None:
        """detect_only_json 返回 (results, elapsed_ms)"""
        frame = self._load_test_image()
        results, elapsed_ms = self._pipe.detect_only_json(frame)

        self.assertIsInstance(results, list)
        self.assertIsInstance(elapsed_ms, float)
        self.assertGreaterEqual(elapsed_ms, 0, "推理耗时应 >= 0")

    def test_12_empty_image(self) -> None:
        """传入全零图像不应崩溃"""
        empty = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, results = self._pipe.detect_and_draw(empty)

        self.assertIsInstance(annotated, np.ndarray)
        self.assertIsInstance(results, list)
        # 全零图像可能检测不到人脸，这是可接受的

    def test_13_small_single_pixel_image(self) -> None:
        """传入极小尺寸图像不应崩溃"""
        tiny = np.zeros((1, 1, 3), dtype=np.uint8)
        annotated, results = self._pipe.detect_and_draw(tiny)

        self.assertIsInstance(annotated, np.ndarray)
        self.assertIsInstance(results, list)

    def test_14_annotated_output_save(self) -> None:
        """标注结果可以正常保存为图片文件"""
        frame = self._load_test_image()
        annotated, _ = self._pipe.detect_and_draw(frame)

        output_path = Path(__file__).resolve().parent / "_test_output.jpg"
        try:
            success = cv2.imwrite(str(output_path), annotated)
            self.assertTrue(success, "cv2.imwrite 保存失败")
            self.assertTrue(output_path.exists(), "输出文件不存在")
            # 验证保存的图片可读
            reloaded = cv2.imread(str(output_path))
            self.assertIsNotNone(reloaded, "保存的图片无法重新读取")
            self.assertEqual(reloaded.shape, annotated.shape)
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_15_skip_every_n_pipeline(self) -> None:
        """skip_every_n > 1 的 Pipeline 不应崩溃"""
        pipe_skip = Pipeline(skip_every_n=3)
        pipe_skip.ensure_loaded()

        frame = self._load_test_image()
        # 模拟多帧处理
        for _ in range(5):
            annotated, results = pipe_skip.detect_and_draw(frame)
            self.assertIsInstance(annotated, np.ndarray)
            self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
