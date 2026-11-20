"""Mini NPU 단위 테스트 (표준 라이브러리 unittest).  실행: python -m unittest discover -s tests -v"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import evaluate_patterns, load_filters, load_json  # noqa: E402
from npu_core import CROSS, UNDECIDED, X, Matrix, judge, mac, mac_flat, normalize_label  # noqa: E402
from patterns import make_cross, make_x  # noqa: E402

CROSS_3 = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
X_3 = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]


class MatrixTest(unittest.TestCase):
    def test_get_set(self):
        m = Matrix(3)
        m.set(1, 2, 7)
        self.assertEqual(m.get(1, 2), 7.0)

    def test_from_rows_rejects_bad_shape_and_values(self):
        for rows in ([[1, 2], [3]], [[1, "a"], [3, 4]], [[True, 0], [0, 0]], []):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                Matrix.from_rows(rows)

    def test_large_sizes(self):
        for n in (3, 5, 13, 25):
            self.assertEqual(Matrix(n).size, n)


class MacTest(unittest.TestCase):
    def test_mission_example(self):
        cross, x = Matrix.from_rows(CROSS_3), Matrix.from_rows(X_3)
        self.assertEqual(mac(cross, cross), 5.0)
        self.assertEqual(mac(cross, x), 1.0)

    def test_size_mismatch(self):
        with self.assertRaises(ValueError):
            mac(Matrix(3), Matrix(5))

    def test_flat_equals_2d(self):
        for n in (3, 5, 13, 25):
            p, f = make_cross(n), make_x(n)
            self.assertAlmostEqual(mac(p, f), mac_flat(p.flatten(), f.flatten()))


class JudgeTest(unittest.TestCase):
    def test_epsilon_tie(self):
        # 0.3 * 3 = 0.8999999999999999 (부동소수점 오차) → 동점 처리
        self.assertNotEqual(0.3 + 0.3 + 0.3, 0.9)
        self.assertEqual(judge(0.9, 0.3 + 0.3 + 0.3), UNDECIDED)

    def test_clear_winner(self):
        self.assertEqual(judge(5.0, 1.0, CROSS, X), CROSS)
        self.assertEqual(judge(1.0, 5.0, CROSS, X), X)


class LabelTest(unittest.TestCase):
    def test_normalize(self):
        for raw, expected in (("+", CROSS), ("cross", CROSS), ("Cross", CROSS), ("x", X), (" X ", X)):
            self.assertEqual(normalize_label(raw), expected)

    def test_unknown_label(self):
        for raw in ("o", "", 1):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                normalize_label(raw)


class DatasetTest(unittest.TestCase):
    def evaluate(self, data):
        filters, _ = load_filters(data)
        return {r.key: r for r in evaluate_patterns(data, filters)}

    def test_real_data_json(self):
        data, error = load_json(Path(__file__).resolve().parent.parent / "data.json")
        self.assertIsNone(error)
        results = self.evaluate(data)
        self.assertEqual(len(results), 6)
        self.assertTrue(results["size_5_2"].passed)
        self.assertEqual(results["size_5_1"].decision, UNDECIDED)

    def test_broken_cases_fail_without_crash(self):
        good = make_cross(5).rows
        data = {
            "filters": {"size_5": {"cross": good, "x": make_x(5).rows}},
            "patterns": {
                "size_5_1": {"input": good, "expected": "+"},          # 정상 → PASS
                "size_5_2": {"input": make_cross(3).rows, "expected": "+"},  # 크기 불일치
                "size_7_1": {"input": make_cross(7).rows, "expected": "x"},  # 필터 없음
                "size_5_3": {"input": good, "expected": "circle"},     # 라벨 오류
                "bad_key": {"input": good, "expected": "+"},           # 키 형식 오류
                "size_5_4": {"input": good},                           # expected 누락
            },
        }
        results = self.evaluate(data)
        self.assertTrue(results["size_5_1"].passed)
        for key in ("size_5_2", "size_7_1", "size_5_3", "bad_key", "size_5_4"):
            with self.subTest(key=key):
                self.assertFalse(results[key].passed)
                self.assertTrue(results[key].reason)

    def test_missing_or_broken_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.json"
            self.assertIsNotNone(load_json(path)[1])
            path.write_text("{broken", encoding="utf-8")
            self.assertIsNotNone(load_json(path)[1])
            path.write_text(json.dumps({"filters": {}, "patterns": {}}), encoding="utf-8")
            self.assertIsNone(load_json(path)[1])


if __name__ == "__main__":
    unittest.main()
