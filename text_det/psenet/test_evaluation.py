"""Step 7 tests: python text_detection/psenet/test_evaluation.py"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval_metrics import (counts_to_metrics, evaluate_image, polygon_iou,
                          read_icdar_annotations)
from evaluate import write_rrc_result, render


def rect(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def det(box, score=0.9):
    return {'box_xy': box, 'score': score}


class TestEvaluation(unittest.TestCase):
    def test_polygon_iou(self):
        a = rect(0, 0, 10, 10)
        self.assertAlmostEqual(polygon_iou(a, a), 1.0)
        self.assertAlmostEqual(polygon_iou(a, rect(5, 0, 15, 10)), 1 / 3)
        self.assertEqual(polygon_iou(a, rect(20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(polygon_iou(a, a[::-1]), 1.0)

    def test_perfect_match(self):
        r = evaluate_image([rect(0, 0, 10, 10)], [False], [det(rect(0, 0, 10, 10))])
        self.assertEqual((r['tp'], r['fp'], r['fn']), (1, 0, 0))
        self.assertEqual(r['f1'], 1.0)

    def test_duplicate_predictions_are_false_positives(self):
        g = rect(0, 0, 10, 10)
        r = evaluate_image([g], [False], [det(g), det(g)])
        self.assertEqual((r['tp'], r['fp'], r['fn']), (1, 1, 0))
        self.assertAlmostEqual(r['precision'], 0.5)

    def test_dontcare_removes_predictions_only_when_overlap_large(self):
        ignored = rect(0, 0, 20, 20)
        care = rect(40, 0, 60, 20)
        detections = [det(rect(1, 1, 19, 19)), det(care),
                      det(rect(16, 0, 36, 20))]
        r = evaluate_image([ignored, care], [True, False], detections)
        # first prediction ignored, second TP, third FP (only 20% inside ignore)
        self.assertEqual(r['ignored_det'], [0])
        self.assertEqual((r['tp'], r['fp'], r['fn']), (1, 1, 0))

    def test_strict_iou_threshold(self):
        g = rect(0, 0, 10, 10)
        # prediction covers exactly half the GT: IoU = 0.5
        p = rect(0, 0, 5, 10)
        r = evaluate_image([g], [False], [det(p)])
        self.assertEqual((r['tp'], r['fp'], r['fn']), (0, 1, 1))
        # clear overlap, should match
        r = evaluate_image([g], [False], [det(rect(0, 0, 7, 10))])
        self.assertEqual(r['tp'], 1)

    def test_all_empty_and_false_negative(self):
        self.assertEqual(evaluate_image([], [], [])['tp'], 0)
        r = evaluate_image([rect(0, 0, 10, 10)], [False], [])
        self.assertEqual((r['tp'], r['fp'], r['fn']), (0, 0, 1))
        self.assertEqual(counts_to_metrics(3, 2, 1)['recall'], 0.75)

    def test_parse_gt_and_rrc_export(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'gt_img_1.txt'
            p.write_text('\ufeff0,0,10,0,10,10,0,10,hello,world\n'
                         '20,20,30,20,30,30,20,30,###\n', encoding='utf-8')
            polygons, ignored = read_icdar_annotations(p)
            self.assertEqual(ignored, [False, True])
            self.assertEqual(polygons[0].shape, (4, 2))
            q = Path(d) / 'res_img_1.txt'
            write_rrc_result(q, [det(rect(0, 0, 10, 10))])
            self.assertEqual(q.read_text().strip(), '0,0,10,0,10,10,0,10')

    def test_render_unchanged_shape(self):
        img = np.ones((80, 120, 3), dtype=np.uint8) * 200
        box = rect(5, 5, 30, 30)
        r = evaluate_image([box], [False], [det(box)])
        out = render(img, [box], [False], [det(box)], r)
        self.assertEqual(out.shape, img.shape)
        self.assertFalse(np.array_equal(out, img))


if __name__ == '__main__':
    unittest.main(verbosity=2)
