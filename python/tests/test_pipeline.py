import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd


PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from batch_validate import calculate_aggregate
from evaluate_lesion_masks import component_metrics
from inference import (
    predict_batch_with_details, predict_batch_with_model,
    predict_with_model_details,
)
from modules.explainability import assess_gradcam_quality, calibrate_confidence
from modules.quality import assess_image_quality, generate_recapture_feedback
from modules.segmentation import (
    detect_exudates,
    detect_hemorrhages,
    detect_microaneurysms,
    detect_neovascularization,
    localize_optic_disc, refine_subpixel_peak,
)
from utils.metrics import quadratic_weighted_kappa
from utils.calibration import temperature_scale_probabilities, validate_temperature
from utils.safety import (
    TRIAGE_AUTO_CLEAR, TRIAGE_HUMAN_REVIEW, TRIAGE_RECAPTURE, TRIAGE_REFER,
    assign_triage,
)
from utils.quality_calibration import apply_iqs_thresholds, fit_iqs_thresholds
from utils.preprocessing import model_space_fov_mask, preprocess_for_model, preprocess_fundus
from simulate_district_workflow import Scenario, _multi_server, simulate_scenario
from bootstrap_grading_metrics import stratified_bootstrap
from analyze_selective_prediction import curve as selective_prediction_curve


class QualityTests(unittest.TestCase):
    def test_empty_image_has_complete_zero_metrics(self):
        score, metrics = assess_image_quality(np.array([], dtype=np.uint8))
        self.assertEqual(score, 0.0)
        self.assertEqual(metrics['iqs'], 0.0)
        self.assertTrue(generate_recapture_feedback(metrics))

    def test_preprocess_shapes_and_finite_values(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        image[10:70, 20:100] = [120, 80, 60]
        processed, green, fov = preprocess_fundus(image, target_size=128)
        model_input = preprocess_for_model(image, size=128)
        self.assertEqual(processed.shape, (128, 128, 3))
        self.assertEqual(green.shape, (128, 128))
        self.assertEqual(fov.shape, (128, 128))
        self.assertEqual(model_input.shape, (128, 128, 3))
        self.assertTrue(np.isfinite(model_input).all())

    def test_black_model_input_does_not_create_empty_crop(self):
        result = preprocess_for_model(np.zeros((32, 48, 3), dtype=np.uint8), size=64)
        self.assertEqual(result.shape, (64, 64, 3))
        self.assertTrue(np.isfinite(result).all())

    def test_model_space_fov_mask_matches_model_shape(self):
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        cv2.circle(image, (40, 20), 18, (100, 80, 60), -1)
        mask = model_space_fov_mask(image, size=64)
        self.assertEqual(mask.shape, (64, 64))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(mask.any())


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        self.green = np.full((128, 128), 100, dtype=np.uint8)
        self.fov = np.zeros((128, 128), dtype=np.uint8)
        cv2.circle(self.fov, (64, 64), 58, 255, -1)
        self.vessels = np.zeros_like(self.green)
        self.od_center = (92, 64)
        self.od_radius = 10

    def test_uniform_image_has_no_dark_lesions(self):
        ma_mask, ma_count, centroids = detect_microaneurysms(
            self.green, self.vessels, self.fov, self.od_center, self.od_radius)
        hem_mask, hem_count, hem_stats = detect_hemorrhages(
            self.green, self.vessels, ma_mask, self.fov,
            self.od_center, self.od_radius)
        self.assertEqual(ma_count, 0)
        self.assertEqual(centroids, [])
        self.assertFalse(ma_mask.any())
        self.assertEqual(hem_count, 0)
        self.assertEqual(hem_stats, {'dot_count': 0, 'blot_count': 0})
        self.assertFalse(hem_mask.any())

    def test_normal_single_vessel_is_not_neovascularization(self):
        cv2.line(self.vessels, (15, 64), (110, 64), 255, 2)
        nv_mask, detected = detect_neovascularization(
            self.green, self.vessels, self.od_center, self.od_radius, self.fov)
        self.assertFalse(detected)
        self.assertFalse(nv_mask.any())

    def test_partially_clipped_bright_disc_is_localized_near_edge(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.circle(image, (64, 64), 60, (90, 70, 50), -1)
        cv2.circle(image, (122, 64), 12, (250, 220, 180), -1)
        center, radius = localize_optic_disc(image)
        self.assertGreater(center[0], 100)
        self.assertLess(abs(center[1] - 64), 15)
        self.assertGreater(radius, 0)

    def test_uniform_fundus_has_no_exudate_candidates(self):
        image = np.full((128, 128, 3), 100, dtype=np.uint8)
        _, hard_count, soft_count, distance = detect_exudates(
            image, self.green, (64, 64), self.od_center, self.od_radius, self.fov)
        self.assertEqual(hard_count, 0)
        self.assertEqual(soft_count, 0)
        self.assertEqual(distance, -1.0)

    def test_quadratic_microaneurysm_peak_refinement_is_subpixel(self):
        yy, xx = np.mgrid[0:24, 0:24]
        expected_x, expected_y = 10.25, 12.35
        response = np.exp(-((xx-expected_x)**2 + (yy-expected_y)**2)/(2*1.4**2))
        peak_y, peak_x = np.unravel_index(np.argmax(response), response.shape)
        refined_x, refined_y = refine_subpixel_peak(response, peak_x, peak_y)
        self.assertLess(abs(refined_x-expected_x), 0.08)
        self.assertLess(abs(refined_y-expected_y), 0.08)
        self.assertNotEqual(refined_x, float(peak_x))


class MetricTests(unittest.TestCase):
    def test_qwk_perfect_and_constant(self):
        self.assertAlmostEqual(quadratic_weighted_kappa([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]), 1.0)
        self.assertAlmostEqual(quadratic_weighted_kappa([2, 2], [2, 2]), 1.0)

    def test_qwk_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            quadratic_weighted_kappa([0, 1], [0])

    def test_calibration_is_normalized(self):
        calibrated = calibrate_confidence([0.1, 0.2, 0.3, 0.25, 0.15])
        self.assertAlmostEqual(float(calibrated.sum()), 1.0)
        self.assertTrue((calibrated >= 0).all())

    def test_subunit_temperature_sharpens_distribution(self):
        raw = np.array([0.7, 0.2, 0.1])
        calibrated = temperature_scale_probabilities(raw, 0.8)
        self.assertGreater(calibrated[0], raw[0])
        self.assertAlmostEqual(float(calibrated.sum()), 1.0)

    def test_invalid_temperature_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_temperature(0)

    def test_gradcam_qc_flags_border_concentration(self):
        heatmap = np.zeros((100, 100), dtype=float)
        heatmap[:10] = 1
        fov = np.ones((100, 100), dtype=bool)
        metrics = assess_gradcam_quality(heatmap, fov)
        self.assertTrue(metrics['shortcut_flag'])
        self.assertGreater(metrics['border_enrichment'], 1.5)

    def test_lesion_component_matching_counts_unmatched_candidates(self):
        truth = np.zeros((32, 32), dtype=bool)
        prediction = np.zeros_like(truth)
        truth[5:10, 5:10] = True
        prediction[5:10, 5:10] = True
        prediction[20:24, 20:24] = True
        metrics = component_metrics(prediction, truth)
        self.assertEqual(metrics['lesion_tp'], 1)
        self.assertEqual(metrics['lesion_fp'], 1)
        self.assertEqual(metrics['lesion_fn'], 0)

    def test_aggregate_perfect_predictions(self):
        records = []
        for grade in range(5):
            records.append({
                'status': 'success',
                'true_grade': grade,
                'predicted_grade': grade,
                'referable_probability': 0.9 if grade >= 2 else 0.1,
            })
        aggregate, confusion, per_grade = calculate_aggregate(records)
        self.assertEqual(aggregate['counts']['evaluated'], 5)
        self.assertAlmostEqual(aggregate['metrics']['qwk'], 1.0)
        self.assertAlmostEqual(aggregate['metrics']['referable_auc'], 1.0)
        self.assertTrue(np.array_equal(confusion, np.eye(5, dtype=int)))
        self.assertEqual(len(per_grade), 5)

    def test_empty_batch_prediction_does_not_load_model(self):
        grades, probabilities, scores = predict_batch_with_model([])
        self.assertEqual(grades.size, 0)
        self.assertEqual(probabilities.shape, (0, 5))
        self.assertEqual(scores.size, 0)

    def test_empty_detailed_batch_has_dual_head_schema(self):
        details = predict_batch_with_details([])
        self.assertEqual(details['grade_probabilities'].shape, (0, 5))
        self.assertEqual(details['referable_head_probabilities'].size, 0)
        self.assertEqual(details['triage_actions'].size, 0)

    def test_single_detailed_prediction_preserves_string_triage(self):
        batched = {
            'grades': np.array([2], dtype=int),
            'grade_probabilities': np.array([[0.0, 0.0, 1.0, 0.0, 0.0]]),
            'head_disagreements': np.array([True], dtype=bool),
            'triage_actions': np.array([TRIAGE_HUMAN_REVIEW], dtype=object),
        }
        with patch('inference.predict_batch_with_details', return_value=batched):
            details = predict_with_model_details(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertEqual(details['grades'], 2)
        self.assertTrue(details['head_disagreements'])
        self.assertEqual(details['triage_actions'], TRIAGE_HUMAN_REVIEW)
        self.assertEqual(details['grade_probabilities'].shape, (5,))

    def test_four_way_triage_policy(self):
        self.assertEqual(assign_triage(
            quality_rejected=True, grade_referable=False,
            referable_head_positive=False), TRIAGE_RECAPTURE)
        self.assertEqual(assign_triage(
            quality_rejected=False, grade_referable=False,
            referable_head_positive=False), TRIAGE_AUTO_CLEAR)
        self.assertEqual(assign_triage(
            quality_rejected=False, grade_referable=True,
            referable_head_positive=True), TRIAGE_REFER)
        self.assertEqual(assign_triage(
            quality_rejected=False, grade_referable=True,
            referable_head_positive=False), TRIAGE_HUMAN_REVIEW)

    def test_eyeq_ordered_threshold_application(self):
        predictions = apply_iqs_thresholds(
            np.array([0.1, 0.4, 0.8]), reject_threshold=0.25,
            good_threshold=0.65)
        self.assertTrue(np.array_equal(predictions, [2, 1, 0]))

    def test_eyeq_threshold_fit_on_separable_scores(self):
        scores = np.array([0.05, 0.10, 0.15, 0.40, 0.45, 0.50, 0.80, 0.85, 0.90])
        labels = np.array([2, 2, 2, 1, 1, 1, 0, 0, 0])
        reject, good, evidence = fit_iqs_thresholds(scores, labels, seed=7)
        self.assertGreater(good, reject)
        predictions = apply_iqs_thresholds(scores, reject, good)
        self.assertTrue(np.array_equal(predictions, labels))
        self.assertAlmostEqual(evidence['training_macro_f1'], 1.0)

    def test_parallel_server_queue_uses_both_clinicians(self):
        starts, finishes = _multi_server(np.array([0.0, 0.0, 0.0]), 30.0, 2)
        self.assertTrue(np.array_equal(starts, [0.0, 0.0, 30.0]))
        self.assertTrue(np.array_equal(finishes, [30.0, 30.0, 60.0]))

    def test_district_simulation_conserves_arrivals(self):
        result = simulate_scenario(
            Scenario(
                bandwidth_mbps=5.0,
                clinicians=1,
                quality_reject_rate=0.10,
                total_patients=1000,
            ),
            seed=7,
        )
        self.assertEqual(
            result['completed'] + result['unfinished_at_stop'],
            result['expected_arrivals'],
        )
        self.assertGreaterEqual(result['required_review_clinicians_at_80pct_utilization'], 1)
        self.assertGreaterEqual(result['p95_review_wait_seconds'], result['p90_review_wait_seconds'])

    def test_bootstrap_metrics_perfect_predictions(self):
        frame = pd.DataFrame({
            'status': ['success'] * 10,
            'true_grade': [0, 1, 2, 3, 4] * 2,
            'predicted_grade': [0, 1, 2, 3, 4] * 2,
            'referable_probability': [0.01, 0.1, 0.8, 0.9, 0.99] * 2,
        })
        evidence = stratified_bootstrap(frame, replicates=100, seed=1)
        self.assertEqual(evidence['images'], 10)
        self.assertEqual(evidence['metrics']['qwk']['point_estimate'], 1.0)
        self.assertEqual(
            evidence['metrics']['referable_sensitivity']['ci95_low'], 1.0)

    def test_selective_prediction_routes_disagreement_and_low_confidence(self):
        frame = pd.DataFrame({
            'status': ['success'] * 3,
            'true_grade': [0, 2, 2],
            'predicted_grade': [0, 2, 1],
            'confidence': [0.95, 0.80, 0.40],
            'head_disagreement': [False, True, False],
        })
        result = selective_prediction_curve(frame, np.array([0.5])).iloc[0]
        self.assertEqual(result.auto_decision_images, 1)
        self.assertEqual(result.human_review_images, 2)
        self.assertEqual(result.disagreement_reviews, 1)
        self.assertEqual(result.low_confidence_reviews, 1)


if __name__ == '__main__':
    unittest.main()
