import unittest

import numpy as np
import pandas as pd
from scipy import fft

from src.continuous import group_events, match_events
from src.course_demos import FRAME_CENTERS, SPEC_FREQUENCIES, make_signal, tones_after_resampling
from src.operating_curve import aggregate_at_threshold, moving_average
from src.preprocessing import SAMPLE_RATE, SPECTROGRAM_SHAPE, denoised_spectrogram, stft_power
from src.template_detector import score_spectrograms


class SignalTests(unittest.TestCase):
    def test_resample_preserves_150_hz_and_rejects_700_hz_alias(self):
        polyphase, naive = tones_after_resampling()
        self.assertEqual(len(polyphase), 3 * SAMPLE_RATE)
        wanted = 2 * np.abs(fft.rfft(polyphase)) / len(polyphase)
        unfiltered = 2 * np.abs(fft.rfft(naive)) / len(naive)
        self.assertAlmostEqual(wanted[150 * 3], 1.0, delta=0.01)
        self.assertLess(wanted[300 * 3], unfiltered[300 * 3] / 1000)

    def test_chirp_direction_and_spectrogram_shape(self):
        spec = denoised_spectrogram(stft_power(make_signal()))[0]
        self.assertEqual(spec.shape, SPECTROGRAM_SHAPE)
        np.testing.assert_allclose(np.median(spec, axis=1), 0, atol=1e-5)
        interval = (FRAME_CENTERS >= 1.2) & (FRAME_CENTERS <= 1.8)
        ridge = SPEC_FREQUENCIES[spec[:, interval].argmax(axis=0)]
        slope = np.polyfit(FRAME_CENTERS[interval], ridge, 1)[0]
        self.assertAlmostEqual(slope, 120, delta=7)

    def test_silence_is_finite_for_both_ablations(self):
        silence = np.zeros(3 * SAMPLE_RATE, dtype=np.float32)
        for bandpass, median in ((True, True), (True, False), (False, True)):
            spec = denoised_spectrogram(stft_power(silence, bandpass=bandpass), remove_median=median)
            self.assertTrue(np.isfinite(spec).all())
            self.assertEqual(spec.shape, (1, *SPECTROGRAM_SHAPE))

    def test_template_prefers_chirp_to_stationary_tone(self):
        specs = denoised_spectrogram(stft_power(np.stack([make_signal(), make_signal(upcall=False)])))
        chirp, stationary = score_spectrograms(specs, 2)
        self.assertGreater(chirp, stationary + 0.15)


class EventTests(unittest.TestCase):
    def setUp(self):
        self.scores = np.array([0.1, 0.95, 0.9, 0.1, 0.71, 0.1])
        self.truth = pd.DataFrame({"timestamp": [2.2, 3.4], "snr": [8.0, 5.0], "uncertain": [False, False]})

    def test_nearby_calls_separated_by_a_negative_window(self):
        events = group_events(self.scores, 0.5)
        self.assertEqual([event["peak_s"] for event in events], [2.0, 3.5])
        self.assertEqual(match_events(events, self.truth), [(2.2, 8.0), (3.4, 5.0)])

    def test_annotation_cannot_match_two_events(self):
        self.truth.loc[0, "timestamp"] = 2.55
        events = group_events(self.scores, 0.5)
        self.assertEqual(match_events(events, self.truth.iloc[:1]), [(2.55, 8.0), (None, None)])

    def test_recording_with_no_calls_counts_false_alarms(self):
        result = aggregate_at_threshold({"only": self.scores}, {"only": self.truth.iloc[:0]}, 0.5, 1, 0.5)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 2, 0))
        self.assertEqual(result["false_alarms_per_hour"], 4)
        self.assertTrue(np.isnan(result["recall"]))

    def test_uncertain_annotation_does_not_improve_recall_or_add_fp(self):
        self.truth.loc[1, "uncertain"] = True
        result = aggregate_at_threshold({"only": self.scores}, {"only": self.truth}, 0.5, 1, 0.5)
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["ignored_uncertain"]), (1, 0, 0, 1))

    def test_five_point_average_preserves_grid_and_suppresses_isolated_peak(self):
        scores = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
        filtered = moving_average(scores)
        self.assertEqual(len(filtered), len(scores))
        self.assertAlmostEqual(filtered[3], 0.2)
        self.assertEqual(group_events(filtered, 0.5), [])


if __name__ == "__main__":
    unittest.main()
