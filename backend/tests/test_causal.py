import unittest
import numpy as np
from causal import prepare_split, prices_from_returns


class CausalityTests(unittest.TestCase):
    def arrays(self):
        t = np.arange(160)
        return np.column_stack((t, t * 2)).astype(float), np.column_stack((t / 1000, t / 500)), t + 100., t

    def test_future_perturbations_cannot_change_training(self):
        x, y, c, p = self.arrays()
        first = prepare_split(x, y, c, p, sequence_length=10)
        boundary = first.val_origins[0]
        x[boundary:] += 100000
        y[boundary:] += 100000
        c[boundary:] += 100000
        second = prepare_split(x, y, c, p, sequence_length=10)
        np.testing.assert_array_equal(first.X_train, second.X_train)
        np.testing.assert_array_equal(first.y_train, second.y_train)
        np.testing.assert_array_equal(first.feature_scaler.mean_, second.feature_scaler.mean_)
        np.testing.assert_array_equal(first.target_scaler.mean_, second.target_scaler.mean_)

    def test_current_close_alignment_and_purge(self):
        x, y, c, p = self.arrays()
        s = prepare_split(x, y, c, p, sequence_length=30)
        self.assertTrue(np.all(s.train_origins + y.shape[1] < s.val_origins[0]))
        self.assertEqual(s.purged_samples, 2)
        np.testing.assert_array_equal(s.X_train[:, -1, -1], c[s.train_origins])
        np.testing.assert_array_equal(s.X_val[:, -1, -1], c[s.val_origins])
        # The old +29 close shift would expose future values at every origin.
        self.assertNotEqual(s.X_train[0, -1, -1], c[s.train_origins[0] + 29])

    def test_cumulative_horizons_use_one_anchor(self):
        np.testing.assert_allclose(prices_from_returns([100], [[.1, .2]]), [[110., 120.]])

    def test_invalid_or_unsorted_input_rejected(self):
        x, y, c, p = self.arrays()
        for positions in (p[::-1], p[:3]):
            with self.assertRaises(ValueError):
                prepare_split(x, y, c, positions)
        x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            prepare_split(x, y, c, p)


if __name__ == '__main__':
    unittest.main()
