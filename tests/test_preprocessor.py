import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os


class TestPreprocessor(unittest.TestCase):
    def test_scaler_save_load(self):
        X = np.random.rand(100, 10)
        scaler = StandardScaler()
        scaler.fit(X)
        joblib.dump(scaler, "models/preprocessor.pkl")

        self.assertTrue(os.path.exists("models/preprocessor.pkl"))

        loaded = joblib.load("models/preprocessor.pkl")
        X_trans = loaded.transform(X)
        self.assertEqual(X.shape, X_trans.shape)


if __name__ == '__main__':
    unittest.main()
