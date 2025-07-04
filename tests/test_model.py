import unittest
from src.model import build_model
from src.utils import load_config


class TestModel(unittest.TestCase):
    def test_build_model(self):
        config = load_config("config/config.yaml")
        model = build_model(config)
        self.assertIsNotNone(model)
        self.assertEqual(len(model.outputs), 2)


if __name__ == '__main__':
    unittest.main()
