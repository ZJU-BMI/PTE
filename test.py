import unittest

import base.experiments


class TestBaseModel(unittest.TestCase):
    def test_base_model(self):
        base.experiments.test_model()


if __name__ == "__main__":
    unittest.main()
