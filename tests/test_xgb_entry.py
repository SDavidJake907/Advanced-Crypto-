import unittest

from core.models.xgb_entry import XGBEntryModel


class XGBEntryModelTests(unittest.TestCase):
    def test_build_feature_importance_report_sorts_features(self) -> None:
        model = XGBEntryModel()

        class _Stub:
            feature_importances_ = [0.2, 0.6, 0.1] + [0.0] * 12

        model.model = _Stub()
        report = model.build_feature_importance_report()

        self.assertTrue(report["model_loaded"])
        self.assertEqual(report["feature_count"], 15)
        self.assertEqual(report["ranked_features"][0]["feature"], "momentum_14")


if __name__ == "__main__":
    unittest.main()
