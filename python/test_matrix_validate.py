import unittest

from matrix_validate import validate_cases


class MatrixValidationTests(unittest.TestCase):
    def test_accepts_named_safe_case(self):
        cases = validate_cases({"cases": [{"name": "safe", "target_offset_ned_m": [0.5, 0, 0],
                                            "miss_offset_y_m": -2.0, "repeats": 1}]})
        self.assertEqual(cases[0]["name"], "safe")

    def test_rejects_duplicate_names(self):
        case = {"name": "same", "target_offset_ned_m": [0, 0, 0],
                "miss_offset_y_m": -2.0, "repeats": 1}
        with self.assertRaises(ValueError):
            validate_cases({"cases": [case, dict(case)]})

    def test_rejects_out_of_bounds_position(self):
        with self.assertRaises(ValueError):
            validate_cases({"cases": [{"name": "unsafe", "target_offset_ned_m": [2, 0, 0],
                                        "miss_offset_y_m": -2.0, "repeats": 1}]})


if __name__ == "__main__":
    unittest.main()
