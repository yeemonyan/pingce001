import unittest

from scripts.filter_split_records import filter_records, load_ids


class FilterSplitRecordsTest(unittest.TestCase):
    def test_filter_records_preserves_id_order(self):
        by_id = {
            "b": {"id": "b"},
            "a": {"id": "a"},
        }
        self.assertEqual(filter_records(by_id, ["a", "b"]), [{"id": "a"}, {"id": "b"}])

    def test_filter_records_rejects_missing_id(self):
        with self.assertRaises(ValueError):
            filter_records({"a": {"id": "a"}}, ["a", "missing"])


if __name__ == "__main__":
    unittest.main()
