"""
Tests for the raw multi-zone dump.

The dump exists to study resolution changes, availability and missingness, so
what is tested here is mostly that it *refuses to normalise*: mixed resolutions
survive, unexpected production types survive, and an absent data item is
recorded rather than raised.
"""

import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entsoe_tp.parser import parse_document  # noqa: E402
from entsoe_tp.raw_dump import (  # noqa: E402
    COLUMNS, DEFAULT_ZONES, _queries, _shape, combine_parts,
)

NS = 'xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"'


def price_document(start, end, resolution, n_points):
    points = "".join(
        f"<Point><position>{i}</position><price.amount>{40 + i}</price.amount></Point>"
        for i in range(1, n_points + 1)
    )
    return (f'<Publication_MarketDocument {NS}><TimeSeries>'
            f'<contract_MarketAgreement.type>A01</contract_MarketAgreement.type>'
            f'<currency_Unit.name>EUR</currency_Unit.name>'
            f'<price_Measure_Unit.name>MWH</price_Measure_Unit.name>'
            f'<Period><timeInterval><start>{start}</start><end>{end}</end></timeInterval>'
            f'<resolution>{resolution}</resolution>{points}'
            f'</Period></TimeSeries></Publication_MarketDocument>')


class TestResolutionIsPerObservation(unittest.TestCase):
    """A range spanning a market time unit change carries both resolutions."""

    def test_parser_records_resolution_on_every_row(self):
        frame = parse_document(
            price_document("2025-10-01T00:00Z", "2025-10-01T01:00Z", "PT15M", 4),
            "price.amount", expect_resolution=None,
        )
        self.assertIn("resolution", frame.columns)
        self.assertEqual(list(frame["resolution"]), ["PT15M"] * 4)

    def test_quarter_hourly_timestamps_are_15_minutes_apart(self):
        frame = parse_document(
            price_document("2025-10-01T00:00Z", "2025-10-01T01:00Z", "PT15M", 4),
            "price.amount", expect_resolution=None,
        )
        deltas = frame["timestamp"].diff().dropna().unique()
        self.assertEqual(list(deltas), [pd.Timedelta(minutes=15)])

    def test_mixed_resolutions_survive_concatenation(self):
        """Two documents either side of the switchover, as the client returns them."""
        hourly = parse_document(
            price_document("2025-09-01T00:00Z", "2025-09-01T03:00Z", "PT60M", 3),
            "price.amount", expect_resolution=None)
        quarterly = parse_document(
            price_document("2025-10-01T00:00Z", "2025-10-01T01:00Z", "PT15M", 4),
            "price.amount", expect_resolution=None)

        combined = pd.concat([hourly, quarterly], ignore_index=True)
        shaped = _shape(combined, "SE3", "EIC", "price", {"documentType": "A44"})

        self.assertEqual(sorted(shaped["resolution"].unique()), ["PT15M", "PT60M"])
        self.assertEqual(len(shaped), 7)


class TestShaping(unittest.TestCase):

    def test_output_schema_is_fixed(self):
        frame = parse_document(
            price_document("2016-01-01T00:00Z", "2016-01-01T02:00Z", "PT60M", 2),
            "price.amount", expect_resolution=None)
        shaped = _shape(frame, "DK1", "10YDK-1--------W", "price",
                        {"documentType": "A44"})
        self.assertEqual(list(shaped.columns), COLUMNS)
        self.assertEqual(set(shaped["zone"]), {"DK1"})
        self.assertEqual(set(shaped["currency"]), {"EUR"})

    def test_empty_frame_still_has_the_schema(self):
        """An absent data item must not break the concatenation."""
        shaped = _shape(pd.DataFrame(), "LV", "EIC", "generation_forecast",
                        {"documentType": "A69"})
        self.assertTrue(shaped.empty)
        self.assertEqual(list(shaped.columns), COLUMNS)

    def test_timestamps_stay_utc(self):
        frame = parse_document(
            price_document("2016-01-01T00:00Z", "2016-01-01T02:00Z", "PT60M", 2),
            "price.amount", expect_resolution=None)
        shaped = _shape(frame, "FI", "EIC", "price", {"documentType": "A44"})
        self.assertEqual(str(shaped["timestamp_utc"].dt.tz), "UTC")


class TestProductionTypesAreNotFiltered(unittest.TestCase):
    """Which types a zone publishes is a finding, so nothing is dropped."""

    def generation_document(self, psr_types):
        series = "".join(
            f'<TimeSeries><quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name>'
            f'<MktPSRType><psrType>{psr}</psrType></MktPSRType>'
            f'<Period><timeInterval><start>2016-01-01T00:00Z</start>'
            f'<end>2016-01-01T01:00Z</end></timeInterval><resolution>PT60M</resolution>'
            f'<Point><position>1</position><quantity>{value}</quantity></Point>'
            f'</Period></TimeSeries>'
            for psr, value in psr_types
        )
        return f'<GL_MarketDocument {NS}>{series}</GL_MarketDocument>'

    def test_all_returned_types_are_kept(self):
        frame = parse_document(
            self.generation_document([("B16", 100), ("B18", 700), ("B19", 300)]),
            "quantity", expect_resolution=None)
        shaped = _shape(frame, "DK1", "EIC", "generation_forecast",
                        {"documentType": "A69"})
        self.assertEqual(sorted(shaped["psr_type"].unique()), ["B16", "B18", "B19"])

    def test_an_unexpected_type_is_kept_not_dropped(self):
        frame = parse_document(self.generation_document([("B01", 5)]),
                               "quantity", expect_resolution=None)
        shaped = _shape(frame, "NO4", "EIC", "generation_forecast",
                        {"documentType": "A69"})
        self.assertEqual(list(shaped["psr_type"].unique()), ["B01"])


class TestZonesAndQueries(unittest.TestCase):

    def test_all_nordic_and_baltic_zones_resolve(self):
        from entsoe_tp.areas import lookup

        self.assertEqual(len(DEFAULT_ZONES), 15)
        for zone in DEFAULT_ZONES:
            self.assertTrue(lookup(zone).eic)

    def test_three_data_items_are_queried(self):
        queries = _queries("EIC")
        self.assertEqual(sorted(queries), ["generation_forecast", "load_forecast",
                                           "price"])
        self.assertEqual(queries["price"]["params"]["documentType"], "A44")
        self.assertEqual(queries["load_forecast"]["params"]["documentType"], "A65")
        self.assertEqual(queries["generation_forecast"]["params"]["documentType"],
                         "A69")


class TestCombineParts(unittest.TestCase):

    def test_parts_are_streamed_into_one_file(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow not installed")

        from entsoe_tp.raw_dump import _write_part

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for zone in ("DK1", "SE3"):
                frame = parse_document(
                    price_document("2016-01-01T00:00Z", "2016-01-01T02:00Z",
                                   "PT60M", 2),
                    "price.amount", expect_resolution=None)
                shaped = _shape(frame, zone, "EIC", "price",
                                {"documentType": "A44"})
                path = os.path.join(tmp, f"{zone}.parquet")
                _write_part(shaped, path)
                paths.append(path)

            out = os.path.join(tmp, "all.parquet")
            total = combine_parts(paths, out)

            self.assertEqual(total, 4)
            combined = pd.read_parquet(out)
            self.assertEqual(sorted(combined["zone"].unique()), ["DK1", "SE3"])


if __name__ == "__main__":
    unittest.main()
