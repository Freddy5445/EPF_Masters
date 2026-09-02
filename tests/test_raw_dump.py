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
    merge_requery, variables_in_part,
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
        shaped = _shape(pd.DataFrame(), "DE_LU", "EIC", "generation_forecast",
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

    def test_every_default_zone_resolves(self):
        from entsoe_tp.areas import lookup

        self.assertEqual(len(DEFAULT_ZONES), 13)
        for zone in DEFAULT_ZONES:
            self.assertTrue(lookup(zone).eic)

    def test_de_lu_is_captured_with_the_nordics(self):
        """DE-LU is the market the Nordic zones are coupled to, so its price and
        forecasts are captured on the same footing rather than separately."""
        self.assertIn("DE_LU", DEFAULT_ZONES)

    def test_the_baltic_zones_are_gone(self):
        """Dropped for unusable 15-minute generation forecasts; see section 4 of
        data_cleaning.ipynb. They must not come back through areas.py either, or
        a stale reference would start resolving again."""
        from entsoe_tp.areas import AREAS

        for zone in ("EE", "LV", "LT"):
            self.assertNotIn(zone, DEFAULT_ZONES)
            self.assertNotIn(zone, AREAS)

    def test_four_data_items_are_queried(self):
        queries = _queries("EIC")
        self.assertEqual(sorted(queries), ["generation_forecast", "load_forecast",
                                           "price", "reservoir"])
        self.assertEqual(queries["price"]["params"]["documentType"], "A44")
        self.assertEqual(queries["load_forecast"]["params"]["documentType"], "A65")
        self.assertEqual(queries["generation_forecast"]["params"]["documentType"],
                         "A69")
        self.assertEqual(queries["reservoir"]["params"]["documentType"], "A72")
        self.assertEqual(queries["reservoir"]["params"]["processType"], "A16")

    def test_weekly_reservoir_needs_no_special_handling(self):
        """P7D is recorded like any other resolution; nothing resamples it."""
        weekly = (f'<GL_MarketDocument {NS}><TimeSeries>'
                  f'<quantity_Measure_Unit.name>MWH</quantity_Measure_Unit.name>'
                  f'<Period><timeInterval><start>2020-01-06T00:00Z</start>'
                  f'<end>2020-01-27T00:00Z</end></timeInterval>'
                  f'<resolution>P7D</resolution>'
                  + "".join(f"<Point><position>{i}</position>"
                            f"<quantity>{1000 * i}</quantity></Point>"
                            for i in range(1, 4))
                  + f'</Period></TimeSeries></GL_MarketDocument>')

        frame = parse_document(weekly, "quantity", expect_resolution=None)
        shaped = _shape(frame, "NO2", "EIC", "reservoir", {"documentType": "A72"})

        self.assertEqual(list(shaped["resolution"].unique()), ["P7D"])
        self.assertEqual(len(shaped), 3)
        deltas = shaped["timestamp_utc"].diff().dropna().unique()
        self.assertEqual(list(deltas), [pd.Timedelta(days=7)])


class TestResumeIsPerSeries(unittest.TestCase):
    """A part written before a data item existed is incomplete, not stale.

    Resuming by zone would skip it entirely and never fetch the new series;
    rebuilding it would re-fetch the three already downloaded. Neither is what
    "add the missing series" means, so resume works per data item.
    """

    def part(self, tmp, variables):
        from entsoe_tp.raw_dump import _write_part

        frames = []
        for variable in variables:
            frame = parse_document(
                price_document("2020-01-06T00:00Z", "2020-01-06T02:00Z",
                               "PT60M", 2),
                "price.amount", expect_resolution=None)
            frames.append(_shape(frame, "NO2", "EIC", variable,
                                 {"documentType": "A44"}))
        path = os.path.join(tmp, "NO2.parquet")
        _write_part(pd.concat(frames, ignore_index=True), path)
        return path

    def test_reports_which_series_a_part_holds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.part(tmp, ["price", "load_forecast"])
            self.assertEqual(variables_in_part(path),
                             {"price", "load_forecast"})

    def test_a_missing_part_holds_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(variables_in_part(os.path.join(tmp, "nope.parquet")),
                             set())

    def test_the_new_series_is_identified_as_missing(self):
        """The case that matters: a part built before reservoir existed."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self.part(tmp, ["price", "load_forecast",
                                   "generation_forecast"])
            have = variables_in_part(path)
            missing = [v for v in _queries("x") if v not in have]
            self.assertEqual(missing, ["reservoir"])

    def test_a_complete_part_leaves_nothing_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.part(tmp, list(_queries("x")))
            have = variables_in_part(path)
            self.assertEqual([v for v in _queries("x") if v not in have], [])

    def test_fetch_zone_can_be_limited_to_one_series(self):
        from entsoe_tp.raw_dump import fetch_zone

        calls = []

        class Client:
            def fetch(self, params, start, end, progress=None):
                calls.append(params["documentType"])
                return []

        fetch_zone(Client(), "NO2", pd.Timestamp("2020-01-01", tz="UTC"),
                   pd.Timestamp("2020-01-02", tz="UTC"),
                   variables=["reservoir"], quiet=True)

        self.assertEqual(calls, ["A72"])


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


class TestRequeryMerge(unittest.TestCase):

    def row(self, timestamp, value, resolution="PT60M"):
        frame = parse_document(
            price_document(timestamp, str(pd.Timestamp(timestamp) + pd.Timedelta(hours=1)),
                           resolution, 1 if resolution == "PT60M" else 4),
            "price.amount", expect_resolution=None)
        return _shape(frame, "DK1", "EIC", "price", {"documentType": "A44"})

    def test_existing_key_is_not_appended(self):
        existing = self.row("2020-01-01T00:00Z", 41)
        merged, additions, conflicts = merge_requery(existing, existing.copy())

        self.assertEqual(len(merged), len(existing))
        self.assertTrue(additions.empty)
        self.assertTrue(conflicts.empty)

    def test_new_native_resolution_rows_are_appended(self):
        existing = self.row("2020-01-01T00:00Z", 41)
        fetched = self.row("2025-04-09T00:00Z", 41, resolution="PT15M")
        merged, additions, conflicts = merge_requery(existing, fetched)

        self.assertEqual(len(additions), 4)
        self.assertEqual(set(additions["resolution"]), {"PT15M"})
        self.assertEqual(len(merged), len(existing) + 4)
        self.assertTrue(conflicts.empty)

    def test_conflicting_existing_value_is_reported_not_appended(self):
        existing = self.row("2020-01-01T00:00Z", 41)
        fetched = existing.copy()
        fetched["value"] += 10
        merged, additions, conflicts = merge_requery(existing, fetched)

        self.assertEqual(len(merged), len(existing))
        self.assertTrue(additions.empty)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts.iloc[0]["existing_value"], 41)
        self.assertEqual(conflicts.iloc[0]["queried_value"], 51)


if __name__ == "__main__":
    unittest.main()
