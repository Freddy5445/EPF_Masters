"""Tests for the installed-capacity download [14.1.A].

A68 is a stock reported once a year, not a time series, so it does not go
through the hourly period expansion the rest of the package uses. What is pinned
here is that shape: one value per production type per zone-year, an
acknowledgement understood as "nothing published" rather than an error, and a
table whose columns mean what their headers say.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entsoe_tp.areas import lookup  # noqa: E402
from entsoe_tp.client import TransparencyError  # noqa: E402
from entsoe_tp.installed_capacity import (  # noqa: E402
    DEFAULT_YEARS, DEFAULT_ZONES, PSR_TYPES, collect, fetch_year, main,
    parse_installed_capacity, to_wide,
)

NS = 'xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"'


def document(entries, unit="MAW"):
    """An A68 document carrying ``[(psrType, quantity), ...]``."""
    series = "".join(f"""
  <TimeSeries>
    <mRID>{n}</mRID><businessType>A37</businessType>
    <quantity_Measure_Unit.name>{unit}</quantity_Measure_Unit.name>
    <curveType>A01</curveType>
    <MktPSRType><psrType>{psr}</psrType></MktPSRType>
    <Period>
      <timeInterval><start>2019-12-31T23:00Z</start><end>2020-12-31T23:00Z</end></timeInterval>
      <resolution>P1Y</resolution>
      <Point><position>1</position><quantity>{qty}</quantity></Point>
    </Period>
  </TimeSeries>""" for n, (psr, qty) in enumerate(entries, 1))
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<GL_MarketDocument {NS}>{series}\n</GL_MarketDocument>'


ACKNOWLEDGEMENT = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Acknowledgement_MarketDocument xmlns="urn:iec62325.351:tc57wg16:'
    '451-7:acknowledgementdocument:8:0">'
    '<Reason><code>999</code><text>No matching data found</text></Reason>'
    '</Acknowledgement_MarketDocument>')


class TestParsing(unittest.TestCase):

    def test_one_row_per_production_type(self):
        rows = parse_installed_capacity(document([("B19", 3665), ("B18", 1277)]))
        self.assertEqual([(r["psr_type"], r["value"]) for r in rows],
                         [("B19", 3665.0), ("B18", 1277.0)])

    def test_unit_is_normalised(self):
        """The platform writes MAW; nobody else does."""
        rows = parse_installed_capacity(document([("B19", 100)]))
        self.assertEqual(rows[0]["unit"], "MW")

    def test_no_data_is_empty_not_an_error(self):
        """A zone that did not report a year is normal, especially early on."""
        self.assertEqual(parse_installed_capacity(ACKNOWLEDGEMENT), [])

    def test_malformed_xml_is_reported(self):
        with self.assertRaises(TransparencyError):
            parse_installed_capacity("<not xml")

    def test_a_point_without_a_quantity_is_skipped(self):
        xml = document([("B19", 1)]).replace("<quantity>1</quantity>", "")
        self.assertEqual(parse_installed_capacity(xml), [])


class TestCodeList(unittest.TestCase):

    def test_the_types_this_study_cares_about_are_named(self):
        for code, name in (("B16", "Solar"), ("B18", "Wind offshore"),
                           ("B19", "Wind onshore"), ("B12", "Hydro water reservoir"),
                           ("B10", "Hydro pumped storage"), ("B14", "Nuclear")):
            self.assertEqual(PSR_TYPES[code], name)

    def test_codes_are_contiguous(self):
        """A gap would mean a code was dropped when the list was transcribed."""
        numbers = sorted(int(c[1:]) for c in PSR_TYPES)
        self.assertEqual(numbers, list(range(1, len(numbers) + 1)))


class TestZones(unittest.TestCase):

    def test_every_default_zone_resolves(self):
        for zone in DEFAULT_ZONES:
            self.assertTrue(lookup(zone).eic)

    def test_sweden_is_the_country_not_a_bidding_zone(self):
        """Installed capacity is wanted for Sweden as a whole, not SE1-SE4."""
        self.assertIn("SE", DEFAULT_ZONES)
        for bidding_zone in ("SE1", "SE2", "SE3", "SE4"):
            self.assertNotIn(bidding_zone, DEFAULT_ZONES)
        self.assertNotEqual(lookup("SE").eic, lookup("SE3").eic)

    def test_germany_is_the_country_not_the_bidding_zone(self):
        """Installed capacity is reported nationally, against the country EIC.

        A query against the DE-LU market area can come back empty where Germany
        is populated, so the two must not be confused.
        """
        self.assertIn("DE", DEFAULT_ZONES)
        self.assertNotIn("DE_LU", DEFAULT_ZONES)
        self.assertEqual(lookup("DE").eic, "10Y1001A1001A83F")
        self.assertNotEqual(lookup("DE").eic, lookup("DE_LU").eic)

    def test_the_de_lu_bidding_zone_is_still_available(self):
        """It is what the price and load series use; only capacity moved."""
        self.assertEqual(lookup("DE_LU").eic, "10Y1001A1001A82H")

    def test_countries_and_bidding_zones_have_distinct_codes(self):
        """A duplicated EIC would silently make two zones the same query."""
        from entsoe_tp.areas import AREAS
        eics = [a.eic for a in AREAS.values()]
        self.assertEqual(len(eics), len(set(eics)))

    def test_default_years(self):
        self.assertEqual(DEFAULT_YEARS, [2015, 2020, 2025])


class TestFetch(unittest.TestCase):

    def test_one_request_per_zone_year(self):
        """client.fetch would chunk a year into twelve identical requests."""
        client = mock.Mock()
        client._get.return_value = document([("B19", 3665)])
        rows = fetch_year(client, "DK1", 2020)

        self.assertEqual(client._get.call_count, 1)
        params = client._get.call_args[0][0]
        self.assertEqual(params["documentType"], "A68")
        self.assertEqual(params["processType"], "A33")
        self.assertEqual(params["in_Domain"], lookup("DK1").eic)
        self.assertEqual(params["periodStart"], "202001010000")
        self.assertEqual(params["periodEnd"], "202101010000")
        self.assertEqual(rows[0]["zone"], "DK1")
        self.assertEqual(rows[0]["year"], 2020)

    def test_an_empty_zone_year_does_not_fail_the_run(self):
        client = mock.Mock()
        client._get.side_effect = [document([("B19", 1)]), ACKNOWLEDGEMENT]
        with mock.patch("entsoe_tp.installed_capacity.TransparencyClient",
                        return_value=client):
            frame, empty = collect(["DK1"], [2015, 2020], quiet=True)
        self.assertEqual(len(frame), 1)
        self.assertEqual(empty, [("DK1", 2020)])

    def test_nothing_anywhere_is_an_error(self):
        client = mock.Mock()
        client._get.return_value = ACKNOWLEDGEMENT
        with mock.patch("entsoe_tp.installed_capacity.TransparencyClient",
                        return_value=client):
            with self.assertRaises(TransparencyError):
                collect(["DK1"], [2015], quiet=True)


class TestTable(unittest.TestCase):

    def _frame(self):
        return pd.DataFrame([
            {"zone": "DK1", "area_name": "Denmark West", "eic": "x", "year": 2020,
             "psr_type": "B19", "production_type": "Wind onshore",
             "value": 3665.0, "unit": "MW"},
            {"zone": "DK1", "area_name": "Denmark West", "eic": "x", "year": 2020,
             "psr_type": "B04", "production_type": "Fossil gas",
             "value": 800.0, "unit": "MW"},
            {"zone": "NO1", "area_name": "Norway 1", "eic": "y", "year": 2020,
             "psr_type": "B12", "production_type": "Hydro water reservoir",
             "value": 5000.0, "unit": "MW"},
        ])

    def test_one_row_per_zone_year(self):
        wide = to_wide(self._frame())
        self.assertEqual(list(wide[["zone", "year"]].itertuples(index=False, name=None)),
                         [("DK1", 2020), ("NO1", 2020)])

    def test_absent_types_are_zero_not_blank(self):
        """A zone reporting a breakdown without lignite has no lignite."""
        wide = to_wide(self._frame())
        self.assertEqual(wide.loc[wide.zone == "NO1", "Wind onshore"].iloc[0], 0.0)
        self.assertFalse(wide.isna().any().any())

    def test_total_is_the_row_sum(self):
        wide = to_wide(self._frame())
        row = wide[wide.zone == "DK1"].iloc[0]
        self.assertAlmostEqual(row["Total"], 4465.0)

    def test_columns_follow_the_platform_code_order(self):
        """So the breakdown reads the way the published tables do."""
        wide = to_wide(self._frame())
        types = [c for c in wide.columns if c not in ("zone", "year", "Total")]
        order = [n for n in PSR_TYPES.values() if n in types]
        self.assertEqual(types, order)


class TestCLI(unittest.TestCase):

    def test_writes_a_csv(self):
        client = mock.Mock()
        client._get.return_value = document([("B19", 3665), ("B16", 20)])
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "installed_capacity.csv")
            with mock.patch("entsoe_tp.installed_capacity.TransparencyClient",
                            return_value=client):
                code = main(["--zones", "DK1", "--years", "2020", "--out", out,
                             "--quiet"])
            self.assertEqual(code, 0)
            wide = pd.read_csv(out)
            self.assertEqual(list(wide["zone"]), ["DK1"])
            self.assertAlmostEqual(wide["Total"].iloc[0], 3685.0)

    def test_long_format(self):
        client = mock.Mock()
        client._get.return_value = document([("B19", 3665), ("B16", 20)])
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "long.csv")
            with mock.patch("entsoe_tp.installed_capacity.TransparencyClient",
                            return_value=client):
                main(["--zones", "DK1", "--years", "2020", "--out", out,
                      "--long", "--quiet"])
            long = pd.read_csv(out)
            self.assertEqual(len(long), 2)
            self.assertIn("production_type", long.columns)

    def test_unknown_zone_is_refused(self):
        self.assertEqual(main(["--zones", "XX9", "--years", "2020", "--quiet"]), 1)

    def test_bad_years_are_refused(self):
        self.assertEqual(main(["--zones", "DK1", "--years", "twenty", "--quiet"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
