"""
Parser tests. No network: every document here is a literal fixture.

The A03 cases are the important ones -- that is the failure mode that produces a
plausible-looking series which is quietly wrong.
"""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entsoe_tp.parser import (  # noqa: E402
    TransparencyError, UnexpectedResolution, parse_document, to_series,
)

NS = 'xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"'


def price_document(points, resolution="PT60M", start="2016-01-01T23:00Z",
                   end="2016-01-02T23:00Z", contract="A01"):
    body = "".join(
        f"<Point><position>{p}</position><price.amount>{v}</price.amount></Point>"
        for p, v in points
    )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<Publication_MarketDocument {NS}>
  <type>A44</type>
  <TimeSeries>
    <mRID>1</mRID>
    <contract_MarketAgreement.type>{contract}</contract_MarketAgreement.type>
    <curveType>A03</curveType>
    <Period>
      <timeInterval><start>{start}</start><end>{end}</end></timeInterval>
      <resolution>{resolution}</resolution>
      {body}
    </Period>
  </TimeSeries>
</Publication_MarketDocument>"""


ACKNOWLEDGEMENT = """<?xml version="1.0" encoding="UTF-8"?>
<Acknowledgement_MarketDocument
    xmlns="urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0">
  <mRID>be3917c7-0704-4</mRID>
  <Reason><code>{code}</code><text>{text}</text></Reason>
</Acknowledgement_MarketDocument>"""


class TestCurveTypeA03(unittest.TestCase):
    """A gap in `position` means the previous value continues."""

    def test_gaps_are_forward_filled(self):
        # Positions 1, 5 and 20 published for a 24-slot period. Under A03 that
        # describes three flat blocks, not three hours of data.
        frame = parse_document(price_document([(1, 10.0), (5, 20.0), (20, 30.0)]),
                               "price.amount")
        series = to_series(frame)

        self.assertEqual(len(series), 24)
        self.assertEqual(series.iloc[0], 10.0)
        self.assertEqual(series.iloc[3], 10.0)   # slot 4 inherits slot 1
        self.assertEqual(series.iloc[4], 20.0)   # slot 5 changes
        self.assertEqual(series.iloc[18], 20.0)  # slot 19 inherits slot 5
        self.assertEqual(series.iloc[19], 30.0)  # slot 20 changes
        self.assertEqual(series.iloc[23], 30.0)  # carried to the end of the period

    def test_dense_points_are_unchanged(self):
        points = [(i, float(i)) for i in range(1, 25)]
        series = to_series(parse_document(price_document(points), "price.amount"))

        self.assertEqual(len(series), 24)
        self.assertEqual(list(series.values), [float(i) for i in range(1, 25)])

    def test_timestamps_are_utc_and_hourly(self):
        series = to_series(parse_document(price_document([(1, 5.0)]), "price.amount"))

        self.assertEqual(str(series.index.tz), "UTC")
        self.assertEqual(series.index[0], pd.Timestamp("2016-01-01T23:00Z"))
        self.assertEqual(series.index[-1], pd.Timestamp("2016-01-02T22:00Z"))

    def test_position_beyond_period_is_rejected(self):
        with self.assertRaises(TransparencyError):
            parse_document(price_document([(1, 1.0), (99, 2.0)]), "price.amount")


class TestResolution(unittest.TestCase):
    def test_quarter_hourly_raises_by_default(self):
        doc = price_document([(1, 10.0)], resolution="PT15M")
        with self.assertRaises(UnexpectedResolution) as ctx:
            parse_document(doc, "price.amount")
        self.assertIn("PT15M", str(ctx.exception))

    def test_quarter_hourly_parses_when_not_constrained(self):
        doc = price_document([(1, 10.0)], resolution="PT15M")
        series = to_series(parse_document(doc, "price.amount", expect_resolution=None))
        self.assertEqual(len(series), 96)


class TestSlotCountAcrossDstTransition(unittest.TestCase):
    """A period's bounds are UTC, but its slot count is in market time.

    A DST transition inside a period leaves the two an hour apart, so flooring
    the quotient undercounts the slots and rejects a valid document. Weekly
    reservoir data is where this shows: it made 12 of 15 zones fail to parse.
    """

    def test_weekly_period_spanning_spring_forward_parses(self):
        # The exact period the reservoir dump failed on. Five weekly slots, but
        # only 34d23h of UTC because clocks went forward on 2016-03-27, so the
        # declared position 5 exceeded the floored slot count of 4.
        document = price_document(
            [(1, 10.0), (2, 20.0), (3, 30.0), (4, 40.0), (5, 50.0)],
            resolution="P7D",
            start="2016-02-28T23:00Z", end="2016-04-03T22:00Z",
        )

        frame = parse_document(document, "price.amount", expect_resolution=None)

        self.assertEqual(len(frame), 5)
        self.assertEqual(list(frame["value"]), [10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertEqual(list(frame["resolution"].unique()), ["P7D"])

    def test_the_last_weekly_slot_is_no_longer_dropped(self):
        """Before the fix this raised; the fifth week is the one at stake."""
        document = price_document(
            [(1, 10.0), (5, 50.0)], resolution="P7D",
            start="2016-02-28T23:00Z", end="2016-04-03T22:00Z",
        )
        frame = parse_document(document, "price.amount", expect_resolution=None)

        # A03 carries 10.0 across weeks 2-4, then week 5 publishes 50.0.
        self.assertEqual(list(frame["value"]), [10.0, 10.0, 10.0, 10.0, 50.0])

    def test_the_period_really_is_short_of_five_whole_weeks(self):
        """Guards the premise: without the rounding this input would floor to 4."""
        span = pd.Timestamp("2016-04-03T22:00Z") - pd.Timestamp("2016-02-28T23:00Z")
        self.assertEqual(span, pd.Timedelta(days=34, hours=23))
        self.assertEqual(int(span / pd.Timedelta(days=7)), 4)
        self.assertEqual(round(span / pd.Timedelta(days=7)), 5)

    def test_hourly_slot_counts_are_unchanged(self):
        """The quotient is exact at PT60M, so rounding must not alter anything."""
        document = price_document([(1, 10.0)],
                                  start="2016-01-01T23:00Z",
                                  end="2016-01-02T23:00Z")
        self.assertEqual(len(parse_document(document, "price.amount")), 24)

    def test_a_period_overdeclaring_by_more_than_one_still_raises(self):
        """Rounding absorbs the DST hour, not genuine corruption."""
        document = price_document([(30, 1.0)], resolution="P7D",
                                  start="2016-02-28T23:00Z",
                                  end="2016-04-03T22:00Z")
        with self.assertRaises(TransparencyError):
            parse_document(document, "price.amount", expect_resolution=None)


class TestAcknowledgement(unittest.TestCase):
    def test_no_matching_data_is_empty_not_an_error(self):
        doc = ACKNOWLEDGEMENT.format(code="999", text="No matching data found")
        frame = parse_document(doc, "price.amount")
        self.assertTrue(frame.empty)
        self.assertTrue(to_series(frame).empty)

    def test_other_reasons_raise(self):
        doc = ACKNOWLEDGEMENT.format(code="413", text="Data set too large")
        with self.assertRaises(TransparencyError) as ctx:
            parse_document(doc, "price.amount")
        self.assertIn("413", str(ctx.exception))


class TestRealSample(unittest.TestCase):
    """Regression against a real published document.

    This is the Austrian day-ahead price response shipped as the example for
    12.1.D in ENTSO-E's own Postman collection (2024-07-28, PT15M). It publishes
    **95 points for a 96-slot day** -- position 14 is simply absent, because the
    price did not change from position 13. That is A03 working exactly as
    specified, in real data, and it is the case a dense parser gets wrong.
    """

    # position:price pairs exactly as published, position 14 absent.
    PAIRS = (
        "1:104.98,2:94.98,3:79.98,4:71.3,5:89.19,6:82.16,7:72.42,8:54.98,9:80.99,"
        "10:79.98,11:63.6,12:54.98,13:79.98,15:55.02,16:50.45,17:74.98,18:69.97,"
        "19:54.98,20:52.6,21:65.58,22:64.56,23:51.13,24:50.04,25:60.43,26:55.02,"
        "27:54.98,28:53.89,29:74.91,30:52.58,31:50.05,32:47.51,33:75.39,34:56.39,"
        "35:36.54,36:32.57,37:64.94,38:40.05,39:12.58,40:-30.71,41:64.95,42:25.04,"
        "43:0.21,44:-39.45,45:47.42,46:6.98,47:-13.65,48:-18.24,49:22.41,50:-12.47,"
        "51:-20.16,52:-17.47,53:-37.79,54:-37.6,55:-30.82,56:-39.57,57:-58.29,"
        "58:-55.26,59:-52.64,60:-21.84,61:-58.92,62:-51.02,63:-46.66,64:32.5,"
        "65:-62.54,66:-41.35,67:12.51,68:34.48,69:-82.82,70:-5.5,71:42.48,72:79.98,"
        "73:5.03,74:28.09,75:74.91,76:79.98,77:49.47,78:62.45,79:80.99,80:85.54,"
        "81:80.99,82:87.43,83:86.93,84:90.04,85:97.42,86:80.08,87:79.98,88:79.91,"
        "89:96.22,90:89.91,91:79.92,92:69.95,93:99.94,94:80.99,95:74.91,96:51.59"
    )

    def setUp(self):
        points = [tuple(p.split(":")) for p in self.PAIRS.split(",")]
        self.points = [(int(pos), float(val)) for pos, val in points]
        self.doc = price_document(
            self.points, resolution="PT15M",
            start="2024-07-27T22:00Z", end="2024-07-28T22:00Z",
        )

    def test_sample_really_is_short_one_point(self):
        self.assertEqual(len(self.points), 95)
        self.assertEqual(max(p for p, _ in self.points), 96)

    def test_expands_to_a_full_day(self):
        series = to_series(parse_document(self.doc, "price.amount",
                                          expect_resolution=None))
        self.assertEqual(len(series), 96)

    def test_absent_position_carries_the_previous_price(self):
        series = to_series(parse_document(self.doc, "price.amount",
                                          expect_resolution=None))
        # Slot 14 must repeat slot 13's 79.98. Reading the points densely would
        # put 55.02 here and shift the remaining 82 slots 15 minutes early.
        self.assertEqual(series.iloc[12], 79.98)
        self.assertEqual(series.iloc[13], 79.98)
        self.assertEqual(series.iloc[14], 55.02)
        self.assertEqual(series.iloc[-1], 51.59)

    def test_aggregating_to_hourly_matches_published_day(self):
        series = to_series(parse_document(self.doc, "price.amount",
                                          expect_resolution=None))
        hourly = series.resample("1h").mean()
        self.assertEqual(len(hourly), 24)
        # First hour is slots 1-4: mean of 104.98, 94.98, 79.98, 71.3.
        self.assertAlmostEqual(hourly.iloc[0], (104.98 + 94.98 + 79.98 + 71.3) / 4)


class TestAggregation(unittest.TestCase):
    def test_overlapping_publications_are_not_summed(self):
        """Two documents covering the same hours must not double the price."""
        a = parse_document(price_document([(1, 10.0)]), "price.amount")
        b = parse_document(price_document([(1, 10.0)]), "price.amount")
        series = to_series(pd.concat([a, b], ignore_index=True), aggregate="first")

        self.assertEqual(len(series), 24)
        self.assertEqual(series.iloc[0], 10.0)

    def test_production_types_are_summed(self):
        """A69 arrives as one TimeSeries per production type; they add up."""
        solar = parse_document(price_document([(1, 100.0)]), "price.amount")
        wind = parse_document(price_document([(1, 250.0)]), "price.amount")
        solar["psr_type"] = "B16"
        wind["psr_type"] = "B19"

        series = to_series(pd.concat([solar, wind], ignore_index=True), aggregate="sum")

        self.assertEqual(len(series), 24)
        self.assertEqual(series.iloc[0], 350.0)

    def test_repeated_production_type_is_deduplicated_before_summing(self):
        solar = parse_document(price_document([(1, 100.0)]), "price.amount")
        solar["psr_type"] = "B16"
        duplicate = solar.copy()

        series = to_series(pd.concat([solar, duplicate], ignore_index=True),
                           aggregate="sum")

        self.assertEqual(series.iloc[0], 100.0)


if __name__ == "__main__":
    unittest.main()
