"""
Client tests. A fake session stands in for the network, so nothing here makes a
request or needs a token.
"""

import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import entsoe_tp.client as client_module  # noqa: E402
from entsoe_tp.client import (  # noqa: E402
    PAGE_SIZE, TokenError, TransparencyClient, _count_timeseries, _month_chunks,
    _redact, load_token,
)
from entsoe_tp.parser import TransparencyError  # noqa: E402


def document(n_timeseries):
    body = "".join(
        "<TimeSeries><mRID>1</mRID></TimeSeries>" for _ in range(n_timeseries)
    )
    return ('<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:'
            f'451-3:publicationdocument:7:3">{body}</Publication_MarketDocument>')


class FakeResponse:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


class FakeSession:
    """Returns queued responses and records the params it was called with."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        return self.responses.pop(0) if self.responses else FakeResponse(200, document(0))


class TestChunking(unittest.TestCase):
    def test_months_tile_the_range_without_gaps_or_overlap(self):
        start = pd.Timestamp("2016-01-01T00:00Z")
        end = pd.Timestamp("2016-04-01T00:00Z")
        chunks = _month_chunks(start, end)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0][0], start)
        self.assertEqual(chunks[-1][1], end)
        for (_, previous_end), (next_start, _) in zip(chunks, chunks[1:]):
            self.assertEqual(previous_end, next_start)

    def test_final_chunk_is_clipped_to_end(self):
        chunks = _month_chunks(pd.Timestamp("2016-01-01T00:00Z"),
                               pd.Timestamp("2016-01-10T00:00Z"))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][1], pd.Timestamp("2016-01-10T00:00Z"))


class TestPagination(unittest.TestCase):
    def setUp(self):
        # Throttling would make these tests sleep for no benefit.
        self._interval = client_module.MIN_REQUEST_INTERVAL
        client_module.MIN_REQUEST_INTERVAL = 0

    def tearDown(self):
        client_module.MIN_REQUEST_INTERVAL = self._interval

    def test_full_page_triggers_another_request(self):
        session = FakeSession([
            FakeResponse(200, document(PAGE_SIZE)),
            FakeResponse(200, document(7)),
        ])
        c = TransparencyClient(token="x", session=session)
        docs = c.fetch({"documentType": "A44"},
                       pd.Timestamp("2016-01-01T00:00Z"),
                       pd.Timestamp("2016-02-01T00:00Z"))

        self.assertEqual(len(docs), 2)
        self.assertNotIn("offset", session.calls[0])
        self.assertEqual(session.calls[1]["offset"], str(PAGE_SIZE))

    def test_partial_page_stops(self):
        session = FakeSession([FakeResponse(200, document(3))])
        c = TransparencyClient(token="x", session=session)
        c.fetch({"documentType": "A44"},
                pd.Timestamp("2016-01-01T00:00Z"),
                pd.Timestamp("2016-02-01T00:00Z"))

        self.assertEqual(len(session.calls), 1)

    def test_window_is_formatted_as_the_api_expects(self):
        session = FakeSession([FakeResponse(200, document(1))])
        c = TransparencyClient(token="x", session=session)
        c.fetch({"documentType": "A44"},
                pd.Timestamp("2016-01-01T00:00Z"),
                pd.Timestamp("2016-02-01T00:00Z"))

        self.assertEqual(session.calls[0]["periodStart"], "201601010000")
        self.assertEqual(session.calls[0]["periodEnd"], "201602010000")

    def test_token_is_attached_to_every_request(self):
        session = FakeSession([FakeResponse(200, document(1))])
        c = TransparencyClient(token="secret-token", session=session)
        c.fetch({"documentType": "A44"},
                pd.Timestamp("2016-01-01T00:00Z"),
                pd.Timestamp("2016-02-01T00:00Z"))

        self.assertEqual(session.calls[0]["securityToken"], "secret-token")


class TestErrors(unittest.TestCase):
    def setUp(self):
        self._interval = client_module.MIN_REQUEST_INTERVAL
        client_module.MIN_REQUEST_INTERVAL = 0

    def tearDown(self):
        client_module.MIN_REQUEST_INTERVAL = self._interval

    def test_401_fails_immediately_without_retrying(self):
        session = FakeSession([FakeResponse(401, "Unauthorized")])
        c = TransparencyClient(token="bad", session=session)

        with self.assertRaises(TokenError):
            c.fetch({}, pd.Timestamp("2016-01-01T00:00Z"),
                    pd.Timestamp("2016-02-01T00:00Z"))

        self.assertEqual(len(session.calls), 1)

    def test_unexpected_status_raises(self):
        session = FakeSession([FakeResponse(400, "Bad Request")])
        c = TransparencyClient(token="x", session=session)

        with self.assertRaises(TransparencyError):
            c.fetch({}, pd.Timestamp("2016-01-01T00:00Z"),
                    pd.Timestamp("2016-02-01T00:00Z"))


class TestRedaction(unittest.TestCase):
    def test_token_is_stripped_from_urls(self):
        url = ("https://web-api.tp.entsoe.eu/api?documentType=A44&"
               "securityToken=abc123def&periodStart=201601010000")
        cleaned = _redact(url)

        self.assertNotIn("abc123def", cleaned)
        self.assertIn("<redacted>", cleaned)
        self.assertIn("documentType=A44", cleaned)

    def test_handles_empty_input(self):
        self.assertEqual(_redact(""), "")
        self.assertIsNone(_redact(None))


class TestCache(unittest.TestCase):
    def setUp(self):
        self._interval = client_module.MIN_REQUEST_INTERVAL
        client_module.MIN_REQUEST_INTERVAL = 0

    def tearDown(self):
        client_module.MIN_REQUEST_INTERVAL = self._interval

    def test_second_call_is_served_from_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = FakeSession([FakeResponse(200, document(1))])
            c = TransparencyClient(token="x", session=session, cache_dir=tmp)
            window = (pd.Timestamp("2016-01-01T00:00Z"), pd.Timestamp("2016-02-01T00:00Z"))

            first = c.fetch({"documentType": "A44"}, *window)
            second = c.fetch({"documentType": "A44"}, *window)

            self.assertEqual(first, second)
            self.assertEqual(len(session.calls), 1)  # no second request

    def test_cache_key_excludes_the_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            window = (pd.Timestamp("2016-01-01T00:00Z"), pd.Timestamp("2016-02-01T00:00Z"))

            a = TransparencyClient(token="token-one", session=FakeSession(
                [FakeResponse(200, document(1))]), cache_dir=tmp)
            a.fetch({"documentType": "A44"}, *window)

            # A rotated token must still hit the same cache entry.
            session = FakeSession([FakeResponse(200, document(1))])
            b = TransparencyClient(token="token-two", session=session, cache_dir=tmp)
            b.fetch({"documentType": "A44"}, *window)

            self.assertEqual(len(session.calls), 0)

    def test_token_never_appears_in_a_cache_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = TransparencyClient(token="super-secret", session=FakeSession(
                [FakeResponse(200, document(1))]), cache_dir=tmp)
            c.fetch({"documentType": "A44"},
                    pd.Timestamp("2016-01-01T00:00Z"),
                    pd.Timestamp("2016-02-01T00:00Z"))

            for name in os.listdir(tmp):
                self.assertNotIn("super-secret", name)


class TestTokenLoading(unittest.TestCase):
    def test_environment_wins(self):
        os.environ["ENTSOE_API_TOKEN"] = "from-env"
        try:
            self.assertEqual(load_token(), "from-env")
        finally:
            del os.environ["ENTSOE_API_TOKEN"]

    def test_reads_dotenv_file(self):
        os.environ.pop("ENTSOE_API_TOKEN", None)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("# a comment\n\nENTSOE_API_TOKEN='from-file'\n")
            self.assertEqual(load_token(env_path=path), "from-file")

    def test_missing_token_explains_how_to_get_one(self):
        os.environ.pop("ENTSOE_API_TOKEN", None)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TokenError) as ctx:
                load_token(env_path=os.path.join(tmp, "absent.env"))
            self.assertIn("ENTSOE_API_TOKEN", str(ctx.exception))


class TestTimeSeriesCounting(unittest.TestCase):
    def test_counts_namespaced_elements(self):
        self.assertEqual(_count_timeseries(document(5)), 5)
        self.assertEqual(_count_timeseries(document(0)), 0)

    def test_malformed_xml_counts_as_zero(self):
        self.assertEqual(_count_timeseries("<not xml"), 0)


if __name__ == "__main__":
    unittest.main()
