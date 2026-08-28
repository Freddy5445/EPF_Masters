"""
Bidding-zone EIC codes and their market timezones.

EIC codes are taken from the Transparency Platform's "Area List with Energy
Identification Code (EIC)" article. The timezone is the one the platform
publishes the zone's daily data in -- generally where the zone is geographically
located, which is what determines where the 23-hour and 25-hour DST days fall.
"""

from collections import namedtuple

Area = namedtuple("Area", ["code", "eic", "tz", "name"])

_AREAS = [
    # --- Nordics -----------------------------------------------------------
    Area("DK1", "10YDK-1--------W", "Europe/Copenhagen", "Denmark West"),
    Area("DK2", "10YDK-2--------M", "Europe/Copenhagen", "Denmark East"),
    Area("FI", "10YFI-1--------U", "Europe/Helsinki", "Finland"),
    Area("NO1", "10YNO-1--------2", "Europe/Oslo", "Norway 1 (Oslo)"),
    Area("NO2", "10YNO-2--------T", "Europe/Oslo", "Norway 2 (Kristiansand)"),
    Area("NO3", "10YNO-3--------J", "Europe/Oslo", "Norway 3 (Trondheim)"),
    Area("NO4", "10YNO-4--------9", "Europe/Oslo", "Norway 4 (Tromso)"),
    Area("NO5", "10Y1001A1001A48H", "Europe/Oslo", "Norway 5 (Bergen)"),
    Area("SE1", "10Y1001A1001A44P", "Europe/Stockholm", "Sweden 1 (Lulea)"),
    Area("SE2", "10Y1001A1001A45N", "Europe/Stockholm", "Sweden 2 (Sundsvall)"),
    Area("SE3", "10Y1001A1001A46L", "Europe/Stockholm", "Sweden 3 (Stockholm)"),
    Area("SE4", "10Y1001A1001A47J", "Europe/Stockholm", "Sweden 4 (Malmo)"),
    # Sweden as a single control area. Installed capacity is published at this
    # level as well as per bidding zone, and the country total is not always the
    # sum of SE1-SE4: a unit sitting on a zone boundary, or one the TSO reports
    # only nationally, appears in the country figure alone.
    Area("SE", "10YSE-1--------K", "Europe/Stockholm", "Sweden"),
    # --- Central Western Europe -------------------------------------------
    Area("BE", "10YBE----------2", "Europe/Brussels", "Belgium"),
    Area("DE_LU", "10Y1001A1001A82H", "Europe/Berlin", "Germany-Luxembourg"),
    # Germany as a country, distinct from the DE-LU bidding zone above. Installed
    # capacity is reported per country, and the German TSOs report it here rather
    # than against the market area, so a capacity query for DE_LU can come back
    # empty where DE is populated.
    Area("DE", "10Y1001A1001A83F", "Europe/Berlin", "Germany"),
    Area("FR", "10YFR-RTE------C", "Europe/Paris", "France"),
    Area("NL", "10YNL----------L", "Europe/Amsterdam", "Netherlands"),
    Area("AT", "10YAT-APG------L", "Europe/Vienna", "Austria"),
    Area("CH", "10YCH-SWISSGRIDZ", "Europe/Zurich", "Switzerland"),
    # --- Others sometimes used as comparators ------------------------------
    Area("ES", "10YES-REE------0", "Europe/Madrid", "Spain"),
    Area("PT", "10YPT-REN------W", "Europe/Lisbon", "Portugal"),
    Area("IT_NORD", "10Y1001A1001A73I", "Europe/Rome", "Italy North"),
    Area("PL", "10YPL-AREA-----S", "Europe/Warsaw", "Poland"),
    Area("CZ", "10YCZ-CEPS-----N", "Europe/Prague", "Czech Republic"),
    Area("EE", "10Y1001A1001A39I", "Europe/Tallinn", "Estonia"),
    Area("LV", "10YLV-1001A00074", "Europe/Riga", "Latvia"),
    Area("LT", "10YLT-1001A0008Q", "Europe/Vilnius", "Lithuania"),
]

AREAS = {a.code: a for a in _AREAS}


def lookup(code):
    """Return the :class:`Area` for a zone code, e.g. ``"DK1"``."""
    try:
        return AREAS[code.upper()]
    except KeyError:
        known = ", ".join(sorted(AREAS))
        raise KeyError(f"Unknown bidding zone {code!r}. Known zones: {known}") from None
