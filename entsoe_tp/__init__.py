"""
Client for the ENTSO-E Transparency Platform RESTful API.

Downloads day-ahead prices and day-ahead forecasts for a bidding zone and shapes
them into the CSV layout that ``epftoolbox.data.read_data`` expects.

The package is deliberately self-contained: it does not import or modify anything
in the vendored ``epftoolbox`` tree.
"""

from .areas import AREAS, Area, lookup
from .client import TransparencyClient, TransparencyError, load_token

__all__ = [
    "AREAS",
    "Area",
    "lookup",
    "TransparencyClient",
    "TransparencyError",
    "load_token",
]
