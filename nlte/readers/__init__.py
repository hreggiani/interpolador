"""Built-in grid readers.

Importing this package registers the formats a manifest may name.  Third-party
readers can register themselves through the ``nlte.readers`` entry-point group.
"""

from . import idl_sav, tabular  # noqa: F401

__all__ = ["tabular", "idl_sav"]
