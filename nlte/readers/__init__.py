"""Built-in grid readers.

Importing this package registers the formats a manifest may name.  Third-party
readers can register themselves through the ``nlte.readers`` entry-point group.
"""

from . import (amarsi_fe_ffnn, breidablik, ffnn, idl_sav,  # noqa: F401
               inasan_lst, tabular)

__all__ = ["tabular", "idl_sav", "ffnn", "breidablik", "inasan_lst",
           "amarsi_fe_ffnn"]
