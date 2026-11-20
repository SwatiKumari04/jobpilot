"""Where jobs come from. Each source returns a list of JobIn."""
from . import remotive, synthetic

__all__ = ["remotive", "synthetic"]
