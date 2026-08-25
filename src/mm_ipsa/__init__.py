"""MM-IPSA Research: generación y evaluación reproducible de escenarios."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mm-ipsa-research")
except PackageNotFoundError:
    __version__ = "0+unknown"
