"""Discovery of grid sources from declarative manifests.

Every table -- including the eight that ship with this package -- is described by
a TOML manifest sitting next to its data file.  Adding a tabular grid therefore
means writing a manifest, not writing code; only genuinely new binary formats
need a reader, registered with :func:`register_reader`.

Manifests are globbed from the grid root (``$NLTE_GRIDS``, else ``grids/`` beside
the package).  Loading is lazy and cached: a query for magnesium never touches
the 263 MB carbon tables.
"""

from __future__ import annotations

import os
import tomllib
import warnings
from pathlib import Path

from .solar import normalize_element

#: name -> reader callable ``(manifest: dict, path: Path) -> list[GridSource]``
READERS: dict = {}


def register_reader(name: str):
    """Register a loader for ``format = "<name>"`` manifests."""

    def wrap(fn):
        READERS[name] = fn
        return fn

    return wrap


_READERS_LOADED = False


def ensure_readers() -> None:
    """Import the built-in readers and any plugins, once per process.

    Deferred because the readers import from this module; called by every entry
    point that can reach a reader, so a directly constructed :class:`Registry`
    behaves the same as the shared one.
    """
    global _READERS_LOADED
    if _READERS_LOADED:
        return
    _READERS_LOADED = True
    from . import readers  # noqa: F401  -- registers the built-in formats

    _load_entry_point_readers()


def _load_entry_point_readers() -> None:
    """Let third-party packages contribute readers via the 'nlte.readers' group."""
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python < 3.8
        return
    try:
        eps = entry_points(group="nlte.readers")
    except TypeError:  # pragma: no cover - older API
        eps = entry_points().get("nlte.readers", [])
    for ep in eps:
        try:
            READERS.setdefault(ep.name, ep.load())
        except Exception:  # pragma: no cover - a broken plugin must not break imports
            continue


def default_root() -> Path:
    """Where to look for grids and manifests."""
    env = os.environ.get("NLTE_GRIDS")
    if env:
        return Path(env).expanduser()
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / "grids", Path.cwd() / "grids"):
        if candidate.is_dir():
            return candidate
    return here.parent / "grids"


def cache_dir(root: Path) -> Path:
    """Directory for parsed-grid caches; readers use it for slow text tables."""
    d = Path(root) / ".cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


class Manifest:
    """A parsed manifest, plus the metadata needed to index it without loading."""

    def __init__(self, path: Path, data: dict):
        self.path = Path(path)
        self.data = data
        try:
            self.element = normalize_element(data["element"])
            self.format = data["format"]
            self.base_id = data["id"]
        except KeyError as exc:
            raise ValueError(f"{path}: manifest is missing required key {exc}") from None
        self.ion = int(data.get("ion", 1))

    @property
    def key(self) -> str:
        return f"{self.path}:{self.base_id}"

    @property
    def datafile(self) -> Path:
        name = self.data.get("file")
        if not name:
            return self.path
        # A manifest may point outside the repository at a large released
        # distribution rather than duplicating it here.
        return (self.path.parent / Path(name).expanduser()).resolve()

    def __repr__(self) -> str:
        return f"<Manifest {self.base_id} ({self.element} {self.ion}) at {self.path.name}>"


def expand_manifest(path: Path, data: dict) -> list:
    """Expand a manifest file into one :class:`Manifest` per table it describes.

    A ``[[tables]]`` array lets one file describe a family of tables that share
    everything but a few keys -- the 22 Amarsi 2020 CSVs differ only by file
    name, element and whether scattering was included.  Each entry overrides the
    top-level keys.
    """
    tables = data.get("tables")
    if not tables:
        return [Manifest(path, data)]
    base = {k: v for k, v in data.items() if k != "tables"}
    return [Manifest(path, {**base, **entry}) for entry in tables]


class Registry:
    """Index of available sources; loads each manifest at most once."""

    def __init__(self, root=None):
        ensure_readers()
        self.root = Path(root) if root is not None else default_root()
        self._manifests: list | None = None
        self._loaded: dict = {}       # manifest path -> list[GridSource]
        self._extra: dict = {}        # id -> GridSource registered in memory
        self.missing: dict = {}       # manifest key -> why its data could not be read

    # -- discovery -------------------------------------------------------

    @property
    def manifests(self) -> list:
        if self._manifests is None:
            found = []
            if self.root.is_dir():
                for p in sorted(self.root.rglob("*.toml")):
                    if ".cache" in p.parts:
                        continue
                    with open(p, "rb") as fh:
                        found.extend(expand_manifest(p, tomllib.load(fh)))
            self._manifests = found
        return self._manifests

    def elements(self) -> list:
        """Elements covered, without loading any grid."""
        seen = {m.element for m in self.manifests}
        seen.update(s.element for s in self._extra.values())
        return sorted(seen)

    def refresh(self) -> None:
        """Forget discovery and cached loads (call after adding a manifest)."""
        self._manifests = None
        self._loaded.clear()

    # -- loading ---------------------------------------------------------

    def _load(self, manifest: Manifest) -> list:
        key = manifest.key
        if key not in self._loaded:
            try:
                reader = READERS[manifest.format]
            except KeyError:
                raise KeyError(
                    f"{manifest.path.name}: no reader for format {manifest.format!r}; "
                    f"known formats are {sorted(READERS)}"
                ) from None
            try:
                sources = reader(manifest.data, manifest.datafile)
            except FileNotFoundError as exc:
                # A manifest may point outside the repository at a large released
                # distribution.  If that data is not on this machine, skip the
                # source loudly rather than breaking every query for the element.
                self.missing[key] = str(exc)
                warnings.warn(
                    f"{manifest.base_id}: data not found ({exc}); skipping this source",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._loaded[key] = []
                return self._loaded[key]
            for s in sources:
                s.path = str(manifest.datafile)
            self._loaded[key] = list(sources)
        return self._loaded[key]

    def for_element(self, element: str, ion: int | None = None) -> list:
        """All sources for an element, loading only the manifests that match."""
        sym = normalize_element(element)
        out = []
        for m in self.manifests:
            if m.element != sym or (ion is not None and m.ion != ion):
                continue
            out.extend(self._load(m))
        out.extend(
            s for s in self._extra.values()
            if s.element == sym and (ion is None or s.ion == ion)
        )
        return out

    def get(self, source_id: str):
        if source_id in self._extra:
            return self._extra[source_id]
        for m in self.manifests:
            if source_id == m.base_id or source_id.startswith(m.base_id + "_"):
                for s in self._load(m):
                    if s.id == source_id:
                        return s
        for m in self.manifests:            # fall back to an exhaustive search
            for s in self._load(m):
                if s.id == source_id:
                    return s
        raise KeyError(f"unknown source {source_id!r}")

    def all(self) -> list:
        """Load and return every source (slow -- mainly for validation)."""
        out = []
        for m in self.manifests:
            out.extend(self._load(m))
        out.extend(self._extra.values())
        return out

    def add(self, source) -> None:
        """Register an in-memory source, e.g. one built by a script."""
        self._extra[source.id] = source


_REGISTRY: Registry | None = None


def get_registry() -> Registry:
    """The process-wide registry."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = Registry()
    return _REGISTRY


def set_grid_root(root) -> Registry:
    """Point the registry at a different grid directory."""
    global _REGISTRY
    _REGISTRY = Registry(root)
    return _REGISTRY


def register_source(source) -> None:
    """Add a :class:`~nlte.source.GridSource` built in memory."""
    get_registry().add(source)
