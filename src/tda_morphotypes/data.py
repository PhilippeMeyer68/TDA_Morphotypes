"""Access to the SPRING meshes (Yang et al., 2014), derived from CAESAR.

Expected layout: ``<data_dir>/male/SPRINGxxxx.obj`` and
``<data_dir>/female/SPRINGxxxx.obj`` (1517 and 1531 meshes). Scans are always
handled in lexicographic order of their names, males before females when both
sexes are combined.
"""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np

from .config import SEXES

DATA_DIR = "data/raw"


def all_scans(sex: str, data_dir: str | Path = DATA_DIR) -> list[str]:
    """Sorted names (``SPRINGxxxx``) of the scans of one sex."""
    if sex not in SEXES:
        raise ValueError(f"sex must be one of {SEXES}, got {sex!r}")
    folder = Path(data_dir) / sex
    names = sorted(p.stem for p in folder.glob("*.obj"))
    if not names:
        raise FileNotFoundError(f"no .obj file in {folder}; see README.md to download the data")
    return names


def scan_path(name: str, sex: str, data_dir: str | Path = DATA_DIR) -> Path:
    return Path(data_dir) / sex / f"{name}.obj"


def load_points(path: str | Path) -> np.ndarray:
    """Vertices of a mesh, as a float64 array of shape (n_vertices, 3), in metres."""
    return np.asarray(meshio.read(path).points, dtype=np.float64)


def load_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangular faces of a mesh."""
    mesh = meshio.read(path)
    return np.asarray(mesh.points, dtype=np.float64), mesh.cells_dict["triangle"]


def scan_from_index(i: int, sex: str, data_dir: str | Path = DATA_DIR) -> np.ndarray:
    """Point cloud of the i-th scan of one sex."""
    return load_points(scan_path(all_scans(sex, data_dir)[i], sex, data_dir))


def short_name(name: str) -> str:
    """``SPRING0013`` -> ``S0013``, the notation of the paper."""
    return "S" + name.removeprefix("SPRING")
