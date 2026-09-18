"""The computations of the paper, section by section.

Each ``run_*`` function follows the corresponding original notebook:

* ``compute_all_diagrams`` / ``compute_all_distances``: "creation des
  dictionnaires" and ``computation.py`` (decolored diagrams ``X_decolor`` and
  Wasserstein distances, cached in ``cache_dir``);
* ``run_anomalies``: "anomalies detection" (section 3);
* ``run_gdi``: ``TDA_All``, ``Silhouette``, ``TDA_All-trunk`` and
  ``Silhouette-Trunks`` (section 4);
* ``run_morphotypes``: "clustering silhouette" (section 5);
* ``run_figures``: ``Homology_Plot`` and the illustrations of section 2.

Tables, curves and figures are written to ``results_dir``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from . import config as cfg
from .anomalies import anomaly_percentages, best_range, solos
from .clustering import KMeans, KMedoids, make_dict_clusters
from .data import all_scans, load_mesh, load_points, scan_path, short_name
from .geometry import body_to_trunk, normalize_scan, prepare_points, scan_height
from .indices import DB_index, GDI, Dunn, cluster_table, local_optima, merge_heights, silhouette_index
from .parallel import default_jobs, parallel_map, shared
from .persistence import DiagramSet, decolored_diag, point_set
from .plotting import (plot_anomaly_percentages, plot_dendrogram, plot_diagram_and_barcode,
                       plot_gdi_body_vs_trunk, plot_gdi_curves, plot_homology_classes, plot_indices, plot_trunk,
                       render_meshes)
from .representatives import find_homologies
from .silhouettes import silhouette_vectors
from .wasserstein import combine, decolored_dist, distance_matrix


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _scan_diagrams(path) -> list[np.ndarray]:
    setting = shared()
    return decolored_diag(prepare_points(load_points(path), setting), setting.min_persistence)


class Workspace:
    """Folders of the data, of the cached intermediate results and of the results."""

    def __init__(self, data_dir="data/raw", cache_dir="data/cache", results_dir="results", n_jobs=None):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.results_dir = Path(results_dir)
        self.n_jobs = n_jobs or default_jobs()

    def all_scans(self, sex: str) -> list[str]:
        return all_scans(sex, self.data_dir)

    def has_meshes(self, sex: str) -> bool:
        folder = self.data_dir / sex
        return folder.is_dir() and any(folder.glob("*.obj"))

    def mesh(self, sex: str, name: str):
        return load_mesh(scan_path(name, sex, self.data_dir))

    def scan(self, sex: str, name: str) -> np.ndarray:
        return load_points(scan_path(name, sex, self.data_dir))

    def output(self, *parts) -> Path:
        path = self.results_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # --- cached intermediate results ---------------------------------------

    def X_decolor(self, setting: cfg.DiagramSetting, sex: str) -> DiagramSet:
        """Decolored diagrams of every scan of one sex (cached).

        Without the meshes, the diagrams shipped in the cache are used as they are.
        """
        path = self.cache_dir / "diagrams" / f"{setting.name}_{sex}.npz"
        meta = path.with_suffix(".json")
        names = self.all_scans(sex) if self.has_meshes(sex) else None
        if path.exists() and meta.exists():
            saved = json.loads(meta.read_text())
            if saved["setting"] == asdict(setting) and names in (None, saved["names"]):
                return DiagramSet.load(path)
        if names is None:
            raise FileNotFoundError(f"no mesh in {self.data_dir / sex} and no diagrams for {setting.name} "
                                    f"in {path.parent}; see README.md to download the data")
        expected = {"setting": asdict(setting), "names": names}
        paths = [scan_path(n, sex, self.data_dir) for n in names]
        diagrams = parallel_map(_scan_diagrams, paths, n_jobs=self.n_jobs, shared=setting, chunksize=4,
                                desc=f"diagrams {setting.name} {sex}")
        result = DiagramSet(names, diagrams)
        result.save(path)
        meta.write_text(json.dumps(expected))
        return result

    def X_decolor_all(self, setting: cfg.DiagramSetting) -> tuple[DiagramSet, int]:
        """Males followed by females (section 4), and the number of males."""
        male, female = (self.X_decolor(setting, sex) for sex in cfg.SEXES)
        return male + female, len(male)

    def distances(self, key: str, X_decolor: DiagramSet, dims) -> dict[int, np.ndarray]:
        """Condensed matrices of the Wasserstein distances of each degree (cached)."""
        out = {}
        for d in dims:
            path = self.cache_dir / "distances" / f"{key}_H{d}.npy"
            meta = path.with_suffix(".json")
            expected = {"names": X_decolor.names, "order": cfg.WASSERSTEIN_ORDER,
                        "internal_p": cfg.WASSERSTEIN_INTERNAL_P}
            if path.exists() and meta.exists() and json.loads(meta.read_text()) == expected:
                out[d] = np.load(path)
                continue
            out[d] = distance_matrix(X_decolor.degree(d), cfg.WASSERSTEIN_ORDER, cfg.WASSERSTEIN_INTERNAL_P,
                                     n_jobs=self.n_jobs, desc=f"distances {key} H{d}")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, out[d])
            meta.write_text(json.dumps(expected))
        return out


def compute_all_diagrams(ws: Workspace) -> None:
    for setting in (cfg.BODY_M, cfg.BODY_CM, cfg.TRUNK):
        for sex in cfg.SEXES:
            ws.X_decolor(setting, sex)


def compute_all_distances(ws: Workspace) -> None:
    for sex in cfg.SEXES:
        ws.distances(f"body_m_{sex}", ws.X_decolor(cfg.BODY_M, sex), cfg.ANOMALY_DEGREES)
    for part, setting in (("body", cfg.BODY_CM), ("trunk", cfg.TRUNK)):
        ws.distances(f"{part}_all", ws.X_decolor_all(setting)[0], cfg.GDI_DEGREES)


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _render(ws: Workspace, sex: str, names, path: Path, titles=None, **kwargs) -> None:
    """Render meshes of scans, if the meshes are available."""
    if not ws.has_meshes(sex):
        log(f"  {path.name} skipped: no mesh in {ws.data_dir / sex}")
        return
    titles = titles or [short_name(n) for n in names]
    render_meshes([(*ws.mesh(sex, n), t) for n, t in zip(names, titles)], path, **kwargs)


# --- Section 3: anomaly detection -------------------------------------------

def run_anomalies(ws: Workspace) -> dict:
    summary = {}
    for sex in cfg.SEXES:
        log(f"[anomalies] {sex}")
        X_decolor = ws.X_decolor(cfg.BODY_M, sex)
        distances = combine(ws.distances(f"body_m_{sex}", X_decolor, cfg.ANOMALY_DEGREES), cfg.ANOMALY_DEGREES)
        Z = linkage(distances, method="complete")
        anomalies = [X_decolor.names.index(a) for a in cfg.KNOWN_ANOMALIES[sex]]
        percentages = anomaly_percentages(Z, anomalies)
        start, end = best_range(percentages)
        row = percentages.set_index("nb_clus").loc[start]
        isolated = [X_decolor.names[i] for i in solos(Z, start)]
        percentages.to_csv(ws.output("anomalies", f"{sex}_percentages.csv"), index=False)
        summary[sex] = {
            "known": list(cfg.KNOWN_ANOMALIES[sex]),
            "best_range": [start, end],
            "isolated_are_anomalies_pct": float(row["percent1"]),
            "anomalies_detected_pct": float(row["percent2"]),
            "criterion_pct": float(row["percent3"]),
            "solos_at_range_start": isolated,
            "detected_at_range_start": sorted(set(isolated) & set(cfg.KNOWN_ANOMALIES[sex])),
        }
        figs = {"male": ("fig07", "fig08"), "female": ("fig11", "fig12")}[sex]
        plot_dendrogram(Z, cfg.ANOMALY_DENDROGRAM_LEAVES[sex],
                        ws.output("figures", f"{figs[0]}_anomalies_dendrogram_{sex}.png"),
                        labels=[short_name(n) for n in X_decolor.names], ylabel="Wasserstein distance")
        plot_anomaly_percentages(percentages, (start, end), ws.output("figures", f"anomalies_criterion_{sex}.png"))
        _render(ws, sex, summary[sex]["detected_at_range_start"],
                ws.output("figures", f"{figs[1]}_anomalies_{sex}.png"))
    _write_json(ws.output("anomalies", "summary.json"), summary)
    return summary


# --- Section 4: gender discrimination index ------------------------------------

def _GDI_list(clusters_for, male_nb_scans: int) -> list[float]:
    """GDI for nb_clus = 2..30, ``clusters_for(nb_clus)`` giving the labels of the clustering."""
    return [GDI(make_dict_clusters(clusters_for(nb_clus)), male_nb_scans) for nb_clus in cfg.GDI_N_CLUSTERS]


def run_gdi(ws: Workspace) -> dict:
    rows = []

    def add(representation, method, GDI_list):
        rows.extend({"representation": representation, "method": method, "k": k, "gdi": g}
                    for k, g in zip(cfg.GDI_N_CLUSTERS, GDI_list))
        log(f"[gdi] {representation:18s} {method:9s} mean GDI {np.mean(GDI_list):.4f}")

    for part, setting, sil_setting in (("body", cfg.BODY_CM, cfg.GDI_BODY_SILHOUETTE),
                                       ("trunk", cfg.TRUNK, cfg.GDI_TRUNK_SILHOUETTE)):
        X_decolor, male_nb_scans = ws.X_decolor_all(setting)

        # persistence diagrams with the Wasserstein distance (Figures 13 and 16)
        distances = combine(ws.distances(f"{part}_all", X_decolor, cfg.GDI_DEGREES), cfg.GDI_DEGREES)
        for method in ("complete", "ward"):
            Z = linkage(distances, method=method)
            add(f"wasserstein_{part}", method, _GDI_list(lambda k: fcluster(Z, k, criterion="maxclust"),
                                                          male_nb_scans))
        distance = squareform(distances)
        add(f"wasserstein_{part}", "kmedoids", _GDI_list(
            lambda k: KMedoids(n_clusters=k, method="pam", metric="precomputed", init="k-medoids++",
                               random_state=cfg.RANDOM_STATE).fit(distance).labels_, male_nb_scans))

        # persistence silhouettes with the Euclidean distance (Figures 14 and 17)
        if part == "body":
            X_decolor = X_decolor.filtered(cfg.BODY_SILHOUETTE_MIN_PERSISTENCE)
        sh = silhouette_vectors(X_decolor, sil_setting)
        Z = linkage(sh, method="ward")
        add(f"silhouette_{part}", "ward", _GDI_list(lambda k: fcluster(Z, k, criterion="maxclust"), male_nb_scans))
        add(f"silhouette_{part}", "kmeans", _GDI_list(
            lambda k: KMeans(k, random_state=cfg.RANDOM_STATE).fit_predict(sh), male_nb_scans))
        add(f"silhouette_{part}", "kmedoids", _GDI_list(
            lambda k: KMedoids(n_clusters=k, method="pam", init="k-medoids++",
                               random_state=cfg.RANDOM_STATE).fit(sh).labels_, male_nb_scans))

    curves = pd.DataFrame(rows)
    curves.to_csv(ws.output("gdi", "gdi_curves.csv"), index=False)
    means = curves.groupby(["representation", "method"], sort=False)["gdi"].mean()
    summary = {
        "table1_mean_gdi_wasserstein": {part: {m: float(means[f"wasserstein_{part}", m])
                                               for m in ("complete", "ward", "kmedoids")}
                                        for part in ("body", "trunk")},
        "table2_mean_gdi_silhouette": {part: {m: float(means[f"silhouette_{part}", m])
                                              for m in ("ward", "kmeans", "kmedoids")}
                                       for part in ("body", "trunk")},
    }
    _write_json(ws.output("gdi", "summary.json"), summary)
    plot_gdi_curves(curves, "wasserstein_body", ws.output("figures", "fig13_gdi_wasserstein.png"))
    plot_gdi_curves(curves, "silhouette_body", ws.output("figures", "fig14_gdi_silhouette.png"))
    plot_gdi_body_vs_trunk(curves, "wasserstein", ws.output("figures", "fig16_gdi_wasserstein_body_vs_trunk.png"))
    plot_gdi_body_vs_trunk(curves, "silhouette", ws.output("figures", "fig17_gdi_silhouette_body_vs_trunk.png"))
    return summary


# --- Section 5: morphotypes ------------------------------------------------------

def run_morphotypes(ws: Workspace) -> dict:
    summary = {}
    for sex in cfg.SEXES:
        log(f"[morphotypes] {sex}")
        # the known scan anomalies are removed
        X_decolor = ws.X_decolor(cfg.BODY_M, sex).without(cfg.KNOWN_ANOMALIES[sex])
        sh = silhouette_vectors(X_decolor, cfg.MORPHOTYPE_SILHOUETTE)
        D = squareform(pdist(sh))
        Z = linkage(sh, method="ward", metric="euclidean")

        # clustering quality indices for each number of clusters
        ks = [k for k in cfg.INDEX_N_CLUSTERS if k < len(sh)]
        rows = []
        for nb_clus, height in zip(ks, merge_heights(Z, ks)):
            dict_clusters = make_dict_clusters(fcluster(Z, nb_clus, criterion="maxclust"))
            rows.append({"n_clusters": nb_clus, "merge_height": height,
                         "davies_bouldin": DB_index(dict_clusters, sh),
                         "silhouette": silhouette_index(dict_clusters, D),
                         "dunn": Dunn(dict_clusters, sh)})
        indices = pd.DataFrame(rows)
        indices.to_csv(ws.output("morphotypes", f"{sex}_indices.csv"), index=False)

        # the clustering retained in the paper
        nb_clus = cfg.MORPHOTYPE_N_CLUSTERS[sex]
        clusters = fcluster(Z, nb_clus, criterion="maxclust")
        dict_clusters = make_dict_clusters(clusters)
        pd.DataFrame({"scan": X_decolor.names, "cluster": [f"C{c}" for c in clusters]}).to_csv(
            ws.output("morphotypes", f"{sex}_clusters.csv"), index=False)
        table = cluster_table(dict_clusters, sh, X_decolor.names)
        table.to_csv(ws.output("morphotypes", f"{sex}_table.csv"), index=False)
        window = indices[indices["n_clusters"].between(*cfg.INDEX_OPTIMA_WINDOW)]
        summary[sex] = {
            "n_scans": len(X_decolor),
            "table": table.to_dict(orient="list"),
            "cluster_1_members": [X_decolor.names[i] for i in dict_clusters[1]],
            # k = 2 only isolates the most atypical individuals; the paper reads local optima
            "local_optima": {
                index: local_optima(window["n_clusters"].tolist(), window[index].tolist(), maximize)
                for index, maximize in (("davies_bouldin", False), ("silhouette", True), ("dunn", True))
            },
        }
        figs = {"male": ("fig18", "fig20"), "female": ("fig21", "fig22")}[sex]
        plot_dendrogram(Z, cfg.MORPHOTYPE_DENDROGRAM_LEAVES,
                        ws.output("figures", f"{figs[0]}_ward_dendrogram_{sex}.png"))
        plot_indices(indices, nb_clus, ws.output("figures", f"morphotypes_indices_{sex}.png"))
        medoids = table["medoid"].tolist()
        titles = [f"({c}) {short_name(m)}" for c, m in zip(table["cluster"], medoids)]
        _render(ws, sex, medoids, ws.output("figures", f"{figs[1]}_medoids_{sex}.png"), titles, ncols=4)
        if sex == "male":
            _render(ws, sex, summary[sex]["cluster_1_members"], ws.output("figures", "fig19_male_cluster_C1.png"))
    _write_json(ws.output("morphotypes", "summary.json"), summary)
    return summary


# --- Illustrations (section 2) ---------------------------------------------------

def _homologies(scan: np.ndarray, setting: cfg.DiagramSetting) -> tuple[point_set, list]:
    PS = point_set(scan)
    PS.normalize(setting.target_height, setting.height_decimals)
    PS.Persistence(setting.min_persistence)
    return PS, find_homologies(PS)


def run_figures(ws: Workspace) -> dict:
    if not ws.has_meshes("male"):
        log(f"[figures] skipped: these figures are computed from the meshes, not found in {ws.data_dir}")
        return {}
    summary = {}
    sex, name = cfg.EXAMPLE_SCAN
    log(f"[figures] {name}")
    _render(ws, sex, [name], ws.output("figures", f"fig01_mesh_{short_name(name)}.png"))
    scan = ws.scan(sex, name)
    PS, homologies = _homologies(scan, cfg.BODY_CM)
    plot_diagram_and_barcode(homologies, ws.output("figures", f"fig02_diagram_barcode_{short_name(name)}.png"))
    plot_homology_classes(PS.points, [c for c in homologies if c.degree > 0],
                          ws.output("figures", f"fig05_homologies_{short_name(name)}.png"))
    summary["example_homologies"] = [{"number": c.number, "degree": c.degree, "degree_number": c.degree_number,
                                      "birth": c.birth, "death": c.death} for c in homologies]
    plot_trunk(PS.points, body_to_trunk(scan), ws.output("figures", f"fig15_trunk_{short_name(name)}.png"))

    sex, name = cfg.ANOMALY_EXAMPLE_SCAN
    log(f"[figures] {name}")
    PS, homologies = _homologies(ws.scan(sex, name), cfg.BODY_CM)
    h2 = [c for c in homologies if c.degree == 2 and c.degree_number in cfg.ANOMALY_EXAMPLE_H2]
    plot_diagram_and_barcode(homologies, ws.output("figures", f"fig09_diagram_barcode_{short_name(name)}.png"),
                             highlight=[c.number for c in h2])
    plot_homology_classes(PS.points, sorted(h2, key=lambda c: c.degree_number),
                          ws.output("figures", f"fig10_homologies_{short_name(name)}.png"), ncols=3, by_degree=True)
    summary["anomaly_example_h2"] = [{"degree_number": c.degree_number, "barcode_number": c.number,
                                      "birth": c.birth, "death": c.death} for c in h2]

    summary["normalisation_example"] = normalisation_example(ws)
    _render(ws, "male", [n for _, n in cfg.NORMALISATION_EXAMPLE],
            ws.output("figures", "fig06_normalisation_example.png"),
            [f"({t}) {short_name(n)}" for t, (_, n) in zip("abc", cfg.NORMALISATION_EXAMPLE)], hide_head=None)
    _write_json(ws.output("figures", "summary.json"), summary)
    return summary


def normalisation_example(ws: Workspace) -> dict:
    """Closest pair among three individuals before and after normalisation (Figure 6)."""
    scans = {n: ws.scan(s, n) for s, n in cfg.NORMALISATION_EXAMPLE}
    out = {"heights_m": {n: round(scan_height(scan), 3) for n, scan in scans.items()}}
    variants = {
        "before": {n: decolored_diag(scan, cfg.BODY_M.min_persistence) for n, scan in scans.items()},
        "after": {n: decolored_diag(normalize_scan(scan, cfg.BODY_M.target_height, cfg.BODY_M.height_decimals),
                                    cfg.BODY_M.min_persistence) for n, scan in scans.items()},
    }
    names = list(scans)
    for when, X in variants.items():
        pairs = {}
        for i in range(3):
            for j in range(i + 1, 3):
                per_degree = decolored_dist(X[names[i]], X[names[j]], dim=list(cfg.ANOMALY_DEGREES))
                pairs[f"{names[i]}-{names[j]}"] = float(np.sqrt(np.sum(np.square(per_degree))))
        out[f"distances_{when}"] = pairs
        out[f"closest_{when}"] = min(pairs, key=pairs.get)
    return out
