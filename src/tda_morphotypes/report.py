"""Comparison of the results with the paper and summary page ``results/README.md``."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_DIR = resources.files("tda_morphotypes") / "reference"
GDI_TOLERANCE = 1e-12


def _decimals(value) -> int:
    text = repr(value)
    return len(text.split(".")[1]) if "." in text else 0


def _same(expected, obtained) -> bool:
    """Obtained value rounded like the value printed in the paper."""
    if expected is None:
        return True
    if isinstance(expected, str):
        return expected == obtained
    return round(float(obtained), _decimals(expected)) == expected


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


class Checker:
    def __init__(self):
        self.rows = []

    def add(self, section, item, expected, obtained, ok, note=""):
        self.rows.append({"section": section, "item": item, "paper": expected, "obtained": obtained,
                          "status": "ok" if ok else ("known discrepancy" if note else "MISMATCH"), "note": note})

    @property
    def failures(self) -> int:
        return sum(r["status"] == "MISMATCH" for r in self.rows)


def compare(results_dir: str | Path) -> Checker:
    results_dir = Path(results_dir)
    paper = json.loads(REFERENCE_DIR.joinpath("paper_values.json").read_text())
    known = paper["_known_discrepancies"]
    check = Checker()

    anomalies = _load(results_dir / "anomalies" / "summary.json")
    if anomalies:
        for sex, ref in paper["anomalies"].items():
            got = anomalies[sex]
            for key in ("best_range", "isolated_are_anomalies_pct", "anomalies_detected_pct", "criterion_pct"):
                check.add("3 anomalies", f"{sex} {key}", ref[key], got[key], ref[key] == got[key])
            check.add("3 anomalies", f"{sex} detected", " ".join(ref["detected_at_range_start"]),
                      " ".join(got["detected_at_range_start"]),
                      sorted(ref["detected_at_range_start"]) == got["detected_at_range_start"])

    gdi = _load(results_dir / "gdi" / "summary.json")
    if gdi:
        for table in ("table1_mean_gdi_wasserstein", "table2_mean_gdi_silhouette"):
            for part, methods in paper[table].items():
                for method, ref in methods.items():
                    got = gdi[table][part][method]
                    check.add("4 GDI", f"{table.split('_')[0]} {part} {method}", ref, round(got, 4), _same(ref, got))
        curves = pd.read_csv(results_dir / "gdi" / "gdi_curves.csv")
        with REFERENCE_DIR.joinpath("gdi_curves_2022.csv").open() as fh:
            ref_curves = pd.read_csv(fh)
        merged = ref_curves.merge(curves, on=["representation", "method", "k"], suffixes=("_2022", ""))
        for (rep, method), group in merged.groupby(["representation", "method"], sort=False):
            diff = float(np.abs(group["gdi"] - group["gdi_2022"]).max())
            check.add("4 GDI", f"curve {rep} {method} (29 values)", "2022 run", f"max diff {diff:.1e}",
                      diff <= GDI_TOLERANCE and len(group) == 29)

    morpho = _load(results_dir / "morphotypes" / "summary.json")
    if morpho:
        for sex, key in (("male", "table3_male_morphotypes"), ("female", "table4_female_morphotypes")):
            ref, got = paper[key], morpho[sex]["table"]
            for column in ("size", "proportion_pct", "mean_distance", "diameter", "distance_to_mean",
                           "distance_mean_medoid", "medoid"):
                for c, (r, g) in enumerate(zip(ref[column], got[column])):
                    note = known.get(f"{key}.{column}[{c}]", "")
                    shown = g if isinstance(g, str) else round(g, 2)
                    check.add("5 morphotypes", f"{sex} C{c + 1} {column}", r, shown, _same(r, g), note)
            for index, k in ref["suggested_by"].items():
                optima = morpho[sex]["local_optima"][index]
                check.add("5 morphotypes", f"{sex} {index} has a local optimum at k = {k}", k,
                          " ".join(map(str, optima)), k in optima)
            if "cluster_1_members" in ref:
                check.add("5 morphotypes", f"{sex} C1 members", " ".join(ref["cluster_1_members"]),
                          " ".join(morpho[sex]["cluster_1_members"]),
                          ref["cluster_1_members"] == morpho[sex]["cluster_1_members"])
    return check


def _markdown_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_summary(results_dir: str | Path) -> Checker:
    """Write ``results/README.md`` and ``results/comparison_with_paper.csv``."""
    results_dir = Path(results_dir)
    check = compare(results_dir)
    table = pd.DataFrame(check.rows)
    if not table.empty:
        table.to_csv(results_dir / "comparison_with_paper.csv", index=False)
    parts = ["# Results", "",
             "Generated by `tda-morphotypes report`. Every value is compared with the published paper",
             "(rounded as printed there) and the GDI curves with the outputs of the original 2022 run",
             "(`src/tda_morphotypes/reference/`).", ""]
    if not table.empty:
        n_ok = (table["status"] == "ok").sum()
        parts += [f"**{n_ok} / {len(table)} values identical to the paper**, "
                  f"{(table['status'] == 'known discrepancy').sum()} known discrepancy, "
                  f"{check.failures} mismatch.", ""]

    anomalies = _load(results_dir / "anomalies" / "summary.json")
    if anomalies:
        parts += ["## Section 3 - Anomaly detection", ""]
        rows = [{"sex": sex, "best range": s["best_range"], "isolated that are anomalies (%)":
                 s["isolated_are_anomalies_pct"], "anomalies isolated (%)": s["anomalies_detected_pct"],
                 "detected": ", ".join(s["detected_at_range_start"])} for sex, s in anomalies.items()]
        parts += [_markdown_table(pd.DataFrame(rows)), "",
                  "![](figures/fig07_anomalies_dendrogram_male.png)", "",
                  "![](figures/fig11_anomalies_dendrogram_female.png)", ""]

    gdi = _load(results_dir / "gdi" / "summary.json")
    if gdi:
        parts += ["## Section 4 - Gender discrimination index", ""]
        for key, title in (("table1_mean_gdi_wasserstein", "Table 1 - mean GDI, persistence diagrams"),
                           ("table2_mean_gdi_silhouette", "Table 2 - mean GDI, persistence silhouettes")):
            df = pd.DataFrame(gdi[key]).T.round(3).reset_index(names="")
            parts += [f"**{title}**", "", _markdown_table(df), ""]
        parts += ["![](figures/fig13_gdi_wasserstein.png)", "", "![](figures/fig14_gdi_silhouette.png)", "",
                  "![](figures/fig16_gdi_wasserstein_body_vs_trunk.png)", "",
                  "![](figures/fig17_gdi_silhouette_body_vs_trunk.png)", ""]

    morpho = _load(results_dir / "morphotypes" / "summary.json")
    if morpho:
        parts += ["## Section 5 - Morphotypes", ""]
        for sex, number, fig in (("male", 3, "fig20_medoids_male.png"), ("female", 4, "fig22_medoids_female.png")):
            df = pd.DataFrame(morpho[sex]["table"]).round(1)
            optima = morpho[sex]["local_optima"]
            parts += [f"**Table {number} - {sex} morphotypes** ({morpho[sex]['n_scans']} scans)", "",
                      _markdown_table(df), "",
                      "Local optima of the indices for 3 to 20 clusters: "
                      + "; ".join(f"{name} {', '.join(map(str, optima[key]))}" for key, name in
                                  (("davies_bouldin", "Davies-Bouldin"), ("silhouette", "Silhouette"),
                                   ("dunn", "Dunn"))) + ".", "",
                      f"![]({'figures/' + fig})", ""]

    if not table.empty:
        issues = table[table["status"] != "ok"]
        parts += ["## Differences with the paper", ""]
        parts += [_markdown_table(issues[["item", "paper", "obtained", "status", "note"]])
                  if len(issues) else "None.", ""]
    (results_dir / "README.md").write_text("\n".join(parts), encoding="utf-8")
    return check
