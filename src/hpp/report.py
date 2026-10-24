"""Render the monitoring summary as a single self-contained HTML report (charts embedded as PNG).

Colour roles follow one rule each: a single blue for magnitude (one sequential hue),
grey for the training reference, and the reserved status colours only for alert
levels - always paired with the level's name, never colour alone.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hpp.config import Config

BLUE = "#2a78d6"
REF_GREY = "#b9b8b0"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
STATUS = {"OK": "#0ca30c", "WARNING": "#fab219", "CRITICAL": "#d03b3b", "INFO": "#898781"}
STATUS_TEXT = {"OK": "#ffffff", "WARNING": "#0b0b0b", "CRITICAL": "#ffffff", "INFO": "#ffffff"}
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _plt():
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titlesize": 10,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    })
    return plt


def _png(fig: Any, path: Path) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _clean_axes(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def chart_psi_heatmap(summary: dict, cfg: Config, out: Path) -> str:
    plt = _plt()
    from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: PLC0415

    batches = [r["batch"] for r in summary["batches"]]
    feats = [row["feature"] for row in summary["batches"][0]["drift"]]
    rows = feats + ["Price (target)", "Predictions"]
    mat = np.full((len(rows), len(batches)), np.nan)
    for j, r in enumerate(summary["batches"]):
        lookup = {row["feature"]: row["psi"] for row in r["drift"]}
        for i, f in enumerate(feats):
            mat[i, j] = lookup.get(f, np.nan)
        mat[len(feats), j] = r["target_psi"] if r["target_psi"] is not None else np.nan
        mat[len(feats) + 1, j] = r["prediction_psi"] if r["prediction_psi"] is not None else np.nan
    bounds = [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 10]
    cmap = ListedColormap(SEQ)
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(batches), 0.42 * len(rows) + 1.2))
    ax.imshow(np.nan_to_num(mat), cmap=cmap, norm=BoundaryNorm(bounds, cmap.N), aspect="auto")
    for i in range(len(rows)):
        for j in range(len(batches)):
            v = mat[i, j]
            if np.isnan(v):
                continue
            mark = "  ▲▲" if v >= cfg.monitoring.psi_critical else ("  ▲" if v >= cfg.monitoring.psi_warning else "")
            ax.text(j, i, f"{v:.3f}{mark}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if v >= 0.15 else INK)
    ax.set_xticks(range(len(batches)), [f"Batch {b}" for b in batches])
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    ax.set_xticks(np.arange(-.5, len(batches), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Population Stability Index vs training reference  (▲ ≥ {cfg.monitoring.psi_warning:g} warning, "
                 f"▲▲ ≥ {cfg.monitoring.psi_critical:g} critical)", loc="left")
    return _png(fig, out / "psi_heatmap.png")


def chart_performance(summary: dict, cfg: Config, out: Path) -> str:
    plt = _plt()
    res = [r for r in summary["batches"] if r["performance"]]
    labels = [f"Batch {r['batch']}" for r in res]
    ref = summary["reference"]["holdout_rmse"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    rmse = [r["performance"]["rmse"] for r in res]
    bars = ax.bar(labels, rmse, color=BLUE, width=0.55)
    for b, v in zip(bars, rmse):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.2f}", ha="center", va="bottom", color=INK, fontsize=9)
    for mult, name, ls in [(1.0, "holdout", "-"), (cfg.monitoring.rmse_ratio_warning, "warning", "--"),
                           (cfg.monitoring.rmse_ratio_critical, "critical", ":")]:
        ax.axhline(ref * mult, color=INK2, lw=1, ls=ls)
        ax.text(len(labels) - 0.5, ref * mult, f" {name} {ref * mult:.1f}", va="center", ha="left",
                color=INK2, fontsize=8)
    ax.set_ylabel("RMSE (lakhs)")
    ax.set_title("Performance drift: RMSE per batch", loc="left")
    _clean_axes(ax)
    ax = axes[1]
    bias = [r["performance"]["bias"] for r in res]
    bars = ax.bar(labels, bias, color=BLUE, width=0.55)
    for b, v in zip(bars, bias):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.2 if v >= 0 else -0.2), f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", color=INK, fontsize=9)
    mean_price = summary["reference"]["mean_price"]
    for pct, name, ls in [(cfg.monitoring.bias_warning_pct, "warning", "--"),
                          (cfg.monitoring.bias_critical_pct, "critical", ":")]:
        ax.axhline(pct * mean_price, color=INK2, lw=1, ls=ls)
        ax.text(len(labels) - 0.5, pct * mean_price, f" {name} +{pct * mean_price:.1f}", va="center",
                color=INK2, fontsize=8)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_ylabel("Mean(actual − predicted), lakhs")
    ax.set_title("Bias per batch (+ = model under-prices)", loc="left")
    _clean_axes(ax)
    fig.tight_layout()
    return _png(fig, out / "performance.png")


def chart_distributions(summary: dict, frames: dict[str, pd.DataFrame], ref: pd.DataFrame, cfg: Config,
                        out: Path, top_n: int = 3) -> str:
    plt = _plt()
    maxpsi: dict[str, float] = {}
    for r in summary["batches"]:
        for row in r["drift"]:
            maxpsi[row["feature"]] = max(maxpsi.get(row["feature"], 0), np.nan_to_num(row["psi"]))
    feats = sorted(maxpsi, key=maxpsi.get, reverse=True)[:top_n] + [cfg.data.target]
    batches = [r["batch"] for r in summary["batches"]]
    fig, axes = plt.subplots(len(feats), len(batches), figsize=(3.3 * len(batches), 2.3 * len(feats)),
                             squeeze=False)
    for i, f in enumerate(feats):
        for j, b in enumerate(batches):
            ax, cur = axes[i, j], frames[b]
            if f in cfg.data.categorical_features:
                levels = cfg.data.categorical_features[f]
                rf = ref[f].value_counts(normalize=True).reindex(levels, fill_value=0)
                cf = cur[f].value_counts(normalize=True).reindex(levels, fill_value=0)
                x = np.arange(len(levels))
                ax.bar(x - 0.2, rf, width=0.38, color=REF_GREY, label="training")
                ax.bar(x + 0.2, cf, width=0.38, color=BLUE, label="incoming batch")
                ax.set_xticks(x, levels, rotation=30, fontsize=7.5)
            else:
                rv = pd.to_numeric(ref[f], errors="coerce").dropna()
                cv = pd.to_numeric(cur[f], errors="coerce").dropna()
                edges = np.histogram_bin_edges(pd.concat([rv, cv]), bins=25)
                ax.hist(rv, bins=edges, density=True, color=REF_GREY, label="training")
                ax.hist(cv, bins=edges, density=True, histtype="step", lw=2, color=BLUE, label="incoming batch")
                ax.set_yticks([])
            psi = next((row["psi"] for row in next(r for r in summary["batches"] if r["batch"] == b)["drift"]
                        if row["feature"] == f), None)
            if f == cfg.data.target:
                psi = next(r for r in summary["batches"] if r["batch"] == b)["target_psi"]
            ax.set_title(f"{f} · batch {b}" + (f" · PSI {psi:.2f}" if psi is not None else ""), loc="left",
                         fontsize=9)
            _clean_axes(ax)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _png(fig, out / "distributions.png")


def chart_pred_vs_actual(summary: dict, preds: dict[str, pd.DataFrame], cfg: Config, out: Path) -> str:
    plt = _plt()
    labeled = [b for b, p in preds.items() if cfg.data.target in p.columns]
    fig, axes = plt.subplots(1, len(labeled), figsize=(3.4 * len(labeled), 3.3), squeeze=False)
    for ax, b in zip(axes[0], labeled):
        p = preds[b]
        ax.scatter(p[cfg.data.target], p["predicted_price_lakhs"], s=10, alpha=0.55, color=BLUE,
                   edgecolors="#fcfcfb", linewidths=0.5)
        lim = [0, max(p[cfg.data.target].max(), p["predicted_price_lakhs"].max()) * 1.05]
        ax.plot(lim, lim, color=INK2, lw=1, ls="--")
        perf = next(r for r in summary["batches"] if r["batch"] == b)["performance"]
        ax.set(xlim=lim, ylim=lim, xlabel="Actual (lakhs)", ylabel="Predicted (lakhs)")
        ax.set_title(f"Batch {b} · RMSE {perf['rmse']:.2f} · bias {perf['bias']:+.2f}", loc="left")
        _clean_axes(ax)
    fig.tight_layout()
    return _png(fig, out / "predicted_vs_actual.png")


def _badge(level: str) -> str:
    return (f'<span class="badge" style="background:{STATUS[level]};color:{STATUS_TEXT[level]}">'
            f'{"●" if level == "OK" else "▲"} {level}</span>')


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else html.escape(str(v))


def render_report(summary: dict, frames: dict[str, pd.DataFrame], preds: dict[str, pd.DataFrame],
                  ref_sample: pd.DataFrame, profile: dict, cfg: Config, out: Path) -> Path:
    charts_dir = out / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    try:
        imgs = {
            "psi": chart_psi_heatmap(summary, cfg, charts_dir),
            "perf": chart_performance(summary, cfg, charts_dir) if any(r["performance"] for r in summary["batches"])
            else None,
            "dist": chart_distributions(summary, frames, ref_sample, cfg, charts_dir),
            "pva": chart_pred_vs_actual(summary, preds, cfg, charts_dir)
            if any(cfg.data.target in p.columns for p in preds.values()) else None,
        }
    except ImportError:
        imgs = {}

    def img(key: str, alt: str) -> str:
        return f'<img alt="{alt}" src="data:image/png;base64,{imgs[key]}">' if imgs.get(key) else ""

    m = summary["model"]
    cards = "".join(
        f'<div class="card"><div class="card-h">Batch {html.escape(r["batch"])}</div>{_badge(r["level"])}'
        f'<div class="kv"><span>rows</span><b>{r["n_rows"]}</b></div>'
        f'<div class="kv"><span>RMSE</span><b>{_fmt((r["performance"] or {}).get("rmse"), 2)}</b></div>'
        f'<div class="kv"><span>RMSE / holdout</span><b>{_fmt((r["performance"] or {}).get("rmse_ratio"), 2)}×</b></div>'
        f'<div class="kv"><span>bias</span><b>{_fmt((r["performance"] or {}).get("bias"), 2)}</b></div>'
        f'<div class="kv"><span>drifted features</span><b>{r["drift_share"]:.0%}</b></div>'
        f'<div class="kv"><span>alerts</span><b>{len(r["alerts"])}</b></div>'
        f'<p class="action">{html.escape(r["recommended_action"])}</p></div>'
        for r in summary["batches"])
    alert_rows = "".join(
        f"<tr><td>{html.escape(a['batch'])}</td><td>{_badge(a['severity'])}</td><td><code>{a['rule']}</code></td>"
        f"<td class='num'>{_fmt(a['value'])}</td><td class='num'>{_fmt(a['threshold'])}</td>"
        f"<td>{html.escape(a['message'])}</td></tr>" for a in summary["alerts"]) or \
        "<tr><td colspan=6>No alerts.</td></tr>"
    perf_rows = "".join(
        f"<tr><td>{r['batch']}</td>" + "".join(
            f"<td class='num'>{_fmt((r['performance'] or {}).get(k), nd)}</td>"
            for k, nd in [("rmse", 2), ("mae", 2), ("r2", 3), ("mape", 3), ("bias", 2), ("rmse_ratio", 2)])
        + f"<td class='num'>{_fmt(r['prediction_summary']['mean'], 1)}</td>"
          f"<td class='num'>{_fmt(frames[r['batch']][cfg.data.target].mean() if cfg.data.target in frames[r['batch']] else None, 1)}</td></tr>"
        for r in summary["batches"])
    ops_rows = "".join(
        f"<tr><td>{r['batch']}</td><td>{html.escape(r['operational']['mode'])}</td>"
        f"<td class='num'>{r['operational']['requests']}</td><td class='num'>{r['operational']['errors']}</td>"
        f"<td class='num'>{r['operational']['error_rate']:.2%}</td>"
        f"<td class='num'>{_fmt(r['operational']['latency_ms_p50'], 1)}</td>"
        f"<td class='num'>{_fmt(r['operational']['latency_ms_p95'], 1)}</td>"
        f"<td class='num'>{_fmt(r['operational']['latency_ms_p99'], 1)}</td>"
        f"<td class='num'>{_fmt(r['operational']['throughput_rps'], 1)}</td></tr>" for r in summary["batches"])
    drift_tables = ""
    for r in summary["batches"]:
        rows = "".join(
            f"<tr><td>{row['feature']}</td><td class='num'>{_fmt(row['psi'])}</td>"
            f"<td class='num'>{_fmt(row['p_value'], 4)}</td><td>{row['test']}</td>"
            f"<td class='num'>{_fmt(row['ref_mean'], 2)}</td><td class='num'>{_fmt(row['cur_mean'], 2)}</td>"
            f"<td class='num'>{row['ref_missing']:.1%}</td><td class='num'>{row['cur_missing']:.1%}</td>"
            f"<td>{_badge(row['severity'])}</td></tr>" for row in r["drift"])
        drift_tables += (f"<details><summary>Batch {r['batch']} – feature drift table</summary><table><thead><tr>"
                         "<th>feature</th><th>PSI</th><th>p-value</th><th>test</th><th>ref mean</th><th>batch mean</th>"
                         "<th>ref missing</th><th>batch missing</th><th>level</th></tr></thead>"
                         f"<tbody>{rows}</tbody></table></details>")
    t = cfg.monitoring
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Monitoring Report</title>
<style>
body{{margin:0;background:#f9f9f7;color:{INK};font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:24px 16px 64px}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:17px;margin:32px 0 8px}}
.sub{{color:{INK2};margin:0 0 16px}}
.panel{{background:#fcfcfb;border:1px solid rgba(11,11,11,.10);border-radius:10px;padding:16px;margin:12px 0}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}}
.card{{background:#fcfcfb;border:1px solid rgba(11,11,11,.10);border-radius:10px;padding:14px}}
.card-h{{font-weight:600;margin-bottom:6px}}
.kv{{display:flex;justify-content:space-between;color:{INK2};font-variant-numeric:tabular-nums}} .kv b{{color:{INK}}}
.action{{color:{INK2};font-size:12.5px;margin:10px 0 0}}
.badge{{display:inline-block;border-radius:999px;padding:1px 9px;font-size:12px;font-weight:600;margin:2px 0 8px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid {GRID};vertical-align:top}}
th{{color:{INK2};font-weight:600}} td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
img{{max-width:100%;height:auto;display:block}} code{{font-size:12px}}
.overflow{{overflow-x:auto}} details{{margin:8px 0}} summary{{cursor:pointer;color:{INK2}}}
.meta{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:4px 16px;color:{INK2};font-size:13px}}
.meta b{{color:{INK};font-weight:600;word-break:break-all}}
</style></head><body><main>
<h1>Monitoring report – incoming batches</h1>
<p class="sub">Generated {summary['created_at']} · overall level {_badge(summary['overall_level'])}
{' · <b>retraining recommended</b>' if summary['retrain_recommended'] else ''}</p>
<div class="panel meta">
<div>Model <b>{html.escape(str(m['model_name']))} v{html.escape(str(m['model_version']))}</b></div>
<div>MLflow run <b>{html.escape(str(m['mlflow_run_id']))}</b></div>
<div>Git commit <b>{html.escape(str(m['git_commit'])[:12])}</b></div>
<div>Data version <b>{html.escape(str(m['data_version']))}</b></div>
<div>Config hash <b>{html.escape(str(m['config_hash']))}</b></div>
<div>Image <b>{html.escape(str(m['image_tag']))}</b></div>
<div>Holdout RMSE <b>{summary['reference']['holdout_rmse']:.3f}</b></div>
<div>Reference mean price <b>{summary['reference']['mean_price']:.2f}</b></div>
</div>
<h2>Status by batch</h2><div class="cards">{cards}</div>
<h2>Alerts</h2><div class="panel overflow"><table><thead><tr><th>batch</th><th>level</th><th>rule</th>
<th class="num">value</th><th class="num">threshold</th><th>explanation</th></tr></thead><tbody>{alert_rows}</tbody></table></div>
<h2>Performance drift</h2><div class="panel">{img('perf', 'RMSE and bias per batch')}
<div class="overflow"><table><thead><tr><th>batch</th><th class="num">RMSE</th><th class="num">MAE</th><th class="num">R²</th>
<th class="num">MAPE</th><th class="num">bias</th><th class="num">RMSE / holdout</th><th class="num">mean predicted</th>
<th class="num">mean actual</th></tr></thead><tbody>{perf_rows}</tbody></table></div></div>
<div class="panel">{img('pva', 'Predicted vs actual per batch')}</div>
<h2>Data drift</h2><div class="panel">{img('psi', 'PSI heatmap')}</div>
<div class="panel">{img('dist', 'Distributions of the most-drifted features')}</div>
<div class="panel">{drift_tables}</div>
<h2>Operational metrics</h2><div class="panel overflow"><table><thead><tr><th>batch</th><th>mode</th>
<th class="num">requests</th><th class="num">errors</th><th class="num">error rate</th><th class="num">p50 ms</th>
<th class="num">p95 ms</th><th class="num">p99 ms</th><th class="num">throughput rps</th></tr></thead><tbody>{ops_rows}</tbody></table>
<p class="sub">Live serving metrics (request rate, latency histogram, errors, predicted-price distribution) are in the
Grafana dashboard fed by the API's <code>/metrics</code> endpoint – see <code>monitoring/</code>.</p></div>
<h2>Alert thresholds</h2><div class="panel overflow"><table><tbody>
<tr><td>Feature drift</td><td>PSI ≥ {t.psi_warning:g} and KS/χ² p &lt; {t.ks_pvalue:g} → WARNING; PSI ≥ {t.psi_critical:g} → CRITICAL</td></tr>
<tr><td>Dataset drift</td><td>≥ {t.drift_share_critical:.0%} of features drifted → CRITICAL</td></tr>
<tr><td>Performance</td><td>RMSE ≥ {t.rmse_ratio_warning:g}× holdout → WARNING; ≥ {t.rmse_ratio_critical:g}× → CRITICAL</td></tr>
<tr><td>Bias</td><td>|bias| ≥ {t.bias_warning_pct:.0%} of mean price → WARNING; ≥ {t.bias_critical_pct:.0%} → CRITICAL</td></tr>
<tr><td>Target / prediction drift</td><td>PSI thresholds as above</td></tr>
<tr><td>Data quality</td><td>data gate fails → CRITICAL; missing share +{t.missing_increase_warning:.0%} → WARNING</td></tr>
<tr><td>Operational</td><td>error rate &gt; {t.error_rate_max:.0%} → CRITICAL; p95 latency &gt; {t.latency_p95_ms_max:g} ms → WARNING</td></tr>
</tbody></table></div>
</main></body></html>"""
    path = out / "report.html"
    path.write_text(doc)
    return path
