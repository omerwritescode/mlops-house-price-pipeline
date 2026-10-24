"""Regenerate docs/images/architecture.png:  python docs/make_diagrams.py"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).parent / "images"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
LANE = ["#f3f2ee", "#fcfcfb"]
BLUE, BLUE_L = "#2a78d6", "#cde2fb"
GATE = "#d03b3b"


def box(ax, x, y, w, h, title, sub="", fc="#ffffff", ec="#c3c2b7", tc=INK, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h - 0.32 if sub else y + h / 2, title, ha="center", va="center", fontsize=8.4,
            color=tc, weight="bold")
    if sub:
        ax.text(x + w / 2, y + (h - 0.45) / 2, sub, ha="center", va="center", fontsize=6.5, color=INK2,
                linespacing=1.35)


def polyline(ax, pts, color=INK2, ls="-", lw=1.2):
    xs, ys = zip(*pts[:-1])
    ax.plot(xs + (pts[-1][0],), ys + (pts[-1][1],), color=color, ls=ls, lw=lw, solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color=color, ls=ls, lw=lw)


def arrow(ax, a, b, color=INK2, ls="-", rad=0.0, lw=1.2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


fig, ax = plt.subplots(figsize=(15, 7.4))
ax.set_xlim(0, 21)
ax.set_ylim(0, 10.3)
ax.axis("off")
lanes = [("1  Scoping &\nconfiguration", 0, 2.6), ("2  Data", 2.6, 5.4), ("3  Modelling", 5.4, 8.2),
         ("4  Validation &\nregistry", 8.2, 11.2), ("5  Deployment\n(CI/CD)", 11.2, 14.4),
         ("6  Serving", 14.4, 17.4), ("7  Monitoring", 17.4, 21)]
for i, (name, x0, x1) in enumerate(lanes):
    ax.add_patch(plt.Rectangle((x0, 0), x1 - x0, 9.2, fc=LANE[i % 2], ec="none"))
    ax.text((x0 + x1) / 2, 9.65, name, ha="center", va="center", fontsize=9.5, color=INK, weight="bold")

# orchestration band
ax.add_patch(FancyBboxPatch((2.75, 0.12), 14.5, 0.5, boxstyle="round,pad=0.02,rounding_size=0.1",
                            fc=BLUE_L, ec=BLUE, lw=1))
ax.text(10.0, 0.37, "Prefect flow  'house-price-pipeline'  -  every stage is a task with on-disk inputs/outputs "
        "(artifacts/<stage>/), re-runnable alone:  hpp stage <name>", ha="center", va="center", fontsize=8.2, color=INK)

# lane 1
box(ax, 0.2, 5.6, 2.2, 1.5, "configs/config.yaml", "single source of truth\nvalidated on load (pydantic)")
box(ax, 0.2, 3.4, 2.2, 1.5, "Design note", "objective · error targets\nlatency & freshness SLOs")
box(ax, 0.2, 1.0, 2.2, 1.6, "Git + GitHub", "code, config, data\n(sha256 data versions)")
# lane 2
box(ax, 2.8, 5.6, 2.4, 1.5, "ingest", "train CSV (+ labelled batches)\n-> dataset + data_version")
box(ax, 2.8, 3.4, 2.4, 1.5, "validate", "schema · ranges · categories\nmissing · duplicates", ec=GATE, lw=2)
ax.text(4.0, 3.2, "DATA GATE", ha="center", fontsize=7.5, color=GATE, weight="bold")
box(ax, 2.8, 1.0, 2.4, 1.6, "Incoming batches", "1 · 2 · 3 (500 each)\nnew data in production")
# lane 3
box(ax, 5.6, 5.6, 2.4, 1.5, "features", "hash split by House_ID\nreference profile for drift")
box(ax, 5.6, 3.4, 2.4, 1.5, "train", "sklearn Pipeline +\nGridSearchCV (5-fold)")
# lane 4
box(ax, 8.4, 5.6, 2.6, 1.5, "evaluate", "RMSE/MAE/R²/MAPE thresholds\n+ compare vs production", ec=GATE, lw=2)
ax.text(9.7, 5.4, "MODEL GATE", ha="center", fontsize=7.5, color=GATE, weight="bold")
box(ax, 8.4, 3.4, 2.6, 1.5, "register", "MLflow Registry\nalias 'candidate'")
box(ax, 8.4, 1.0, 2.6, 1.6, "MLflow Tracking", "params · metrics · config\ngit commit · data version")
# lane 5
box(ax, 11.4, 5.6, 2.8, 1.5, "deploy", "alias 'production' + bundle\natomic swap, keeps 'previous'")
box(ax, 11.4, 3.4, 2.8, 1.5, "GitHub Actions", "PR: lint · tests · light run\nmain: full run · release")
box(ax, 11.4, 1.0, 2.8, 1.6, "Docker image (GHCR)", "tag sha-<commit>, v1.0.N\nrollback = re-point tag")
# lane 6
box(ax, 14.6, 5.6, 2.6, 1.5, "FastAPI service", "/predict · /model-info\n/version · /metrics")
box(ax, 14.6, 3.4, 2.6, 1.5, "deployments/current", "model.joblib + manifest\n(lineage in every response)")
box(ax, 14.6, 1.0, 2.6, 1.6, "hpp lineage", "deployed model -> run ->\ncommit -> data -> config")
# lane 7
box(ax, 17.6, 5.6, 3.2, 1.5, "monitor", "PSI/KS drift · target drift ·\nRMSE/bias per batch · alerts")
box(ax, 17.6, 3.4, 3.2, 1.5, "Prometheus + Pushgateway", "latency · throughput · errors\npredicted-price histogram")
box(ax, 17.6, 1.0, 3.2, 1.6, "Grafana + HTML report", "dashboard, alert rules\nOK / WARNING / CRITICAL")

# flow arrows
arrow(ax, (2.4, 6.35), (2.8, 6.35))
arrow(ax, (4.0, 5.6), (4.0, 4.9))
arrow(ax, (5.2, 4.15), (5.6, 6.0))
arrow(ax, (6.8, 5.6), (6.8, 4.9))
arrow(ax, (8.0, 4.15), (8.4, 6.0))
arrow(ax, (9.7, 5.6), (9.7, 4.9))
arrow(ax, (11.0, 4.15), (11.4, 6.0))
arrow(ax, (14.2, 6.35), (14.6, 4.3))
arrow(ax, (15.9, 4.9), (15.9, 5.6))
arrow(ax, (17.2, 6.35), (17.6, 6.35))
arrow(ax, (17.2, 4.15), (17.6, 4.15))
arrow(ax, (19.2, 3.4), (19.2, 2.6))
polyline(ax, [(4.0, 1.0), (4.0, 0.82), (20.95, 0.82), (20.95, 6.35), (20.8, 6.35)], color=MUTED, ls=(0, (3, 2)))
ax.text(12.5, 0.72, "incoming batches scored + drift-checked by monitor", ha="center", va="center", fontsize=6.3, color=MUTED)
arrow(ax, (9.7, 2.6), (9.7, 3.4), color=MUTED)
arrow(ax, (12.8, 3.4), (12.8, 2.6))
arrow(ax, (14.2, 1.8), (14.6, 1.8), color=MUTED)
# feedback loop
polyline(ax, [(19.2, 7.1), (19.2, 7.75), (4.0, 7.75), (4.0, 7.1)], color=GATE, lw=1.6)
ax.text(11.6, 7.98, "CRITICAL alert -> retrain with labelled batches (hpp retrain) - both gates still apply",
        ha="center", fontsize=8.2, color=GATE, weight="bold")
ax.text(10.5, -0.3, "Gate failure at any point: the flow stops with an explanation, nothing is registered or deployed, "
        "the previous model keeps serving.", ha="center", fontsize=8.4, color=INK2)
fig.savefig(OUT / "architecture.png", dpi=170, bbox_inches="tight", facecolor="#fcfcfb")
print("wrote", OUT / "architecture.png")
