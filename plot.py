"""Draws the computational graph, the finite-difference error sweep and a tangent-line view from results.json."""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

R = json.load(open("results.json"))
val = {**R["inputs"], **R["values"]}
grad = {**R["manual"], "L": 1.0}

FWD, BWD, OP = "#2563eb", "#dc2626", "#6b7280"


def graph() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    nodes = {"a": (1, 3.6), "b": (1, 2.0), "c": (1, 0.6), "e": (4.4, 2.8), "d": (7.2, 1.7), "L": (10, 1.7)}
    labels = {"e": "e = a·b", "d": "d = e+c", "L": "L = d²"}
    edges = [("a", "e"), ("b", "e"), ("e", "d"), ("c", "d"), ("d", "L")]

    for src, dst in edges:
        (x1, y1), (x2, y2) = nodes[src], nodes[dst]
        ax.add_patch(FancyArrowPatch((x1 + 0.8, y1), (x2 - 0.75, y2), arrowstyle="-|>", mutation_scale=14, color="#9ca3af", lw=1.4))

    for name, (x, y) in nodes.items():
        ax.add_patch(FancyBboxPatch((x - 0.7, y - 0.42), 1.5, 0.84, boxstyle="round,pad=0.02,rounding_size=0.15", fc="white", ec="#374151", lw=1.3))
        ax.text(x + 0.05, y + 0.17, labels.get(name, name), fontsize=10.5, fontweight="bold", ha="center", va="center")
        ax.text(x - 0.6, y - 0.18, f"{val[name]:g}", color=FWD, fontsize=11, va="center")
        ax.text(x + 0.7, y - 0.18, f"∂L={grad[name]:g}", color=BWD, fontsize=10, ha="right", va="center")

    ax.text(0.3, 4.4, "Forward: values (blue) flow left → right", color=FWD, fontsize=10)
    ax.text(6.0, 4.4, "Backward: gradients ∂L/∂x (red) flow right → left", color=BWD, fontsize=10)
    notes = [
        (10.05, 2.35, "L = d²  →  ∂L/∂d = 2d = 8"),
        (7.2, 0.95, "d = e + c  →  '+' copies the gradient: 8, 8"),
        (4.5, 3.5, "e = a·b  →  '×' swaps inputs: ∂L/∂a = 8·b = −24,  ∂L/∂b = 8·a = 16"),
    ]
    for x, y, t in notes:
        ax.text(x, y, t, fontsize=9, color="#374151", ha="center")
    fig.tight_layout()
    fig.savefig("assets/computational_graph.png", dpi=150)


def h_sweep() -> None:
    rows = R["h_sweep"]
    h = np.array([r["h"] for r in rows])
    fwd = np.array([r["forward_err"] for r in rows])
    cen = np.maximum(np.array([r["central_err"] for r in rows]), 1e-16)  # log scale can't show 0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.loglog(h, fwd, "o-", label="forward difference  (f(x+h) − f(x)) / h")
    ax.loglog(h, cen, "s-", label="central difference  (f(x+h) − f(x−h)) / 2h")
    ax.axvline(1e-6, color="gray", ls="--", alpha=0.6)
    ax.text(1.3e-6, 1e-12, "h = 1e-6\n(used in the check)", fontsize=8, color="gray")
    ax.invert_xaxis()
    ax.set(title="Error of the numerical derivative ∂L/∂a vs step size h", xlabel="h (smaller →)", ylabel="|numerical − exact|")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("assets/numerical_error_vs_h.png", dpi=150)


def tangent() -> None:
    b, c = R["inputs"]["b"], R["inputs"]["c"]
    a0, L0, slope = R["inputs"]["a"], R["values"]["L"], R["manual"]["a"]
    a = np.linspace(0, 4.5, 200)
    L = (a * b + c) ** 2

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(a, L, label="L(a) = (a·b + c)²   with b = −3, c = 10")
    ax.plot(a, L0 + slope * (a - a0), "--", color=BWD, label=f"tangent at a = {a0:g}: slope ∂L/∂a = {slope:g}")
    ax.plot([a0], [L0], "o", color="black")
    ax.annotate(f"a = {a0:g}, L = {L0:g}", (a0, L0), textcoords="offset points", xytext=(10, 12))
    ax.set(ylim=(-5, 110), xlabel="a", ylabel="L", title="The derivative is the slope of the tangent")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("assets/tangent.png", dpi=150)


if __name__ == "__main__":
    graph()
    h_sweep()
    tangent()
