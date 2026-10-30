"""Gradients of a tiny computational graph, three ways: by hand (chain rule), numerically, and with PyTorch autograd.

Graph:  e = a * b,  d = e + c,  L = d ** 2
"""

import json

import torch

INPUTS = {"a": 2.0, "b": -3.0, "c": 10.0}


def forward(a: float, b: float, c: float) -> dict[str, float]:
    e = a * b
    d = e + c
    L = d**2
    return {"e": e, "d": d, "L": L}


def manual_grads(a: float, b: float, c: float) -> dict[str, float]:
    """Backward pass by hand: start at L and multiply local derivatives on the way back."""
    v = forward(a, b, c)
    dL_dL = 1.0
    dL_dd = dL_dL * 2 * v["d"]  # L = d²       → dL/dd = 2d
    dL_de = dL_dd * 1.0  # d = e + c    → dd/de = 1  (addition passes the gradient through)
    dL_dc = dL_dd * 1.0  #              → dd/dc = 1
    dL_da = dL_de * b  # e = a * b    → de/da = b  (multiplication swaps the inputs)
    dL_db = dL_de * a  #              → de/db = a
    return {"d": dL_dd, "e": dL_de, "a": dL_da, "b": dL_db, "c": dL_dc}


def loss(**kw: float) -> float:
    return forward(kw["a"], kw["b"], kw["c"])["L"]


def numerical_grad(name: str, h: float = 1e-6, central: bool = True) -> float:
    """Nudge one input by h and watch how L changes."""
    up = {**INPUTS, name: INPUTS[name] + h}
    if central:
        down = {**INPUTS, name: INPUTS[name] - h}
        return (loss(**up) - loss(**down)) / (2 * h)
    return (loss(**up) - loss(**INPUTS)) / h


def torch_grads() -> dict[str, float]:
    t = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in INPUTS.items()}
    e = t["a"] * t["b"]
    d = e + t["c"]
    L = d**2
    e.retain_grad()  # intermediate tensors don't keep .grad unless asked
    d.retain_grad()
    L.backward()
    return {"d": d.grad.item(), "e": e.grad.item(), **{k: v.grad.item() for k, v in t.items()}}


def h_sweep() -> list[dict[str, float]]:
    """Error of forward vs central differences for dL/da across step sizes h."""
    exact = manual_grads(**INPUTS)["a"]
    rows = []
    for exp in range(-1, -15, -1):
        h = 10.0**exp
        rows.append(
            {
                "h": h,
                "forward_err": abs(numerical_grad("a", h, central=False) - exact),
                "central_err": abs(numerical_grad("a", h, central=True) - exact),
            }
        )
    return rows


def main() -> None:
    values = forward(**INPUTS)
    manual = manual_grads(**INPUTS)
    numeric = {k: numerical_grad(k) for k in INPUTS}
    autograd = torch_grads()

    print("Forward pass")
    for k, v in {**INPUTS, **values}.items():
        print(f"  {k} = {v:g}")

    print("\nGradients dL/dx")
    print(f"  {'x':>2} {'manual':>10} {'numerical':>14} {'torch':>10}  match")
    for k in ["d", "e", "c", "a", "b"]:
        num = numeric.get(k)
        num_s = f"{num:14.6f}" if num is not None else f"{'–':>14}"
        ok = abs(manual[k] - autograd[k]) < 1e-9 and (num is None or abs(manual[k] - num) < 1e-4)
        print(f"  {k:>2} {manual[k]:10.4f} {num_s} {autograd[k]:10.4f}  {'✓' if ok else '✗'}")

    sweep = h_sweep()
    print("\nNumerical error for dL/da vs step size h")
    print(f"  {'h':>8} {'forward diff':>14} {'central diff':>14}")
    for r in sweep:
        print(f"  {r['h']:8.0e} {r['forward_err']:14.2e} {r['central_err']:14.2e}")

    with open("results.json", "w") as f:
        json.dump(
            {"inputs": INPUTS, "values": values, "manual": manual, "numerical": numeric, "torch": autograd, "h_sweep": sweep},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
