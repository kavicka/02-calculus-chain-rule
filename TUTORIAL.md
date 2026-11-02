# Tutorial · 02 · Derivatives & the chain rule

🇸🇰 [Slovenská verzia](TUTORIAL.sk.md) · 📄 [README (results)](README.md)

This tutorial walks through the project step by step: what was built, why each piece is done that way, and how it works. The README shows the results; this file explains the path to them.

**Goal:** compute the gradients of a tiny calculation in three independent ways (by hand, numerically, with PyTorch) and prove they agree. This is the exact maths that backpropagation runs inside every neural network, and the foundation for the autograd engine in project 04.

**You need:** project 01's setup (`uv`, PyTorch). High-school algebra. No calculus beyond "slope of a line".

---

## Step 1 · Pick the smallest graph that has everything

```python
INPUTS = {"a": 2.0, "b": -3.0, "c": 10.0}

def forward(a, b, c):
    e = a * b      # -6
    d = e + c      #  4
    L = d ** 2     # 16
    return {"e": e, "d": d, "L": L}
```

**Why these three operations:** a neural network layer is "multiply by weights, add a bias, apply a non-linearity, compute a loss". This graph has one of each kind: `×` (weights), `+` (bias), `²` (a loss, like squared error). If you can backprop through these three, you can backprop through a network.

**Why name every intermediate (`e`, `d`):** backprop needs the intermediate values. Writing them down as named nodes turns a formula into a **computational graph**: a chain of simple steps, each with an input and an output. That picture is what PyTorch builds internally.

**Why these numbers:** `b` is negative so we see a negative gradient, and the values are small integers so every result can be checked in your head.

---

## Step 2 · Understand what a gradient is

`∂L/∂a` asks: *if I nudge `a` by a tiny amount, how much does `L` move, per unit of nudge?* It is the slope of `L` as a function of `a`, at the current point.

- Slope −24 means: increase `a` by 0.001, and `L` drops by about 0.024.
- **Gradient descent** uses this directly: to make `L` smaller, step `a` in the *opposite* direction of its gradient.

[`plot.py`](plot.py) draws this as a tangent line touching the curve `L(a)` at `a = 2` (`assets/tangent.png`).

---

## Step 3 · Backprop by hand with the chain rule

**The chain rule:** if `L` depends on `d` and `d` depends on `e`, then

```
∂L/∂e = ∂L/∂d · ∂d/∂e
```

In words: the effect of `e` on `L` = (effect of `d` on `L`) × (effect of `e` on `d`). Each factor is a **local** derivative: it only looks at one operation.

So we start at the end, where `∂L/∂L = 1`, and walk backwards, multiplying:

```python
def manual_grads(a, b, c):
    v = forward(a, b, c)
    dL_dL = 1.0
    dL_dd = dL_dL * 2 * v["d"]  # L = d²     → local 2d     → 8
    dL_de = dL_dd * 1.0         # d = e + c  → local 1      → 8
    dL_dc = dL_dd * 1.0         #            → local 1      → 8
    dL_da = dL_de * b           # e = a * b  → local b      → -24
    dL_db = dL_de * a           #            → local a      → 16
```

**Why backwards and not forwards:** we want the gradient of *one* output with respect to *many* inputs. Going backwards, each node's gradient is computed once and reused by everything feeding into it. A network has millions of inputs (weights) and one loss, so backwards is millions of times cheaper. That is the whole trick of backpropagation.

**Two rules worth memorising:**
- **`+` copies the gradient** to all its inputs unchanged (local derivative is 1). That is why `∂L/∂e` and `∂L/∂c` are both 8.
- **`×` swaps the inputs**: the gradient flowing to `a` is multiplied by `b`, and to `b` by `a`. That is why `∂L/∂a = 8 · (−3)`.

Note that the forward values (`d = 4`, `a`, `b`) are needed during the backward pass. That is why frameworks keep activations in memory during training.

---

## Step 4 · Check it numerically (no calculus)

A completely independent way to get a slope: nudge the input and measure.

```python
def numerical_grad(name, h=1e-6, central=True):
    up = {**INPUTS, name: INPUTS[name] + h}
    if central:
        down = {**INPUTS, name: INPUTS[name] - h}
        return (loss(**up) - loss(**down)) / (2 * h)
    return (loss(**up) - loss(**INPUTS)) / h
```

**Why this is useful:** it shares no code or reasoning with Step 3. If both agree, the hand derivation is almost certainly right. This "gradient check" is how you test a custom backward pass (project 04 uses the same idea in its tests).

**Why central difference `(f(x+h) − f(x−h)) / 2h`:** the one-sided version `(f(x+h) − f(x)) / h` has an error proportional to `h` because it ignores the curve's bending. The symmetric version cancels that bending term, so its error is proportional to `h²`, far smaller.

**Why it is not used for training:** it costs one extra forward pass *per input*. For a model with a million weights, that is a million forward passes per step. Backprop gets all of them in one backward pass.

---

## Step 5 · Let PyTorch do it

```python
t = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in INPUTS.items()}
e = t["a"] * t["b"]
d = e + t["c"]
L = d ** 2
e.retain_grad(); d.retain_grad()
L.backward()
```

**How it works:** `requires_grad=True` tells PyTorch to record every operation done with this tensor. While the forward pass runs, it builds the same graph from Step 1 in memory, with each node remembering its local derivative rule. `L.backward()` walks that graph backwards exactly like Step 3 and writes the results into `.grad`.

**Details in the code and why:**
- `float64` so that the comparison with the hand result is exact, not off in the 7th digit.
- `retain_grad()` on `e` and `d`: PyTorch normally only keeps `.grad` on *leaf* tensors (the inputs you created), to save memory. We ask for the intermediates so the table can show them too.

---

## Step 6 · Compare all three and save results

`main()` prints a table with manual, numerical and torch gradients and a ✓ when they agree (exactly for torch vs manual, within 1e-4 for numerical). Everything goes to `results.json`, and `plot.py` draws the graph, the tangent and the error plot from it. Result: all ✓, e.g. `∂L/∂a`: −24 / −23.999999999 / −24.

---

## Step 7 · Find out how small `h` should be

```python
def h_sweep():
    exact = manual_grads(**INPUTS)["a"]
    for exp in range(-1, -15, -1):
        h = 10.0 ** exp
        ... abs(numerical_grad("a", h, central=False) - exact) ...
        ... abs(numerical_grad("a", h, central=True) - exact) ...
```

Intuition says "smaller `h` = more accurate". The sweep shows that is only half true:

- **Large `h`:** the forward difference is off by about 9·h (0.9 at h = 0.1), because it ignores curvature. The central difference is already exact here, because `L(a)` is a parabola and the symmetric difference cancels its curvature completely.
- **Tiny `h` (below ~1e-8):** both get *worse*. `L(a+h)` and `L(a−h)` become nearly the same float64 number, and subtracting two nearly equal numbers destroys most of the significant digits (**catastrophic cancellation**). At h = 1e-14 the error is ~0.16.

**Lesson:** numerical derivatives are a test tool with a sweet spot (`h ≈ 1e-6` for float64), not a training method.

---

## Try it yourself

1. Add a fourth operation, e.g. `L2 = tanh(L)`, derive its local derivative (`1 − tanh²`) and extend all three methods.
2. Change `b` to `0`. What happens to `∂L/∂a`, and why does it make sense from the "× swaps inputs" rule?
3. Run the sweep in `float32` (use NumPy `float32`). Where is the new sweet spot for `h`?

## Recap

| Step | Idea |
|---|---|
| 1 | A formula = a graph of simple operations with named intermediates |
| 2 | Gradient = local slope; step against it to lower the loss |
| 3 | Chain rule backwards: multiply local derivatives; `+` copies, `×` swaps |
| 4 | Finite differences give an independent check |
| 5 | PyTorch records the graph and runs the same backward walk |
| 7 | `h` too big → curvature error; too small → rounding error |
