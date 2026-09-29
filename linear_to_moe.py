"""Train a linear model, upcycle it into a Mixture-of-Experts, and keep training.

The target is a piecewise-linear map: each input belongs to one of several
regimes, and each regime has its own weight matrix. One linear layer can only
learn a compromise across regimes, so its loss flattens. Copying that trained
layer into every expert makes the MoE reproduce the linear model exactly
(identical experts, gates that sum to 1). Continued training lets the router
send different regimes to different experts, and the task loss falls further.

The same linear model, with the same optimizer, is then trained for the same
number of extra steps. It stays on the plateau. The MoE, starting from the
copied weights, keeps reducing the loss.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor

from ink_page import write_report

D_IN = 32
D_OUT = 32
N_REGIMES = 4
N_EXPERTS = 4
TOP_K = 2
N_TRAIN = 8192
N_TEST = 2048
BATCH = 256
LINEAR_LR = 4e-2
MOE_ROUTER_LR = 1e-2
MOE_EXPERT_LR = 3e-3
AUX_COEF = 1e-2
LINEAR_EPOCHS = 40
MOE_EPOCHS = 40
SEED = 0
ROUTER_SEED = 4
OUT_DIR = Path(__file__).resolve().parent / "outputs"
REPORT_PATH = Path(__file__).resolve().parent / "how-it-works.html"
LOG: list[str] = []

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#1a1a1a",
        "xtick.color": "#1a1a1a",
        "ytick.color": "#1a1a1a",
        "text.color": "#1a1a1a",
    }
)


def say(message: str = "") -> None:
    print(message)
    LOG.append(message)


def synthesize(
    n: int, d_in: int, d_out: int, n_regimes: int, seed: int
) -> tuple[Tensor, Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d_in, generator=g)
    projection = torch.randn(d_in, n_regimes, generator=g)
    regime = (x @ projection).argmax(dim=1)
    weight = torch.randn(n_regimes, d_out, d_in, generator=g) / math.sqrt(d_in)
    bias = 0.1 * torch.randn(n_regimes, d_out, generator=g)
    y = torch.zeros(n, d_out)
    for r in range(n_regimes):
        mask = regime == r
        y[mask] = x[mask] @ weight[r].T + bias[r]
    y = y + 0.02 * torch.randn(n, d_out, generator=g)
    return x, y, regime


class UpcycledMoE(nn.Module):
    """Sparse MoE whose experts start as clones of one trained linear layer."""

    def __init__(self, linear: nn.Linear, n_experts: int, top_k: int) -> None:
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k must be in [1, {n_experts}], got {top_k}")
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            nn.Linear(linear.in_features, linear.out_features) for _ in range(n_experts)
        )
        with torch.no_grad():
            for expert in self.experts:
                expert.weight.copy_(linear.weight)
                expert.bias.copy_(linear.bias)
        self.router = nn.Linear(linear.in_features, n_experts)
        # Own generator, so this init does not depend on whatever else used the global RNG.
        nn.init.normal_(
            self.router.weight,
            std=1e-2,
            generator=torch.Generator().manual_seed(ROUTER_SEED),
        )
        nn.init.zeros_(self.router.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.router(x)
        probs = torch.softmax(logits, dim=-1)
        top_val, top_idx = torch.topk(logits, self.top_k, dim=-1)
        gates = torch.softmax(top_val, dim=-1)

        expert_out = torch.stack([expert(x) for expert in self.experts], dim=1)
        index = top_idx.unsqueeze(-1).expand(-1, -1, expert_out.size(-1))
        chosen = torch.gather(expert_out, 1, index)
        y = (gates.unsqueeze(-1) * chosen).sum(dim=1)

        # Hard dispatch counts. Kept out of the autograd graph on purpose:
        # which expert was picked is discrete; only the softmax masses get gradients.
        dispatch = torch.zeros(
            probs.shape, device=probs.device, dtype=probs.dtype
        )
        dispatch.scatter_(1, top_idx, 1.0)
        fraction = dispatch.mean(dim=0)
        fraction = fraction / fraction.sum().clamp_min(1e-9)
        density = probs.mean(dim=0)
        aux = self.n_experts * torch.sum(fraction * density)
        return y, aux


def batches(x: Tensor, y: Tensor, batch: int, gen: torch.Generator):
    perm = torch.randperm(x.shape[0], generator=gen)
    for start in range(0, x.shape[0], batch):
        idx = perm[start : start + batch]
        yield x[idx], y[idx]


@torch.no_grad()
def mse(model: nn.Module, x: Tensor, y: Tensor, moe: bool) -> float:
    model.eval()
    if moe:
        pred, _ = model(x)
    else:
        pred = model(x)
    return torch.mean((pred - y) ** 2).item()


def train_linear(
    model: nn.Linear,
    x: Tensor,
    y: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    epochs: int,
    lr: float,
    seed: int,
    opt: torch.optim.Optimizer | None = None,
    epoch_offset: int = 0,
    record_start: bool = True,
) -> tuple[list[float], list[float], torch.optim.Optimizer]:
    # SGD, with a modest step, so the descent is visible across epochs.
    # Adam reaches this least-squares fit in a single epoch.
    if opt is None:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    gen = torch.Generator().manual_seed(seed)
    history: list[float] = []
    train_hist: list[float] = []
    if record_start:
        history.append(mse(model, x_test, y_test, moe=False))
        say(f"  linear  epoch {epoch_offset:3d}  test {history[0]:.6f}")
    for step in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for xb, yb in batches(x, y, BATCH, gen):
            opt.zero_grad()
            loss = torch.mean((model(xb) - yb) ** 2)
            loss.backward()
            opt.step()
            total += loss.item() * xb.shape[0]
            seen += xb.shape[0]
        test = mse(model, x_test, y_test, moe=False)
        history.append(test)
        train = total / seen
        train_hist.append(train)
        shown = epoch_offset + step
        if step == 1 or step % 5 == 0 or step == epochs:
            say(f"  linear  epoch {shown:3d}  train {train:.6f}  test {test:.6f}")
    return history, train_hist, opt


@torch.no_grad()
def router_confidence(model: UpcycledMoE, x: Tensor) -> float:
    model.eval()
    probs = torch.softmax(model.router(x), dim=-1)
    return probs.max(dim=-1).values.mean().item()


@torch.no_grad()
def regime_mse(
    model: nn.Module, x: Tensor, y: Tensor, regime: Tensor, n_regimes: int, moe: bool
) -> list[float]:
    model.eval()
    pred = model(x)[0] if moe else model(x)
    scores: list[float] = []
    for r in range(n_regimes):
        mask = regime == r
        if int(mask.sum()) == 0:
            scores.append(float("nan"))
            continue
        scores.append(torch.mean((pred[mask] - y[mask]) ** 2).item())
    return scores


@torch.no_grad()
def expert_drift(model: UpcycledMoE, base_w: list[Tensor], base_b: list[Tensor]) -> list[float]:
    drifts: list[float] = []
    for expert, weight, bias in zip(model.experts, base_w, base_b):
        delta = torch.sqrt(
            (expert.weight - weight).square().sum() + (expert.bias - bias).square().sum()
        )
        scale = torch.sqrt(weight.square().sum() + bias.square().sum()).clamp_min(1e-8)
        drifts.append((delta / scale).item())
    return drifts


def train_moe(
    model: UpcycledMoE,
    x: Tensor,
    y: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    epochs: int,
    seed: int,
    epoch_offset: int = 0,
) -> dict[str, list[float]]:
    opt = torch.optim.Adam(
        [
            {"params": model.experts.parameters(), "lr": MOE_EXPERT_LR},
            {"params": model.router.parameters(), "lr": MOE_ROUTER_LR},
        ]
    )
    gen = torch.Generator().manual_seed(seed)
    history: list[float] = []
    train_hist: list[float] = []
    aux_hist: list[float] = []
    conf_hist: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        aux_total = 0.0
        seen = 0
        for xb, yb in batches(x, y, BATCH, gen):
            opt.zero_grad()
            pred, aux = model(xb)
            task = torch.mean((pred - yb) ** 2)
            (task + AUX_COEF * aux).backward()
            opt.step()
            total += task.item() * xb.shape[0]
            aux_total += float(aux.detach()) * xb.shape[0]
            seen += xb.shape[0]
        test = mse(model, x_test, y_test, moe=True)
        history.append(test)
        train = total / seen
        train_hist.append(train)
        aux_hist.append(aux_total / seen)
        conf_hist.append(router_confidence(model, x_test))
        shown = epoch_offset + epoch
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            say(
                f"  moe     epoch {shown:3d}  train {train:.6f}  test {test:.6f}"
                f"  router {conf_hist[-1]:.3f}"
            )
    return {"test": history, "train": train_hist, "aux": aux_hist, "confidence": conf_hist}


@torch.no_grad()
def routing_table(model: UpcycledMoE, x: Tensor, regime: Tensor, n_regimes: int) -> Tensor:
    model.eval()
    logits = model.router(x)
    choice = logits.argmax(dim=1)
    table = torch.zeros(n_regimes, model.n_experts)
    for r in range(n_regimes):
        picked = choice[regime == r]
        if picked.numel() == 0:
            continue
        counts = torch.bincount(picked, minlength=model.n_experts).float()
        table[r] = counts / counts.sum()
    return table


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)


def save_plot(
    linear_test: list[float],
    linear_continued: list[float],
    moe_test: list[float],
    path: Path,
) -> None:
    # linear_test includes the untrained point at epoch 0.
    n_lin = len(linear_test) - 1
    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    ax.plot(range(len(linear_test)), linear_test, color="#1f4e79", lw=2.2, label="Linear model")
    extra_x = list(range(n_lin, n_lin + len(moe_test) + 1))
    ax.plot(
        extra_x,
        [linear_test[-1], *linear_continued],
        color="#1f4e79",
        lw=2.0,
        ls="--",
        label="Same linear model, 40 more epochs",
    )
    ax.plot(
        extra_x,
        [linear_test[-1], *moe_test],
        color="#c45911",
        lw=2.2,
        label="Those weights, upcycled into an MoE",
    )
    ax.axvline(n_lin, color="#888888", lw=1.0, ls=":")
    drop_at = min(3, len(linear_test) - 1)
    moe_i = min(len(moe_test) - 1, max(0, round(0.75 * (len(moe_test) - 1))))
    ax.annotate(
        "Linear loss falls,\nthen the fit plateaus",
        xy=(drop_at, linear_test[drop_at]),
        xytext=(7, 1.22),
        fontsize=10,
        color="#1f4e79",
        arrowprops={"arrowstyle": "->", "color": "#1f4e79", "lw": 1.0},
    )
    ax.annotate(
        "Copy the layer\ninto 4 experts",
        xy=(n_lin, linear_test[-1]),
        xytext=(n_lin + 6, 1.18),
        fontsize=10,
        color="#333333",
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.0},
    )
    ax.annotate(
        "MoE keeps training\nand the loss keeps falling",
        xy=(n_lin + 1 + moe_i, moe_test[moe_i]),
        xytext=(n_lin + 8, 0.38),
        fontsize=10,
        color="#c45911",
        arrowprops={"arrowstyle": "->", "color": "#c45911", "lw": 1.0},
    )
    ax.set_xlim(-1, n_lin + len(moe_test) + 1)
    ax.set_ylim(0, 1.55)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test MSE")
    ax.set_title("One linear layer, then the same weights as a mixture of experts")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.42, -0.16), ncol=2)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_checkpoints(
    start: float, plateau: float, continued: float, moe_final: float, path: Path
) -> None:
    labels = [
        "Linear\nbefore training",
        "Linear\nat epoch 40",
        "Linear\nat epoch 80",
        "MoE\nat epoch 80",
    ]
    values = [start, plateau, continued, moe_final]
    colors = ["#8aa4c2", "#1f4e79", "#5b7c99", "#c45911"]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    bars = ax.bar(labels, values, color=colors, width=0.68)
    ax.set_ylabel("Test MSE")
    ax.set_ylim(0, max(values) * 1.22)
    ax.set_title("Test loss at the four checkpoints")
    _style_ax(ax)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.035,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_routing(table: Tensor, path: Path) -> None:
    data = table.numpy()
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    image = ax.imshow(data, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(data.shape[1]), [f"Expert {i}" for i in range(data.shape[1])])
    ax.set_yticks(range(data.shape[0]), [f"Regime {i}" for i in range(data.shape[0])])
    ax.set_xlabel("Expert chosen by the router")
    ax.set_ylabel("True regime in the data")
    ax.set_title("Highest-scoring expert for each regime")
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            value = float(data[r, c])
            ink = "white" if value > 0.55 else "#1a1a1a"
            ax.text(c, r, f"{value:.0%}", ha="center", va="center", color=ink, fontsize=13)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Share of test tokens")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_reference_layout(path: Path) -> None:
    """Memory picture for the 30.53B reference shape. These numbers are not measured here."""
    cards = [
        ("One B200 node", "EP = 8  ·  8 GPUs", "61.9 GiB", "87.4 GiB", "167.6 GiB", True),
        ("One H100 node", "EP = 8  ·  8 GPUs", "61.9 GiB", "87.4 GiB", "74.5 GiB", False),
        ("Two H100 nodes", "EP = 16  ·  16 GPUs", "33.8 GiB", "59.3 GiB", "74.5 GiB", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.3))
    fig.patch.set_facecolor("#f6f4ef")
    for ax, (title, subtitle, state, total, card, fits) in zip(axes, cards):
        ax.set_facecolor("#f6f4ef")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            plt.Rectangle(
                (0.04, 0.06),
                0.92,
                0.88,
                facecolor="white",
                edgecolor="#e0dacf",
                linewidth=1.2,
                transform=ax.transAxes,
            )
        )
        ax.text(0.50, 0.80, title, ha="center", va="center", fontsize=13, transform=ax.transAxes)
        ax.text(
            0.50,
            0.68,
            subtitle,
            ha="center",
            va="center",
            fontsize=10,
            color="#666666",
            transform=ax.transAxes,
        )
        for y, label, value in (
            (0.52, "Training state", state),
            (0.40, "With one sequence", total),
            (0.28, "Card memory", card),
        ):
            ax.text(0.10, y, label, ha="left", va="center", fontsize=11, color="#333333", transform=ax.transAxes)
            ax.text(0.90, y, value, ha="right", va="center", fontsize=11, transform=ax.transAxes)
        verdict = "Fits" if fits else "Does not fit"
        ink = "#1f4e79" if fits else "#8c3a2f"
        ax.text(0.50, 0.14, verdict, ha="center", va="center", fontsize=13, color=ink, transform=ax.transAxes)
    fig.suptitle("Where the reference model fits", fontsize=15)
    fig.text(
        0.5,
        0.01,
        "Reference shape, 30.53 billion parameters, 16 bytes of training state each. Not measured in this run.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_terminal(lines: list[str], path: Path) -> None:
    header_in = 0.46
    fig_h = header_in + 0.25 * len(lines) + 0.18
    fig = plt.figure(figsize=(11.4, fig_h), dpi=140, facecolor="#1b1f24")
    fig.text(
        0.018,
        1 - 0.16 / fig_h,
        "linear_to_moe.py",
        color="#c5cdd6",
        fontsize=11,
        family="monospace",
        va="top",
    )
    ax = fig.add_axes((0.02, 0.012, 0.96, 1 - (header_in / fig_h) - 0.012))
    ax.set_facecolor("#1b1f24")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(lines))
    for i, line in enumerate(lines):
        text = line if line else " "
        if text[:1].isalpha() and not text.startswith(" "):
            color = "#8ec8ff"
        elif text.startswith("Saved") or "Saved " in text:
            color = "#9dcc8a"
        else:
            color = "#e6edf3"
        ax.text(
            0.0,
            len(lines) - i - 0.7,
            text,
            color=color,
            fontsize=9.2,
            family="monospace",
            va="center",
        )
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


@torch.no_grad()
def expert_load(model: UpcycledMoE, x: Tensor) -> Tensor:
    """Mean tokens dispatched to each expert, scaled to a batch of BATCH."""
    model.eval()
    totals = torch.zeros(model.n_experts)
    seen = 0
    for start in range(0, x.shape[0], BATCH):
        xb = x[start : start + BATCH]
        top_idx = torch.topk(model.router(xb), model.top_k, dim=-1).indices
        counts = torch.bincount(top_idx.reshape(-1), minlength=model.n_experts).float()
        totals += counts * (BATCH / xb.shape[0])
        seen += 1
    return totals / seen


def main() -> None:
    LOG.clear()
    torch.manual_seed(SEED)
    x, y, regime = synthesize(N_TRAIN + N_TEST, D_IN, D_OUT, N_REGIMES, seed=SEED)
    x_train, x_test = x[:N_TRAIN], x[N_TRAIN:]
    y_train, y_test = y[:N_TRAIN], y[N_TRAIN:]
    regime_test = regime[N_TRAIN:]

    linear = nn.Linear(D_IN, D_OUT)

    say("Phase 1 — train the linear model until its test loss flattens")
    linear_hist, linear_train, opt = train_linear(
        linear,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=LINEAR_EPOCHS,
        lr=LINEAR_LR,
        seed=1,
    )

    moe = UpcycledMoE(linear, n_experts=N_EXPERTS, top_k=TOP_K)
    linear.eval()
    moe.eval()
    with torch.no_grad():
        linear_pred = linear(x_test)
        moe_pred, _ = moe(x_test)
    max_abs = (linear_pred - moe_pred).abs().max().item()
    linear_at_convert = torch.mean((linear_pred - y_test) ** 2).item()
    moe_at_convert = torch.mean((moe_pred - y_test) ** 2).item()
    say("")
    say(f"Conversion — experts cloned from the linear layer (top-{TOP_K} of {N_EXPERTS})")
    say(f"  linear test MSE {linear_at_convert:.6f}")
    say(f"  moe    test MSE {moe_at_convert:.6f}")
    say(f"  max |linear - moe| on the test set  {max_abs:.3e}")
    if max_abs > 1e-5:
        raise SystemExit("MoE predictions did not match the linear model at conversion")
    regime_at_copy = regime_mse(linear, x_test, y_test, regime_test, N_REGIMES, moe=False)
    base_w = [expert.weight.detach().clone() for expert in moe.experts]
    base_b = [expert.bias.detach().clone() for expert in moe.experts]
    confidence_at_copy = router_confidence(moe, x_test)

    say("")
    say("Phase 2 — same linear model keeps training; the MoE trains from the copied weights")
    continued_hist, continued_train, _ = train_linear(
        linear,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=MOE_EPOCHS,
        lr=LINEAR_LR,
        seed=2,
        opt=opt,
        epoch_offset=LINEAR_EPOCHS,
        record_start=False,
    )
    moe_trace = train_moe(
        moe,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=MOE_EPOCHS,
        seed=3,
        epoch_offset=LINEAR_EPOCHS,
    )

    moe_hist = moe_trace["test"]
    regime_after = regime_mse(moe, x_test, y_test, regime_test, N_REGIMES, moe=True)
    regime_linear_end = regime_mse(linear, x_test, y_test, regime_test, N_REGIMES, moe=False)
    drift = expert_drift(moe, base_w, base_b)

    say("")
    say("Test MSE")
    say(f"  linear at plateau     {linear_hist[-1]:.6f}")
    say(f"  linear after +{MOE_EPOCHS:<3}    {continued_hist[-1]:.6f}")
    say(f"  moe at conversion     {moe_at_convert:.6f}")
    say(f"  moe after +{MOE_EPOCHS:<3}       {moe_hist[-1]:.6f}")
    say(f"  moe drop              {moe_at_convert - moe_hist[-1]:.6f}")
    say("")
    say("Per-regime test MSE, linear at the copy vs mixture at the end")
    for r in range(N_REGIMES):
        say(f"  regime {r}:  {regime_at_copy[r]:.4f}  ->  {regime_after[r]:.4f}")
    say("")
    say("Expert drift from the cloned weights (relative L2)")
    say("  " + "  ".join(f"expert {i}: {drift[i]:.3f}" for i in range(N_EXPERTS)))
    say(f"Router confidence  {confidence_at_copy:.3f}  ->  {moe_trace['confidence'][-1]:.3f}")

    table = routing_table(moe, x_test, regime_test, N_REGIMES)
    say("")
    say("Router choice by true regime (row = regime, col = expert, fraction)")
    for r in range(N_REGIMES):
        cells = "  ".join(f"{table[r, e]:.2f}" for e in range(N_EXPERTS))
        say(f"  regime {r}: {cells}")

    linear_n = sum(p.numel() for p in linear.parameters())
    moe_n = sum(p.numel() for p in moe.parameters())
    expert_n = sum(p.numel() for p in moe.experts[0].parameters())
    router_n = sum(p.numel() for p in moe.router.parameters())
    active_n = TOP_K * expert_n + router_n
    say("")
    say("Size")
    say(f"  linear parameters     {linear_n}")
    say(f"  moe total parameters  {moe_n}")
    say(f"  moe active parameters {active_n}  (top-{TOP_K} experts + router)")

    load = expert_load(moe, x_train)
    say("")
    say("Tokens dispatched per expert in an average batch of 256")
    say("  " + "  ".join(f"expert {i}: {load[i]:.0f}" for i in range(N_EXPERTS)))

    if not (linear_hist[-1] < linear_hist[0]):
        raise SystemExit("Linear model did not reduce test loss")
    if not (moe_hist[-1] < moe_at_convert):
        raise SystemExit("MoE did not reduce test loss after conversion")
    if not (moe_hist[-1] < moe_hist[0]):
        raise SystemExit("MoE loss did not keep falling across its own training")
    if moe_hist[-1] >= continued_hist[-1] - 1e-4:
        raise SystemExit("MoE did not beat continued linear training")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "loss": OUT_DIR / "linear_to_moe_loss.png",
        "checkpoints": OUT_DIR / "checkpoints.png",
        "routing": OUT_DIR / "routing.png",
        "layout": OUT_DIR / "reference_layout.png",
        "terminal": OUT_DIR / "training_log.png",
    }
    save_plot(linear_hist, continued_hist, moe_hist, paths["loss"])
    save_checkpoints(
        linear_hist[0], linear_hist[-1], continued_hist[-1], moe_hist[-1], paths["checkpoints"]
    )
    save_routing(table, paths["routing"])
    save_reference_layout(paths["layout"])
    say("")
    for path in paths.values():
        if path.name == "training_log.png":
            continue
        say(f"Saved {path.name}")
    save_terminal(LOG, paths["terminal"])
    say(f"Saved {paths['terminal'].name}")
    series = (
        linear_hist
        + linear_train
        + continued_hist
        + continued_train
        + moe_hist
        + moe_trace["train"]
        + moe_trace["aux"]
        + moe_trace["confidence"]
        + regime_at_copy
        + regime_after
        + drift
    )
    if any(not math.isfinite(value) for value in series):
        raise SystemExit("Simulation produced a non-finite loss or score")
    if moe_trace["confidence"][-1] <= confidence_at_copy:
        raise SystemExit("Router confidence did not rise")
    if any(after > before for before, after in zip(regime_at_copy, regime_after)):
        raise SystemExit("A regime got worse after the mixture trained")

    write_report(
        {
            "path": REPORT_PATH,
            "log": LOG,
            "start": linear_hist[0],
            "plateau": linear_hist[-1],
            "continued": continued_hist[-1],
            "moe_final": moe_hist[-1],
            "max_abs": max_abs,
            "linear_n": linear_n,
            "moe_n": moe_n,
            "active_n": active_n,
            "table": table,
            "load": load,
            "linear_epochs": LINEAR_EPOCHS,
            "moe_epochs": MOE_EPOCHS,
            "linear_test": linear_hist,
            "linear_train": linear_train,
            "continued_test": continued_hist,
            "continued_train": continued_train,
            "moe_test": moe_hist,
            "moe_train": moe_trace["train"],
            "aux": moe_trace["aux"],
            "confidence": moe_trace["confidence"],
            "confidence0": confidence_at_copy,
            "regime_linear": regime_at_copy,
            "regime_linear_end": regime_linear_end,
            "regime_moe": regime_after,
            "drift": drift,
            "batch": BATCH,
            "top_k": TOP_K,
            "aux_coef": AUX_COEF,
        }
    )


if __name__ == "__main__":
    main()
