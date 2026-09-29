"""Single-page report. White page, ink lettering, charts from one simulation."""

from __future__ import annotations

import base64
from pathlib import Path

from torch import Tensor

ROOT = Path(__file__).resolve().parent
FONT_DIR = ROOT / "fonts"
BLUE = "#1d4ed8"
BLUE_SOFT = "#60a5fa"
ORANGE = "#c2410c"
ORANGE_SOFT = "#fb923c"
INK = "#18181b"
MUTED = "#71717a"


def _tick_label(tick: float) -> str:
    if abs(tick) < 1e-9:
        return "0"
    if abs(tick) >= 10:
        return f"{tick:.0f}"
    if abs(tick) >= 1:
        return f"{tick:.1f}"
    return f"{tick:.2f}"


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _font_css() -> str:
    faces = []
    for family, filename in (
        ("Caveat", "Caveat-Regular.ttf"),
        ("Patrick Hand", "PatrickHand-Regular.ttf"),
    ):
        raw = base64.b64encode((FONT_DIR / filename).read_bytes()).decode("ascii")
        faces.append(
            f'@font-face{{font-family:"{family}";'
            f'src:url(data:font/ttf;base64,{raw}) format("truetype");'
            "font-weight:400;font-style:normal;font-display:swap;}"
        )
    return "\n".join(faces)


def _xy(x: float, y: float, x0: float, x1: float, y0: float, y1: float, box: tuple[float, float, float, float]) -> tuple[float, float]:
    left, top, width, height = box
    px = left + (x - x0) / (x1 - x0) * width
    py = top + (1 - (y - y0) / (y1 - y0)) * height
    return px, py


def svg_lines(
    series: list[dict],
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    x_ticks: list[float],
    y_ticks: list[float],
) -> str:
    w, h = 680, 280
    box = (58, 16, w - 74, h - 52)
    left, top, width, height = box
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    for tick in y_ticks:
        _, py = _xy(x0, tick, x0, x1, y0, y1, box)
        parts.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + width}" y2="{py:.1f}" class="hair"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{py + 5:.1f}" text-anchor="end" class="tick">{_tick_label(tick)}</text>'
        )
    for tick in x_ticks:
        px, _ = _xy(tick, y0, x0, x1, y0, y1, box)
        parts.append(
            f'<text x="{px:.1f}" y="{top + height + 22}" text-anchor="middle" class="tick">{int(tick)}</text>'
        )
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" class="axis"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" class="axis"/>'
    )
    for item in series:
        points = []
        for x, y in zip(item["xs"], item["ys"]):
            px, py = _xy(x, y, x0, x1, y0, y1, box)
            points.append(f"{px:.1f},{py:.1f}")
        dash = ' stroke-dasharray="7 6"' if item.get("dash") else ""
        parts.append(
            f'<polyline fill="none" stroke="{item["color"]}" stroke-width="2.7" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash} points="{" ".join(points)}"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def svg_bars(labels: list[str], groups: list[dict], y_max: float) -> str:
    w, h = 680, 280
    left, top, width, height = 58, 16, w - 74, h - 58
    n = len(labels)
    g = len(groups)
    slot = width / n
    bar_w = min(28, (slot * 0.7) / g)
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    for frac in (0, 0.5, 1):
        py = top + (1 - frac) * height
        parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{left + width}" y2="{py:.1f}" class="hair"/>')
        parts.append(
            f'<text x="{left - 8}" y="{py + 5:.1f}" text-anchor="end" class="tick">{_tick_label(y_max * frac)}</text>'
        )
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" class="axis"/>')
    parts.append(
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" class="axis"/>'
    )
    for i, label in enumerate(labels):
        cx = left + slot * (i + 0.5)
        parts.append(
            f'<text x="{cx:.1f}" y="{top + height + 22}" text-anchor="middle" class="tick">{_esc(label)}</text>'
        )
        block = g * bar_w + (g - 1) * 4
        origin = cx - block / 2
        for j, group in enumerate(groups):
            value = group["values"][i]
            bh = 0 if y_max <= 0 else height * (value / y_max)
            x = origin + j * (bar_w + 4)
            y = top + height - bh
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="3" fill="{group["color"]}"/>'
            )
            if g == 1:
                label = f"{value:.0f}" if value >= 10 else f"{value:.2f}"
                parts.append(
                    f'<text x="{x + bar_w / 2:.1f}" y="{max(top + 12, y - 4):.1f}" text-anchor="middle" class="tick">{label}</text>'
                )
    parts.append("</svg>")
    return "".join(parts)


def _legend(items: list[tuple[str, str, bool]]) -> str:
    bits = []
    for name, color, dash in items:
        style = f"background:{color};" + ("background-image:repeating-linear-gradient(90deg,#fff 0 4px,transparent 4px 8px);" if dash else "")
        bits.append(f'<span class="key"><i style="{style}"></i>{_esc(name)}</span>')
    return f'<p class="legend">{"".join(bits)}</p>'


def _arrow(x1: float, y1: float, x2: float, y2: float, marker: str) -> str:
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5 or 1.0
    pad = 12.0
    x2 -= dx / length * pad
    y2 -= dy / length * pad
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#18181b" '
        f'stroke-width="1.6" marker-end="url(#{marker})"/>'
    )


def flow_svg(width: int, height: int, boxes: list[dict], arrows: list[tuple], marker: str) -> str:
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="flow">',
        f'<defs><marker id="{marker}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">',
        '<path d="M0,0 L8,4 L0,8 Z" fill="#18181b"/></marker></defs>',
    ]
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" '
            f'fill="#ffffff" stroke="#d4d4d8" stroke-width="1.4"/>'
        )
        parts.append(
            f'<text x="{x + w / 2}" y="{y + 32}" text-anchor="middle" class="flow-title">{_esc(box["title"])}</text>'
        )
        parts.append(
            f'<text x="{x + w / 2}" y="{y + 56}" text-anchor="middle" class="flow-sub">{_esc(box["sub"])}</text>'
        )
    for x1, y1, x2, y2 in arrows:
        parts.append(_arrow(x1, y1, x2, y2, marker))
    parts.append("</svg>")
    return "".join(parts)


def _card(title: str, note: str, body: str, wide: bool = False) -> str:
    klass = "card wide" if wide else "card"
    return (
        f'<article class="{klass}"><h3>{_esc(title)}</h3>{body}'
        f'<p class="note">{note}</p></article>'
    )


def write_report(report: dict) -> None:
    start = report["start"]
    plateau = report["plateau"]
    continued = report["continued"]
    moe_final = report["moe_final"]
    max_abs = report["max_abs"]
    linear_n = report["linear_n"]
    moe_n = report["moe_n"]
    active_n = report["active_n"]
    table: Tensor = report["table"]
    load: Tensor = report["load"]
    linear_epochs = report["linear_epochs"]
    moe_epochs = report["moe_epochs"]
    total_epochs = linear_epochs + moe_epochs

    linear_x = list(range(len(report["linear_test"])))
    extra_x = list(range(linear_epochs, total_epochs + 1))
    moe_x = list(range(linear_epochs + 1, total_epochs + 1))
    loss_svg = svg_lines(
        [
            {"xs": linear_x, "ys": report["linear_test"], "color": BLUE},
            {
                "xs": extra_x,
                "ys": [plateau, *report["continued_test"]],
                "color": BLUE,
                "dash": True,
            },
            {"xs": extra_x, "ys": [plateau, *report["moe_test"]], "color": ORANGE},
        ],
        0,
        total_epochs,
        0,
        1.55 if max(report["linear_test"]) > 1 else max(report["linear_test"]) * 1.12,
        [0, linear_epochs, total_epochs],
        [0, 0.5, 1.0, 1.5] if max(report["linear_test"]) > 1 else [0, 0.4, 0.8],
    )
    linear_fit = svg_lines(
        [
            {"xs": list(range(1, linear_epochs + 1)), "ys": report["linear_train"], "color": BLUE_SOFT},
            {"xs": linear_x[1:], "ys": report["linear_test"][1:], "color": BLUE},
            {
                "xs": moe_x,
                "ys": report["continued_train"],
                "color": BLUE_SOFT,
                "dash": True,
            },
            {"xs": moe_x, "ys": report["continued_test"], "color": BLUE, "dash": True},
        ],
        0,
        total_epochs,
        0,
        1.55 if max(report["linear_test"]) > 1 else max(report["linear_test"]) * 1.12,
        [0, linear_epochs, total_epochs],
        [0, 0.5, 1.0, 1.5] if max(report["linear_test"]) > 1 else [0, 0.4, 0.8],
    )
    moe_top = max(report["moe_train"][0], report["moe_test"][0], plateau) * 1.08
    moe_fit = svg_lines(
        [
            {"xs": moe_x, "ys": report["moe_train"], "color": ORANGE_SOFT},
            {"xs": moe_x, "ys": report["moe_test"], "color": ORANGE},
        ],
        linear_epochs,
        total_epochs,
        0,
        moe_top,
        [linear_epochs, linear_epochs + moe_epochs // 2, total_epochs],
        [0, moe_top / 2, moe_top],
    )
    y_reg = max(report["regime_linear"] + report["regime_moe"] + report["regime_linear_end"]) * 1.25
    regime_svg = svg_bars(
        [f"regime {i}" for i in range(len(report["regime_linear"]))],
        [
            {"values": report["regime_linear"], "color": BLUE},
            {"values": report["regime_linear_end"], "color": "#a1a1aa"},
            {"values": report["regime_moe"], "color": ORANGE},
        ],
        y_reg,
    )
    y_load = max(float(v) for v in load) * 1.28
    load_svg = svg_bars(
        [f"expert {i}" for i in range(load.numel())],
        [{"values": [float(v) for v in load], "color": INK}],
        y_load,
    )
    y_drift = max(report["drift"]) * 1.28
    drift_svg = svg_bars(
        [f"expert {i}" for i in range(len(report["drift"]))],
        [{"values": report["drift"], "color": ORANGE}],
        y_drift,
    )
    conf_x = [linear_epochs, *moe_x]
    conf_y = [report["confidence0"], *report["confidence"]]
    conf_svg = svg_lines(
        [{"xs": conf_x, "ys": conf_y, "color": BLUE}],
        linear_epochs,
        total_epochs,
        0,
        1,
        [linear_epochs, linear_epochs + moe_epochs // 2, total_epochs],
        [0, 0.5, 1],
    )
    aux_max = max(report["aux"]) * 1.15
    aux_svg = svg_lines(
        [{"xs": moe_x, "ys": report["aux"], "color": MUTED}],
        linear_epochs,
        total_epochs,
        0,
        aux_max,
        [linear_epochs, linear_epochs + moe_epochs // 2, total_epochs],
        [0, aux_max / 2, aux_max],
    )

    route_rows = []
    for r in range(table.shape[0]):
        expert = int(table[r].argmax())
        share = float(table[r, expert])
        route_rows.append(
            f"<tr><td>Regime {r}</td><td>Expert {expert}</td><td>{share:.0%}</td>"
            f"<td>{report['regime_linear'][r]:.3f}</td><td>{report['regime_moe'][r]:.3f}</td></tr>"
        )
    regime_bits = ", ".join(
        f"{report['regime_linear'][i]:.3f} to {report['regime_moe'][i]:.3f}"
        for i in range(len(report["regime_linear"]))
    )
    drift_bits = ", ".join(f"{v:.2f}" for v in report["drift"])
    log_html = _esc("\n".join(report["log"]))
    copy_flow = flow_svg(
        980,
        430,
        [
            {"x": 24, "y": 24, "w": 210, "h": 78, "title": "Four regimes", "sub": "each has its own map"},
            {"x": 300, "y": 24, "w": 230, "h": 78, "title": "Train one linear layer", "sub": f"test loss {plateau:.3f}"},
            {"x": 596, "y": 24, "w": 210, "h": 78, "title": "Copy the weights", "sub": f"gap {max_abs:.1e}"},
            {"x": 150, "y": 250, "w": 250, "h": 78, "title": "Keep the linear model", "sub": f"still {continued:.3f}"},
            {"x": 560, "y": 250, "w": 250, "h": 78, "title": "Train the mixture", "sub": f"down to {moe_final:.3f}"},
        ],
        [
            (234, 63, 300, 63),
            (530, 63, 596, 63),
            (701, 102, 275, 250),
            (701, 102, 685, 250),
        ],
        "arr-copy",
    )
    token_flow = flow_svg(
        980,
        140,
        [
            {"x": 16, "y": 28, "w": 140, "h": 78, "title": "Input", "sub": "one token"},
            {"x": 210, "y": 28, "w": 180, "h": 78, "title": "Router", "sub": "scores 4 experts"},
            {"x": 444, "y": 28, "w": 190, "h": 78, "title": "Top 2", "sub": "gates sum to 1"},
            {"x": 688, "y": 28, "w": 200, "h": 78, "title": "Mix", "sub": "two expert outputs"},
        ],
        [(156, 67, 210, 67), (390, 67, 444, 67), (634, 67, 688, 67)],
        "arr-token",
    )
    layout_flow = flow_svg(
        980,
        520,
        [
            {"x": 350, "y": 16, "w": 280, "h": 72, "title": "Count both sizes", "sub": "total is memory, active is compute"},
            {"x": 350, "y": 130, "w": 280, "h": 72, "title": "Experts inside one node", "sub": "EP equals the GPUs on the node"},
            {"x": 350, "y": 244, "w": 280, "h": 72, "title": "ZeRO-1 on the dense parts", "sub": "split only the optimizer state"},
            {"x": 40, "y": 390, "w": 250, "h": 78, "title": "It fits", "sub": "measure, then keep the faster one"},
            {"x": 690, "y": 390, "w": 250, "h": 78, "title": "It does not fit", "sub": "add a node"},
        ],
        [
            (490, 88, 490, 130),
            (490, 202, 490, 244),
            (420, 316, 165, 390),
            (560, 316, 815, 390),
        ],
        "arr-layout",
    )
    css = _font_css() + """
    :root { --bg:#ffffff; --fg:#18181b; --muted:#71717a; --line:#e4e4e7; }
    * { box-sizing: border-box; }
    body { margin:0; background:#fff; color:var(--fg); font-family:"Patrick Hand", cursive; font-size:21px; line-height:1.4; }
    header, main { max-width:1080px; margin:0 auto; padding:28px 20px 0; }
    main { padding-bottom:64px; }
    h1 { font-family:"Caveat", cursive; font-size:3rem; font-weight:400; line-height:1; margin:0 0 8px; }
    h2 { font-family:"Caveat", cursive; font-size:2rem; font-weight:400; margin:0 0 8px; }
    h3 { font-family:"Caveat", cursive; font-size:1.65rem; font-weight:400; margin:0 0 6px; }
    p { margin:0.4rem 0; }
    .lede { color:var(--muted); font-size:1.2rem; max-width:46rem; }
    .tabs { display:flex; flex-wrap:wrap; gap:8px; margin:18px 0 0; padding-bottom:8px; position:sticky; top:0; background:#fff; z-index:2; }
    .tab { font-family:"Caveat", cursive; font-size:1.45rem; background:#fff; color:var(--fg); border:1px solid var(--line); border-radius:999px; padding:6px 16px; cursor:pointer; }
    .tab[aria-selected="true"] { background:#18181b; color:#fff; border-color:#18181b; }
    .panel[hidden] { display:none; }
    .panel { padding-top:18px; }
    .tiles { display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; margin-top:14px; }
    .tile { border:1px solid var(--line); border-radius:12px; padding:14px 16px; background:#fff; min-height:108px; }
    .tile span { display:block; color:var(--muted); font-size:1rem; }
    .tile b { display:block; font-family:"Caveat", cursive; font-size:2rem; font-weight:400; margin-top:4px; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
    .card { border:1px solid var(--line); border-radius:12px; padding:14px 14px 10px; background:#fff; }
    .wide { grid-column:1 / -1; }
    .three { grid-template-columns:repeat(3, 1fr); }
    .note, .legend { color:var(--muted); font-size:1rem; }
    .legend { display:flex; flex-wrap:wrap; gap:8px 14px; }
    .key { display:inline-flex; align-items:center; gap:6px; }
    .key i { width:22px; height:4px; border-radius:99px; display:inline-block; }
    table { width:100%; border-collapse:collapse; border:1px solid var(--line); border-radius:12px; overflow:hidden; }
    th, td { text-align:left; padding:8px 12px; border-bottom:1px solid var(--line); }
    th { color:var(--muted); font-weight:400; background:#fafafa; }
    tr:last-child td { border-bottom:0; }
    svg.chart, svg.flow { width:100%; height:auto; display:block; }
    .tick, .flow-sub { font-family:"Patrick Hand", cursive; font-size:16px; fill:#3f3f46; }
    .flow-title { font-family:"Caveat", cursive; font-size:26px; fill:#18181b; }
    .hair { stroke:#f4f4f5; stroke-width:1; }
    .axis { stroke:#a1a1aa; stroke-width:1.25; }
    pre { white-space:pre-wrap; background:#fafafa; border:1px solid var(--line); border-radius:12px; padding:14px 16px; font-family:"Patrick Hand", cursive; font-size:16px; line-height:1.45; max-height:70vh; overflow:auto; }
    @media (max-width: 800px) {
      .tiles, .grid, .three { grid-template-columns:1fr; }
      .wide { grid-column:auto; }
      h1 { font-size:2.3rem; }
    }
    """
    charts = f"""
    <div class="grid">
    {_card("Test loss", "Blue is the linear model. Dashed blue is that same model trained onward. Orange is the mixture.", loss_svg + _legend([("Linear", BLUE, False), ("Linear, continued", BLUE, True), ("Mixture", ORANGE, False)]), wide=True)}
    {_card("Linear train and test", "Pale blue is training loss. Solid blue is test loss. Both fall, then the extra epochs sit flat.", linear_fit + _legend([("Train", BLUE_SOFT, False), ("Test", BLUE, False)]))}
    {_card("Mixture train and test", "After the copy, training loss and test loss fall together.", moe_fit + _legend([("Train", ORANGE_SOFT, False), ("Test", ORANGE, False)]))}
    {_card("Error in each regime", "Blue is the linear model at the copy. Gray is that model after more epochs. Orange is the mixture.", regime_svg + _legend([("Linear, at the copy", BLUE, False), ("Linear, later", "#a1a1aa", False), ("Mixture", ORANGE, False)]), wide=True)}
    {_card("Tokens each expert sees", f"One batch of {report['batch']} tokens, top-{report['top_k']}. The counts sum to {report['batch'] * report['top_k']}.", load_svg)}
    {_card("How far each expert moved", "Distance from the cloned weight. A taller bar means that expert changed more.", drift_svg)}
    {_card("Router confidence", "Mean of the highest router score on the test set. It rises as the router commits.", conf_svg)}
    {_card("Balance term", f"Added during mixture training, coefficient {report['aux_coef']:g}. The loss charts leave this term out.", aux_svg)}
    </div>
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Linear model, then a mixture of experts</title>
<style>
{css}
</style>
</head>
<body>
<header>
  <h1>A linear layer, copied into experts</h1>
  <p class="lede">Train one linear model until the loss flattens. Copy those weights into a mixture of experts. Keep training. The linear loss stays put. The mixture loss keeps falling.</p>
  <nav class="tabs" role="tablist">
    <button class="tab" type="button" role="tab" id="tab-overview" aria-selected="true" aria-controls="panel-overview">Overview</button>
    <button class="tab" type="button" role="tab" id="tab-flow" aria-selected="false" aria-controls="panel-flow">Flow</button>
    <button class="tab" type="button" role="tab" id="tab-results" aria-selected="false" aria-controls="panel-results">Results</button>
    <button class="tab" type="button" role="tab" id="tab-charts" aria-selected="false" aria-controls="panel-charts">Charts</button>
    <button class="tab" type="button" role="tab" id="tab-layout" aria-selected="false" aria-controls="panel-layout">Layout</button>
    <button class="tab" type="button" role="tab" id="tab-log" aria-selected="false" aria-controls="panel-log">Log</button>
  </nav>
</header>
<main>
  <section class="panel" id="panel-overview" role="tabpanel" aria-labelledby="tab-overview">
    <div class="tiles">
      <article class="tile"><span>Linear model, trained</span><b>{start:.3f} → {plateau:.3f}</b></article>
      <article class="tile"><span>At the copy, largest gap</span><b>{max_abs:.1e}</b></article>
      <article class="tile"><span>Linear model, {total_epochs} epochs</span><b>{continued:.3f}</b></article>
      <article class="tile"><span>Mixture, same extra epochs</span><b>{moe_final:.3f}</b></article>
      <article class="tile"><span>Router top score</span><b>{report["confidence0"]:.2f} → {report["confidence"][-1]:.2f}</b></article>
      <article class="tile"><span>Stored, and used per token</span><b>{moe_n:,} / {active_n:,}</b></article>
    </div>
    <p>The data has four regimes. Each regime is its own 32×32 map. One linear layer can only learn a compromise, so its test loss falls from {start:.3f} to {plateau:.3f} and then stops. Copying that layer into four experts does not change the predictions. Training the mixture lets each expert take a regime, and the test loss falls to {moe_final:.3f}.</p>
  </section>

  <section class="panel" id="panel-flow" role="tabpanel" aria-labelledby="tab-flow" hidden>
    <h2>What the run does</h2>
    <p>The top row is one linear model. The fork is the copy. The left branch keeps that same model and the same optimizer. The right branch is the mixture.</p>
    {copy_flow}
    <h2>What one token does</h2>
    <p>The router scores every expert. The two highest scores become weights that sum to 1. Those two expert outputs are added. At the moment of the copy the experts are identical, so any mix equals the original layer.</p>
    {token_flow}
  </section>

  <section class="panel" id="panel-results" role="tabpanel" aria-labelledby="tab-results" hidden>
    <h2>The numbers from this run</h2>
    <p>By regime, test error moves {regime_bits}. Expert drift from the clone is {drift_bits}. A larger drift means that expert’s weights moved farther from the linear layer they started as.</p>
    <table>
      <thead><tr><th>Checkpoint</th><th>Test MSE</th></tr></thead>
      <tbody>
        <tr><td>Linear, before training</td><td>{start:.3f}</td></tr>
        <tr><td>Linear, epoch {linear_epochs}</td><td>{plateau:.3f}</td></tr>
        <tr><td>Mixture, at the copy</td><td>{plateau:.3f}</td></tr>
        <tr><td>Linear, epoch {total_epochs}</td><td>{continued:.3f}</td></tr>
        <tr><td>Mixture, epoch {total_epochs}</td><td>{moe_final:.3f}</td></tr>
      </tbody>
    </table>
    <h2>Which expert took which regime</h2>
    <table>
      <thead><tr><th>Regime</th><th>Top expert</th><th>Share</th><th>Linear MSE</th><th>Mixture MSE</th></tr></thead>
      <tbody>{"".join(route_rows)}</tbody>
    </table>
    <p class="note">The linear layer has {linear_n:,} parameters. The mixture stores {moe_n:,} and uses {active_n:,} on each token: two experts plus the router.</p>
  </section>

  <section class="panel" id="panel-charts" role="tabpanel" aria-labelledby="tab-charts" hidden>
    <h2>Charts from this simulation</h2>
    {charts}
  </section>

  <section class="panel" id="panel-layout" role="tabpanel" aria-labelledby="tab-layout" hidden>
    <h2>How to place a large mixture</h2>
    <p>A reference model with 30.53 billion parameters shows the same idea at a size that fills GPUs. Total size is every parameter you store. At 16 bytes of training state each, that is about 455 GiB. Active size is the work per token, here 20.1 GFLOP. This page did not measure those GPU numbers. They are the reference for the diagram.</p>
    {layout_flow}
    <div class="tiles">
      <article class="tile"><span>1. Two sizes</span><b>memory, then compute</b></article>
      <article class="tile"><span>2. Expert parallelism</span><b>GPUs in one node</b></article>
      <article class="tile"><span>3. Dense optimizer</span><b>split with ZeRO-1</b></article>
      <article class="tile"><span>4. State on one B200</span><b>61.9 GiB</b></article>
      <article class="tile"><span>5. Plus one sequence</span><b>+25.5 GiB</b></article>
      <article class="tile"><span>6. Tokens per expert</span><b>a few hundred</b></article>
      <article class="tile"><span>7. All-to-all vs math</span><b>0.050 s, 0.073 s</b></article>
      <article class="tile"><span>8. Another node</span><b>only if it does not fit</b></article>
      <article class="tile"><span>9. Step time</span><b>measure, then choose</b></article>
    </div>
    <div class="grid three" style="margin-top:14px">
      <article class="card"><h3>One B200 node</h3><p>EP = 8, eight GPUs.</p><p>Training state 61.9 GiB. With one sequence, 87.4 GiB. The card has 167.6 GiB.</p><p><strong>Fits.</strong></p></article>
      <article class="card"><h3>One H100 node</h3><p>EP = 8, eight GPUs.</p><p>Same 87.4 GiB on each GPU. The card has 74.5 GiB.</p><p><strong>Does not fit.</strong></p></article>
      <article class="card"><h3>Two H100 nodes</h3><p>EP = 16, sixteen GPUs.</p><p>Training state 33.8 GiB. With one sequence, 59.3 GiB. The card has 74.5 GiB.</p><p><strong>Fits.</strong></p></article>
    </div>
    <p>Keep expert parallelism inside a node. Split the dense optimizer state. Add a node only when the state and the activations do not fit. On one node of 8 B200 GPUs the published setup for this shape is expert parallelism 8, no tensor parallelism, no pipeline parallelism, and recomputation of the expert activation and the layer norm.</p>
    <p>Open choices for a model this large: dense or mixture, how big the total and active sizes are, whether to train from scratch or grow from a dense model, how many experts and whether one is shared, and which GPU node to use. Growing from a dense model, which is what this small run does, pays off when the extra training stays under about twice the cost of that dense run. The hardware question for this shape is one B200 node, or two H100 nodes.</p>
  </section>

  <section class="panel" id="panel-log" role="tabpanel" aria-labelledby="tab-log" hidden>
    <h2>Log of this run</h2>
    <pre>{log_html}</pre>
  </section>
</main>
<script>
const tabs = [...document.querySelectorAll('[role="tab"]')];
const panels = [...document.querySelectorAll('[role="tabpanel"]')];
function show(id) {{
  tabs.forEach((tab) => {{
    const on = tab.getAttribute("aria-controls") === id;
    tab.setAttribute("aria-selected", on ? "true" : "false");
  }});
  panels.forEach((panel) => {{ panel.hidden = panel.id !== id; }});
}}
tabs.forEach((tab) => {{
  tab.addEventListener("click", () => {{
    const id = tab.getAttribute("aria-controls");
    show(id);
    history.replaceState(null, "", "#" + id.replace("panel-", ""));
  }});
}});
const start = (location.hash || "#overview").slice(1);
const initial = document.getElementById("panel-" + start);
if (initial) show(initial.id);
</script>
</body>
</html>
"""
    report["path"].write_text(html, encoding="utf-8")
