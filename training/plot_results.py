"""
Generate visualization artifacts (SVG, no external dependencies) from the
training outputs, so the project ships with real figures for GitHub / resume:

    models/loss_curve.svg     - training loss over epochs
    models/rollout_compare.svg- model's imagined trajectory vs. ground truth

    python training/plot_results.py
"""

import json
import os


def _svg_line_chart(path, xs, series, xlabel, ylabel, title, width=640, height=360):
    """series: list of (name, color, points[(x, y)])"""
    if not xs:
        return
    pad_l, pad_r, pad_t, pad_b = 56, 20, 30, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    all_y = [y for _, _, pts in series for (_, y) in pts]
    y_min, y_max = (min(all_y), max(all_y)) if all_y else (0.0, 1.0)
    if y_max - y_min < 1e-9:
        y_max = y_min + 1.0
    x_min, x_max = (min(xs), max(xs)) if xs else (0.0, 1.0)
    if x_max - x_min < 1e-9:
        x_max = x_min + 1.0

    def tx(x):
        return pad_l + (x - x_min) / (x_max - x_min) * plot_w

    def ty(y):
        return pad_t + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append(f'<rect width="{width}" height="{height}" fill="#0b1020"/>')
    parts.append(f'<text x="{width / 2}" y="20" fill="#cfe8ff" font-size="16" text-anchor="middle">{title}</text>')
    # axes
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#33415c"/>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" stroke="#33415c"/>')
    parts.append(f'<text x="14" y="{pad_t + plot_h / 2}" fill="#8aa0c0" font-size="12" transform="rotate(-90 14 {pad_t + plot_h / 2})" text-anchor="middle">{ylabel}</text>')
    parts.append(f'<text x="{pad_l + plot_w / 2}" y="{height - 8}" fill="#8aa0c0" font-size="12" text-anchor="middle">{xlabel}</text>')
    for name, color, pts in series:
        if not pts:
            continue
        d = "M " + " L ".join(f"{tx(x):.1f},{ty(y):.1f}" for x, y in pts)
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{pad_l + plot_w - 8}" y="{pad_t + 14}" fill="{color}" font-size="12" text-anchor="end">{name}</text>')
    parts.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts))


def main():
    import os
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = os.path.join(PROJECT_ROOT, "models")
    lh_path = os.path.join(base, "loss_history.json")
    rs_path = os.path.join(base, "rollout_sample.json")
    if not os.path.exists(lh_path):
        print("缺少", lh_path, "请先运行 train.py")
        return
    with open(lh_path) as f:
        lh = json.load(f)
    _svg_line_chart(
        os.path.join(base, "loss_curve.svg"),
        lh["epochs"],
        [("MSE", "#4cc9f0", list(zip(lh["epochs"], lh["loss"])))],
        "epoch", "loss (MSE, normalized)", "World Model Training Loss"
    )
    print("Saved models/loss_curve.svg")

    if os.path.exists(rs_path):
        with open(rs_path) as f:
            rs = json.load(f)
        steps = list(range(1, len(rs["true"]) + 1))
        true_pts = list(zip(steps, [p[0] for p in rs["true"]]))
        pred_pts = list(zip(steps, [p[0] for p in rs["pred"]]))
        _svg_line_chart(
            os.path.join(base, "rollout_compare.svg"),
            steps,
            [("ground truth x", "#e0e0e0", true_pts),
             ("imagined x", "#f72585", pred_pts)],
            "rollout step", "x position (px)", "Open-loop Rollout: Imagined vs. Reality"
        )
        print("Saved models/rollout_compare.svg")


if __name__ == "__main__":
    main()
