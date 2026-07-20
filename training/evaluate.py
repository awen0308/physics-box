"""
Standalone evaluation of the trained World Model.

Re-loads models/world_model.json and the collected trajectory, then reports
the open-loop rollout error at several horizons (1, 5, 10, 20, 30 steps).
Run this after training to quantify how "far" the model can imagine into
the future before its predictions diverge from reality.

    python training/evaluate.py
"""

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mlp import MLP

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    model_path = os.path.join(PROJECT_ROOT, "models", "world_model.json")
    csv_path = os.path.join(PROJECT_ROOT, "data", "trajectory.csv")
    if not os.path.exists(model_path):
        print(f"模型不存在: {model_path}")
        return
    if not os.path.exists(csv_path):
        print(f"数据不存在: {csv_path}")
        return

    with open(model_path) as f:
        mdata = json.load(f)
    model = MLP.from_json(mdata)
    X_mean = np.array(mdata["input_mean"])
    X_std = np.array(mdata["input_std"])
    y_mean = np.array(mdata["output_mean"])
    y_std = np.array(mdata["output_std"])

    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})

    def predict_single(x_raw):
        x = (np.array(x_raw, dtype=float) - X_mean) / X_std
        y = model.forward(x.reshape(1, -1)).reshape(-1)
        return y * y_std + y_mean

    def roll(K):
        errs = []
        for i in range(len(rows) - K):
            s = np.array([rows[i]["x"], rows[i]["y"], rows[i]["vx"], rows[i]["vy"]], dtype=float)
            for step in range(K):
                a = [rows[i + step]["ax"], rows[i + step]["ay"]]
                s = predict_single(np.concatenate([s, a]))[:4]
            gt = np.array([rows[i + K]["x"], rows[i + K]["y"]], dtype=float)
            errs.append(float(np.linalg.norm(s[:2] - gt)))
        return float(np.mean(errs)) if errs else 0.0

    print("Open-loop rollout position error (lower = better):")
    for K in [1, 5, 10, 20, 30]:
        if len(rows) > K:
            print(f"  {K:2d} steps: {roll(K):.2f} px")


if __name__ == "__main__":
    main()
