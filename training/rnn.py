"""
Latent Recurrent World Model (GRU-based, pure NumPy, manually backpropagated).

Unlike the feed-forward MLP, this model keeps a *latent hidden state* h that is
carried across time steps. The dynamics are:

    z = sigmoid(Wz @ [h, x] + bz)          # update gate
    r = sigmoid(Wr @ [h, x] + br)          # reset gate
    n = tanh(Wn @ [r*h, x] + bn)           # candidate
    h' = (1 - z) * h + z * n               # new latent state
    s' = Wout @ h' + bout                  # decoded next observation

where x = [state, action] (6-dim) and s' is the next 4-dim observation.
This is the recurrent / latent core idea behind Dreamer-style world models:
temporal latent state instead of a memoryless one-step map.
"""

import csv
import json
import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def load_rows(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(v) for k, v in r.items()})
    return rows


class GRUWorldModel:
    def __init__(self, hidden=32):
        self.hidden = hidden
        self.input_dim = 6
        self.output_dim = 4
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        self.Wz = self.Wr = self.Wn = self.Wout = None
        self.bz = self.br = self.bn = self.bout = None
        self.loss_history = []

    # ---- parameters as a plain dict (shared by forward / backward) ----
    def _params(self):
        return {
            "Wz": self.Wz, "Wr": self.Wr, "Wn": self.Wn, "Wout": self.Wout,
            "bz": self.bz, "br": self.br, "bn": self.bn, "bout": self.bout,
            "H": self.hidden, "input_dim": self.input_dim,
        }

    def _init_params(self, rng):
        d = self.input_dim
        H = self.hidden
        c = d + H
        o = self.output_dim

        def rnd(*shape):
            return rng.standard_normal(shape) * np.sqrt(2.0 / (shape[0] + shape[1]))

        self.Wz = rnd(H, c)
        self.Wr = rnd(H, c)
        self.Wn = rnd(H, c)
        self.Wout = rnd(o, H)
        self.bz = np.zeros(H)
        self.br = np.zeros(H)
        self.bn = np.zeros(H)
        self.bout = np.zeros(o)

    @staticmethod
    def _step(p, h_prev, x):
        H = p["H"]
        c = np.concatenate([h_prev, x])
        z = sigmoid(p["Wz"] @ c + p["bz"])
        r = sigmoid(p["Wr"] @ c + p["br"])
        hr = r * h_prev
        cr = np.concatenate([hr, x])
        n = np.tanh(p["Wn"] @ cr + p["bn"])
        h = (1.0 - z) * h_prev + z * n
        y = p["Wout"] @ h + p["bout"]
        return y, h, z, r, hr, cr, c, n

    def fit(self, rows, epochs=600, lr=0.01, T=16, batch_seqs=8):
        rng = np.random.default_rng(0)
        S = np.array([[r["x"], r["y"], r["vx"], r["vy"]] for r in rows])
        A = np.array([[r["ax"], r["ay"]] for r in rows])
        Snext = np.array([[r["next_x"], r["next_y"], r["next_vx"], r["next_vy"]] for r in rows])
        X = np.concatenate([S, A], axis=1)
        y = Snext
        self.X_mean = X.mean(0)
        self.X_std = X.std(0) + 1e-8
        self.y_mean = y.mean(0)
        self.y_std = y.std(0) + 1e-8
        Xn = (X - self.X_mean) / self.X_std
        yn = (y - self.y_mean) / self.y_std
        self._init_params(rng)

        N = len(rows)
        seqs = [(Xn[i:i + T], yn[i + 1:i + 1 + T]) for i in range(0, max(1, N - T))]
        if not seqs:
            seqs = [(Xn, yn)]
        H = self.hidden
        o = self.output_dim

        p = self._params()
        keys = ["Wz", "Wr", "Wn", "Wout", "bz", "br", "bn", "bout"]
        m = {k: np.zeros_like(p[k]) for k in keys}
        v = {k: np.zeros_like(p[k]) for k in keys}
        b1, b2, eps = 0.9, 0.999, 1e-8

        for e in range(epochs):
            rng.shuffle(seqs)
            total = 0.0
            nseq = 0
            for si in range(0, len(seqs), batch_seqs):
                batch = seqs[si:si + batch_seqs]
                g = {k: np.zeros_like(p[k]) for k in keys}
                loss_b = 0.0
                for (Xs, ys) in batch:
                    h = np.zeros(H)
                    Hs, Zs, Rs, HRs, CRs, Cs, Ns, Ys = [], [], [], [], [], [], [], []
                    for t in range(T):
                        yhat, h, z, r, hr, cr, c, nn = self._step(p, h, Xs[t])
                        Hs.append(h.copy())
                        Zs.append(z)
                        Rs.append(r)
                        HRs.append(hr)
                        CRs.append(cr)
                        Cs.append(c)
                        Ns.append(nn)
                        Ys.append(yhat)
                    dh = np.zeros(H)
                    for t in reversed(range(T)):
                        dy = Ys[t] - ys[t]
                        g["Wout"] += np.outer(dy, Hs[t])
                        g["bout"] += dy
                        dh = dh + p["Wout"].T @ dy
                        n = Ns[t]
                        z = Zs[t]
                        h_prev = Hs[t - 1] if t > 0 else np.zeros(H)
                        dn = dh * z
                        d_zterm = dh * (n - h_prev)
                        dh = dh * (1.0 - z)
                        dn_pre = dn * (1.0 - n * n)
                        g["Wn"] += np.outer(dn_pre, CRs[t])
                        g["bn"] += dn_pre
                        dcr = p["Wn"].T @ dn_pre
                        dhr = dcr[:H]
                        dx = dcr[H:]
                        dr = dhr * h_prev
                        dh = dh + dhr * r
                        dz_pre = d_zterm * (z * (1.0 - z))
                        g["Wz"] += np.outer(dz_pre, Cs[t])
                        g["bz"] += dz_pre
                        dc = p["Wz"].T @ dz_pre
                        dh = dh + dc[:H]
                        dx = dx + dc[H:]
                        dr_pre = dr * (r * (1.0 - r))
                        g["Wr"] += np.outer(dr_pre, Cs[t])
                        g["br"] += dr_pre
                        dc2 = p["Wr"].T @ dr_pre
                        dh = dh + dc2[:H]
                        loss_b += 0.5 * np.sum((Ys[t] - ys[t]) ** 2)
                    nseq += 1
                    total += loss_b
                for k in keys:
                    m[k] = b1 * m[k] + (1.0 - b1) * g[k]
                    v[k] = b2 * v[k] + (1.0 - b2) * (g[k] ** 2)
                    mhat = m[k] / (1.0 - b1 ** (e + 1))
                    vhat = v[k] / (1.0 - b2 ** (e + 1))
                    p[k] -= lr * mhat / (np.sqrt(vhat) + eps)
                self.Wz, self.Wr, self.Wn, self.Wout = p["Wz"], p["Wr"], p["Wn"], p["Wout"]
                self.bz, self.br, self.bn, self.bout = p["bz"], p["br"], p["bn"], p["bout"]
            self.loss_history.append(total / max(nseq, 1) / T)
            if e % 100 == 0 or e == epochs - 1:
                print(f"Epoch {e:04d} | Loss: {self.loss_history[-1]:.8f}")
        return self

    def predict(self, state, action, hidden=None):
        x = np.array(state + action, dtype=float)
        xn = (x - self.X_mean) / self.X_std
        h = np.array(hidden) if hidden is not None else np.zeros(self.hidden)
        yhat, h, _, _, _, _, _, _ = self._step(self._params(), h, xn)
        return yhat * self.y_std + self.y_mean, h

    def rollout(self, init_state, action_seq, steps, hidden=None):
        states = []
        h = np.array(hidden) if hidden is not None else np.zeros(self.hidden)
        s = np.array(init_state, dtype=float)
        for i in range(steps):
            if i < len(action_seq):
                a = action_seq[i]
            elif action_seq:
                a = action_seq[-1]
            else:
                a = [0.0, 0.0]
            s, h = self.predict(s, a, h)
            states.append(s.tolist())
        return states, h.tolist()

    def to_json(self):
        return {
            "recurrent": True,
            "hidden_size": self.hidden,
            "input_mean": self.X_mean.tolist(),
            "input_std": self.X_std.tolist(),
            "output_mean": self.y_mean.tolist(),
            "output_std": self.y_std.tolist(),
            "Wz": self.Wz.tolist(), "Wr": self.Wr.tolist(), "Wn": self.Wn.tolist(),
            "Wout": self.Wout.tolist(),
            "bz": self.bz.tolist(), "br": self.br.tolist(),
            "bn": self.bn.tolist(), "bout": self.bout.tolist(),
        }

    @classmethod
    def from_json(cls, data):
        m = cls(hidden=int(data["hidden_size"]))
        m.X_mean = np.array(data["input_mean"])
        m.X_std = np.array(data["input_std"])
        m.y_mean = np.array(data["output_mean"])
        m.y_std = np.array(data["output_std"])
        m.Wz = np.array(data["Wz"])
        m.Wr = np.array(data["Wr"])
        m.Wn = np.array(data["Wn"])
        m.Wout = np.array(data["Wout"])
        m.bz = np.array(data["bz"])
        m.br = np.array(data["br"])
        m.bn = np.array(data["bn"])
        m.bout = np.array(data["bout"])
        return m


def evaluate_rollout(rows, model, K):
    errs = []
    for i in range(len(rows) - K):
        s = [rows[i]["x"], rows[i]["y"], rows[i]["vx"], rows[i]["vy"]]
        h = None
        for step in range(K):
            a = [rows[i + step]["ax"], rows[i + step]["ay"]]
            s, h = model.predict(s, a, h)
        gt = np.array([rows[i + K]["x"], rows[i + K]["y"]])
        errs.append(float(np.linalg.norm(np.array(s[:2]) - gt)))
    return float(np.mean(errs)) if errs else 0.0


def main():
    csv_path = os.path.join(PROJECT_ROOT, "data", "trajectory.csv")
    if not os.path.exists(csv_path):
        print("训练数据不存在:", csv_path)
        return
    rows = load_rows(csv_path)
    print(f"Loaded {len(rows)} samples for recurrent (GRU) world model")

    model = GRUWorldModel(hidden=32).fit(rows, epochs=800, lr=0.01, T=16)
    model_path = os.path.join(MODELS_DIR, "world_model_rnn.json")
    with open(model_path, "w") as f:
        json.dump(model.to_json(), f, indent=2)

    roll10 = evaluate_rollout(rows, model, 10)
    roll20 = evaluate_rollout(rows, model, 20)
    metrics = {
        "type": "recurrent_gru",
        "hidden_size": model.hidden,
        "rollout_pos_err_10": roll10,
        "rollout_pos_err_20": roll20,
    }
    with open(os.path.join(MODELS_DIR, "metrics_rnn.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Recurrent world model saved to", model_path)
    print(f"Open-loop rollout err @10={roll10:.2f}px  @20={roll20:.2f}px")


if __name__ == "__main__":
    main()
