"""
Comprehensive Meta-Selector Report Generator
=============================================
Generates CDF plots, per-timestep oracle tables, and proof
that the Meta-Selector is working correctly.
"""
import sys, os, json, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

from phase1_reactive.eval.common import (
    load_bundle, collect_specs, load_named_dataset,
    max_steps_from_args, resolve_phase1_k_crit,
)
from phase1_reactive.env.offline_env import ReactiveRoutingEnv, ReactiveEnvConfig
from phase1_reactive.drl.moe_features import (
    bottleneck_scores, sensitivity_scores, topk_from_scores,
)
from phase1_reactive.drl.gnn_inference import load_gnn_selector
from phase1_reactive.drl.meta_selector import build_meta_features, MetaGate

CONFIG = "configs/phase1_reactive_full.yaml"
REPORT_DIR = Path("results/meta_eval/report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ═══════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════
print("=" * 70)
print("META-SELECTOR COMPREHENSIVE REPORT")
print("=" * 70)

bundle = load_bundle(CONFIG)
max_steps = max_steps_from_args(bundle, 500)

gnn_path = Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt")
gnn_model, gnn_cfg = load_gnn_selector(gnn_path, device="cpu")
gnn_model.eval()
print("  GNN loaded.")

# Load all topologies
ALL_TOPO_KEYS = [
    "abilene_backbone", "geant_core", "ebone", "cernet",
    "sprintlink", "tiscali", "germany50_real", "vtlwavenet2011",
]

topo_data = {}
for spec_key in ALL_TOPO_KEYS:
    if spec_key not in bundle.registry:
        print(f"  SKIP {spec_key}: not in registry")
        continue
    spec = bundle.registry[spec_key]
    try:
        ds, pl = load_named_dataset(bundle, spec, max_steps)
        k_crit = resolve_phase1_k_crit(bundle, ds)
        topo_data[ds.key] = (ds, pl, k_crit, spec)
        print(f"  Loaded: {ds.key} ({len(ds.nodes)}N, {len(ds.edges)}E)")
    except Exception as e:
        print(f"  FAILED {spec_key}: {e}")

ALL_TOPOS = list(topo_data.keys())
print(f"\n  Total topologies: {len(ALL_TOPOS)}")
print(f"  Topologies: {ALL_TOPOS}")

UNSEEN_FOLDS = {
    "fold1": {"unseen": ["germany50", "cernet", "topologyzoo_vtlwavenet2011"]},
    "fold2": {"unseen": ["germany50", "rocketfuel_sprintlink", "rocketfuel_tiscali"]},
}
for fold_name, fold_cfg in UNSEEN_FOLDS.items():
    fold_cfg["known"] = [t for t in ALL_TOPOS if t not in fold_cfg["unseen"]]

def make_env(ds, pl, k_crit, split_name):
    cfg = ReactiveEnvConfig(k_crit=k_crit, lp_time_limit_sec=20)
    return ReactiveRoutingEnv(dataset=ds, tm_data=ds.tm, path_library=pl,
                              split_name=split_name, cfg=cfg)

def _gnn_strict_scores(env, device="cpu"):
    from phase1_reactive.drl.gnn_selector import build_graph_tensors, build_od_features
    obs = env.current_obs
    graph_data = build_graph_tensors(env.dataset, telemetry=getattr(obs, 'telemetry', None), device=device)
    od_data = build_od_features(env.dataset, obs.current_tm, env.path_library,
                                 telemetry=getattr(obs, 'telemetry', None), device=device)
    with torch.no_grad():
        scores_t, k_pred, info = gnn_model.forward(graph_data, od_data)
    return scores_t.cpu().numpy().flatten()


# ═══════════════════════════════════════════════════════
# STEP 1: Collect per-timestep BN vs GNN MLU for ALL topologies
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: Per-Timestep Oracle Collection (ALL topologies, val split)")
print("=" * 70)

from te.lp_solver import solve_selected_path_lp

N_STEPS = 50  # per topology

all_oracle_data = {}  # topo -> list of dicts with bn_mlu, gnn_mlu, features, best

for topo_key in ALL_TOPOS:
    ds, pl, k_crit, spec = topo_data[topo_key]
    env = make_env(ds, pl, k_crit, "val")
    obs = env.reset()
    samples = []
    done = False

    for step_i in range(N_STEPS):
        if done:
            break
        tm = obs.current_tm
        tel = getattr(obs, 'telemetry', None)
        features = build_meta_features(ds, tm, tel)

        # BN selection
        bn_scores = bottleneck_scores(tm, env.ecmp_base, env.path_library, env.capacities)
        bn_sel = topk_from_scores(bn_scores, obs.active_mask, k_crit)

        # GNN selection
        gnn_scores = _gnn_strict_scores(env)
        gnn_sel = topk_from_scores(gnn_scores, obs.active_mask, k_crit)

        # LP for both
        mlus = {}
        for name, sel in [("bn", bn_sel), ("gnn", gnn_sel)]:
            try:
                lp = solve_selected_path_lp(tm, sel, env.ecmp_base, pl, env.capacities, time_limit_sec=15)
                mlus[name] = float(lp.routing.mlu)
            except:
                mlus[name] = float("inf")

        best = "gnn" if mlus["gnn"] < mlus["bn"] else "bn"
        advantage = (mlus["bn"] - mlus["gnn"]) / mlus["bn"] * 100 if mlus["bn"] > 0 else 0

        samples.append({
            "step": step_i,
            "bn_mlu": mlus["bn"],
            "gnn_mlu": mlus["gnn"],
            "best": best,
            "gnn_advantage_pct": advantage,
            "features": features,
        })

        next_obs, _, done, _ = env.step(bn_sel)
        obs = next_obs

    all_oracle_data[topo_key] = samples
    n_bn = sum(1 for s in samples if s["best"] == "bn")
    n_gnn = sum(1 for s in samples if s["best"] == "gnn")
    avg_adv = np.mean([s["gnn_advantage_pct"] for s in samples])
    print(f"  {topo_key:35s} {len(samples)} steps  BN={n_bn}  GNN={n_gnn}  "
          f"GNN%={n_gnn/len(samples)*100:5.1f}%  avg_adv={avg_adv:+.3f}%")


# ═══════════════════════════════════════════════════════
# STEP 2: Train gate per fold and evaluate
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Train MetaGate per Fold + Evaluate")
print("=" * 70)

GATE_EXPERTS = ["bn", "gnn"]

def train_gate(samples):
    name_to_idx = {"bn": 0, "gnn": 1}
    X = torch.tensor(np.stack([s["features"] for s in samples]), dtype=torch.float32)
    y = torch.tensor([name_to_idx[s["best"]] for s in samples], dtype=torch.long)

    n_val = max(1, len(X) // 5)
    perm = torch.randperm(len(X))
    X_val, y_val = X[perm[:n_val]], y[perm[:n_val]]
    X_tr, y_tr = X[perm[n_val:]], y[perm[n_val:]]

    model = MetaGate(input_dim=X.shape[1], num_experts=2, hidden_dim=64)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    counts = np.bincount(y_tr.numpy(), minlength=2).astype(np.float32)
    weights = 1.0 / (counts + 1.0)
    weights = weights / weights.sum() * 2
    cw = torch.tensor(weights, dtype=torch.float32)

    best_acc, best_state, no_imp = 0.0, None, 0
    for ep in range(1, 301):
        model.train()
        loss = F.cross_entropy(model(X_tr), y_tr, weight=cw)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(X_val).argmax(1) == y_val).float().mean())
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= 40:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_acc


def evaluate_gate_per_timestep(gate, topo_key, split="test"):
    """Run gate per-timestep on a topology, return per-step BN/GNN/Gate MLUs."""
    ds, pl, k_crit, spec = topo_data[topo_key]
    env = make_env(ds, pl, k_crit, split)
    obs = env.reset()
    rows = []
    done = False

    for step_i in range(N_STEPS):
        if done:
            break
        tm = obs.current_tm
        tel = getattr(obs, 'telemetry', None)
        features = build_meta_features(ds, tm, tel)

        # Gate prediction
        gate.eval()
        with torch.no_grad():
            logits = gate(torch.tensor(features, dtype=torch.float32).unsqueeze(0))
            pred_idx = int(logits.argmax(1).item())
            pred = "gnn" if pred_idx == 1 else "bn"
            conf = float(F.softmax(logits, dim=1)[0, pred_idx].item())

        # Both selections
        bn_scores = bottleneck_scores(tm, env.ecmp_base, env.path_library, env.capacities)
        bn_sel = topk_from_scores(bn_scores, obs.active_mask, k_crit)
        gnn_scores = _gnn_strict_scores(env)
        gnn_sel = topk_from_scores(gnn_scores, obs.active_mask, k_crit)

        # LP for both
        mlus = {}
        for name, sel in [("bn", bn_sel), ("gnn", gnn_sel)]:
            try:
                lp = solve_selected_path_lp(tm, sel, env.ecmp_base, pl, env.capacities, time_limit_sec=15)
                mlus[name] = float(lp.routing.mlu)
            except:
                mlus[name] = float("inf")

        gate_mlu = mlus[pred]
        oracle_mlu = min(mlus["bn"], mlus["gnn"])
        oracle_expert = "gnn" if mlus["gnn"] < mlus["bn"] else "bn"

        rows.append({
            "step": step_i,
            "bn_mlu": mlus["bn"],
            "gnn_mlu": mlus["gnn"],
            "gate_pred": pred,
            "gate_mlu": gate_mlu,
            "oracle_mlu": oracle_mlu,
            "oracle_expert": oracle_expert,
            "gate_correct": pred == oracle_expert,
            "confidence": conf,
        })

        next_obs, _, done, _ = env.step(bn_sel)
        obs = next_obs

    return pd.DataFrame(rows)


# Run per fold
fold_results = {}

for fold_name, fold_cfg in UNSEEN_FOLDS.items():
    known = fold_cfg["known"]
    unseen = fold_cfg["unseen"]
    print(f"\n{'─'*60}")
    print(f"  {fold_name}: known={known}, unseen={unseen}")

    # Collect training samples from known topologies
    train_samples = []
    for t in known:
        train_samples.extend(all_oracle_data[t])
    n_bn = sum(1 for s in train_samples if s["best"] == "bn")
    n_gnn = sum(1 for s in train_samples if s["best"] == "gnn")
    print(f"  Training: {len(train_samples)} samples (BN={n_bn}, GNN={n_gnn})")

    gate, acc = train_gate(train_samples)
    print(f"  Gate val_acc: {acc:.3f}")

    # Evaluate on ALL topologies (test split)
    fold_eval = {}
    for topo_key in ALL_TOPOS:
        is_unseen = topo_key in unseen
        tag = "[UNSEEN]" if is_unseen else "[KNOWN]"
        df = evaluate_gate_per_timestep(gate, topo_key, "test")
        fold_eval[topo_key] = {
            "df": df,
            "is_unseen": is_unseen,
            "gate_mlu": df["gate_mlu"].mean(),
            "bn_mlu": df["bn_mlu"].mean(),
            "gnn_mlu": df["gnn_mlu"].mean(),
            "oracle_mlu": df["oracle_mlu"].mean(),
            "gate_accuracy": df["gate_correct"].mean() * 100,
            "pct_gnn": (df["gate_pred"] == "gnn").mean() * 100,
            "n_gnn_oracle": (df["oracle_expert"] == "gnn").sum(),
            "n_bn_oracle": (df["oracle_expert"] == "bn").sum(),
            "n_gnn_gate": (df["gate_pred"] == "gnn").sum(),
            "n_bn_gate": (df["gate_pred"] == "bn").sum(),
        }
        r = fold_eval[topo_key]
        regret = (r["gate_mlu"] - r["oracle_mlu"]) / r["oracle_mlu"] * 100 if r["oracle_mlu"] > 0 else 0
        r["gate_regret"] = regret
        r["bn_regret"] = (r["bn_mlu"] - r["oracle_mlu"]) / r["oracle_mlu"] * 100 if r["oracle_mlu"] > 0 else 0
        r["gnn_regret"] = (r["gnn_mlu"] - r["oracle_mlu"]) / r["oracle_mlu"] * 100 if r["oracle_mlu"] > 0 else 0

        print(f"  {tag:8s} {topo_key:35s} gate_acc={r['gate_accuracy']:5.1f}%  "
              f"BN_reg={r['bn_regret']:.3f}%  GNN_reg={r['gnn_regret']:.3f}%  "
              f"Gate_reg={r['gate_regret']:.3f}%  picks:BN={r['n_bn_gate']}/GNN={r['n_gnn_gate']}")

    fold_results[fold_name] = fold_eval


# ═══════════════════════════════════════════════════════
# STEP 3: Generate CDF Plots
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Generating CDF Plots and Tables")
print("=" * 70)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

# --- Fig 1: Per-topology CDF of per-timestep MLU (BN vs GNN vs Oracle) ---
for fold_name in UNSEEN_FOLDS:
    fold_eval = fold_results[fold_name]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"CDF of Per-Timestep MLU — {fold_name}\n(BN vs GNN vs MetaGate vs Oracle)", fontsize=15, fontweight="bold")

    for idx, topo_key in enumerate(ALL_TOPOS):
        ax = axes[idx // 4, idx % 4]
        r = fold_eval[topo_key]
        df = r["df"]

        # CDF
        for col, label, color, ls in [
            ("bn_mlu", "Bottleneck", "#2196F3", "-"),
            ("gnn_mlu", "GNN", "#FF5722", "-"),
            ("gate_mlu", "MetaGate", "#4CAF50", "--"),
            ("oracle_mlu", "Oracle", "#9C27B0", ":"),
        ]:
            vals = np.sort(df[col].values)
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf, label=label, color=color, linestyle=ls, linewidth=2)

        unseen_tag = " [UNSEEN]" if r["is_unseen"] else ""
        ax.set_title(f"{topo_key}{unseen_tag}\nacc={r['gate_accuracy']:.0f}% GNN%={r['pct_gnn']:.0f}%", fontsize=9)
        ax.set_xlabel("MLU")
        ax.set_ylabel("CDF")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / f"cdf_per_topology_{fold_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# --- Fig 2: Per-Timestep Oracle Expert Distribution (Stacked Bar) ---
fig, ax = plt.subplots(figsize=(14, 6))
topos = ALL_TOPOS
bn_counts = [sum(1 for s in all_oracle_data[t] if s["best"] == "bn") for t in topos]
gnn_counts = [sum(1 for s in all_oracle_data[t] if s["best"] == "gnn") for t in topos]
x = np.arange(len(topos))
w = 0.6

bars1 = ax.bar(x, bn_counts, w, label="Bottleneck Wins", color="#2196F3", edgecolor="white")
bars2 = ax.bar(x, gnn_counts, w, bottom=bn_counts, label="GNN Wins", color="#FF5722", edgecolor="white")

for i, (b, g) in enumerate(zip(bn_counts, gnn_counts)):
    total = b + g
    pct = g / total * 100
    ax.text(i, total + 0.5, f"{pct:.0f}%", ha="center", fontsize=9, fontweight="bold", color="#FF5722")

ax.set_xticks(x)
ax.set_xticklabels([t.replace("rocketfuel_", "rf_").replace("topologyzoo_", "tz_") for t in topos],
                    rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Number of Timesteps (out of 50)")
ax.set_title("Per-Timestep Oracle: Which Expert Wins at Each Timestep?\n(Bottleneck vs GNN, k=40, LP-verified)", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
path = FIG_DIR / "oracle_expert_distribution.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")


# --- Fig 3: Regret Comparison Bar Chart (BN-only vs GNN-only vs MetaGate) ---
for fold_name in UNSEEN_FOLDS:
    fold_eval = fold_results[fold_name]
    fig, ax = plt.subplots(figsize=(14, 6))

    bn_regs = [fold_eval[t]["bn_regret"] for t in ALL_TOPOS]
    gnn_regs = [fold_eval[t]["gnn_regret"] for t in ALL_TOPOS]
    gate_regs = [fold_eval[t]["gate_regret"] for t in ALL_TOPOS]

    x = np.arange(len(ALL_TOPOS))
    w = 0.25

    ax.bar(x - w, bn_regs, w, label="BN-only Regret", color="#2196F3", edgecolor="white")
    ax.bar(x, gnn_regs, w, label="GNN-only Regret", color="#FF5722", edgecolor="white")
    ax.bar(x + w, gate_regs, w, label="MetaGate Regret", color="#4CAF50", edgecolor="white")

    # Mark unseen
    for i, t in enumerate(ALL_TOPOS):
        if fold_eval[t]["is_unseen"]:
            ax.text(i, max(bn_regs[i], gnn_regs[i], gate_regs[i]) + 0.05, "UNSEEN",
                    ha="center", fontsize=7, color="red", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("rocketfuel_", "rf_").replace("topologyzoo_", "tz_") for t in ALL_TOPOS],
                        rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Regret vs Oracle (%)")
    ax.set_title(f"Regret Comparison: BN-only vs GNN-only vs MetaGate — {fold_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = FIG_DIR / f"regret_comparison_{fold_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# --- Fig 4: Gate Accuracy per Topology ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for fi, fold_name in enumerate(UNSEEN_FOLDS):
    ax = axes[fi]
    fold_eval = fold_results[fold_name]
    accs = [fold_eval[t]["gate_accuracy"] for t in ALL_TOPOS]
    colors = ["#E91E63" if fold_eval[t]["is_unseen"] else "#4CAF50" for t in ALL_TOPOS]
    bars = ax.bar(range(len(ALL_TOPOS)), accs, color=colors, edgecolor="white")
    ax.set_xticks(range(len(ALL_TOPOS)))
    ax.set_xticklabels([t.replace("rocketfuel_", "rf_").replace("topologyzoo_", "tz_") for t in ALL_TOPOS],
                        rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Gate Accuracy (%)")
    ax.set_title(f"MetaGate Accuracy Per Topology — {fold_name}", fontsize=12, fontweight="bold")
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(accs):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
plt.tight_layout()
path = FIG_DIR / "gate_accuracy.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")


# --- Fig 5: GNN Advantage Distribution (Histogram) ---
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Distribution of GNN Advantage Over Bottleneck (per timestep, %)\n"
             "Positive = GNN better, Negative = BN better", fontsize=14, fontweight="bold")
for idx, topo_key in enumerate(ALL_TOPOS):
    ax = axes[idx // 4, idx % 4]
    advs = [s["gnn_advantage_pct"] for s in all_oracle_data[topo_key]]
    ax.hist(advs, bins=20, color="#9C27B0", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    mean_adv = np.mean(advs)
    ax.axvline(x=mean_adv, color="green", linestyle="-", linewidth=2, label=f"mean={mean_adv:+.3f}%")
    n_gnn_wins = sum(1 for a in advs if a > 0)
    ax.set_title(f"{topo_key}\nGNN wins {n_gnn_wins}/{len(advs)} timesteps", fontsize=9)
    ax.set_xlabel("GNN Advantage (%)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
path = FIG_DIR / "gnn_advantage_distribution.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════
# STEP 4: Summary Tables
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY TABLES")
print("=" * 70)

# Table A: Per-Timestep Oracle Summary
print("\n--- Table A: Per-Timestep Oracle Expert Distribution ---")
print(f"{'Topology':35s} {'Nodes':>5s} {'BN':>4s} {'GNN':>4s} {'GNN%':>6s} {'Mean Adv':>9s}")
print("─" * 70)
for t in ALL_TOPOS:
    ds = topo_data[t][0]
    samples = all_oracle_data[t]
    n_bn = sum(1 for s in samples if s["best"] == "bn")
    n_gnn = sum(1 for s in samples if s["best"] == "gnn")
    avg_adv = np.mean([s["gnn_advantage_pct"] for s in samples])
    print(f"{t:35s} {len(ds.nodes):5d} {n_bn:4d} {n_gnn:4d} {n_gnn/len(samples)*100:5.1f}% {avg_adv:+8.3f}%")

# Table B: Per-Fold Regret Comparison
for fold_name in UNSEEN_FOLDS:
    print(f"\n--- Table B: Regret Comparison — {fold_name} ---")
    fold_eval = fold_results[fold_name]
    print(f"{'Topology':35s} {'U':>1s} {'BN_reg':>7s} {'GNN_reg':>8s} {'Gate_reg':>9s} {'Gate_acc':>9s} {'BN':>3s} {'GNN':>3s}")
    print("─" * 85)
    for t in ALL_TOPOS:
        r = fold_eval[t]
        u = "U" if r["is_unseen"] else ""
        print(f"{t:35s} {u:>1s} {r['bn_regret']:7.3f}% {r['gnn_regret']:7.3f}% "
              f"{r['gate_regret']:8.3f}% {r['gate_accuracy']:8.1f}% "
              f"{r['n_bn_gate']:3d} {r['n_gnn_gate']:3d}")

    # Averages
    all_gate = np.mean([fold_eval[t]["gate_regret"] for t in ALL_TOPOS])
    all_bn = np.mean([fold_eval[t]["bn_regret"] for t in ALL_TOPOS])
    all_gnn = np.mean([fold_eval[t]["gnn_regret"] for t in ALL_TOPOS])
    unseen_topos = [t for t in ALL_TOPOS if fold_eval[t]["is_unseen"]]
    u_gate = np.mean([fold_eval[t]["gate_regret"] for t in unseen_topos])
    u_bn = np.mean([fold_eval[t]["bn_regret"] for t in unseen_topos])
    u_gnn = np.mean([fold_eval[t]["gnn_regret"] for t in unseen_topos])
    print(f"{'AVERAGE (all)':35s}   {all_bn:7.3f}% {all_gnn:7.3f}% {all_gate:8.3f}%")
    print(f"{'AVERAGE (unseen)':35s}   {u_bn:7.3f}% {u_gnn:7.3f}% {u_gate:8.3f}%")


# Save everything to CSV
oracle_rows = []
for t in ALL_TOPOS:
    for s in all_oracle_data[t]:
        oracle_rows.append({
            "topology": t,
            "step": s["step"],
            "bn_mlu": s["bn_mlu"],
            "gnn_mlu": s["gnn_mlu"],
            "best_expert": s["best"],
            "gnn_advantage_pct": s["gnn_advantage_pct"],
        })
pd.DataFrame(oracle_rows).to_csv(REPORT_DIR / "per_timestep_oracle.csv", index=False)
print(f"\n  Saved: {REPORT_DIR / 'per_timestep_oracle.csv'}")

results_rows = []
for fold_name in UNSEEN_FOLDS:
    for t in ALL_TOPOS:
        r = fold_results[fold_name][t]
        results_rows.append({
            "fold": fold_name,
            "topology": t,
            "is_unseen": r["is_unseen"],
            "bn_mlu": r["bn_mlu"],
            "gnn_mlu": r["gnn_mlu"],
            "gate_mlu": r["gate_mlu"],
            "oracle_mlu": r["oracle_mlu"],
            "bn_regret_pct": r["bn_regret"],
            "gnn_regret_pct": r["gnn_regret"],
            "gate_regret_pct": r["gate_regret"],
            "gate_accuracy_pct": r["gate_accuracy"],
            "n_bn_gate": r["n_bn_gate"],
            "n_gnn_gate": r["n_gnn_gate"],
        })
pd.DataFrame(results_rows).to_csv(REPORT_DIR / "gate_evaluation_results.csv", index=False)
print(f"  Saved: {REPORT_DIR / 'gate_evaluation_results.csv'}")

print(f"\nAll report files saved to: {REPORT_DIR}/")
print("Done.")
