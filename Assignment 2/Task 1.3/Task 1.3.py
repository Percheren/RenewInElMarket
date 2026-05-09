"""
Task 1.3 - Ex-Post Analysis via 8-Fold Cross-Validation
========================================================
Split 1600 scenarios into 8 folds of 200 each.
For each fold:
  - in-sample  : 200 scenarios (1 fold)
  - out-of-sample: 1400 scenarios (remaining 7 folds)

Solve both the one-price and two-price models on the in-sample set,
then evaluate the resulting p_DA decisions on the out-of-sample set.
Report averaged in-sample vs out-of-sample expected profits for both models.
"""

from scipy.stats import bernoulli
import gurobipy as gp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0.  Shared parameters
# ─────────────────────────────────────────────
T        = 24
capacity = 500          # MW
N_wind   = 20
N_price  = 20
N_si     = 4
W        = N_wind * N_price * N_si   # 1600
n_folds  = 8
fold_size = W // n_folds             # 200 per fold

# ─────────────────────────────────────────────
# 1.  Load / generate scenarios  (same as 1.1/1.2)
# ─────────────────────────────────────────────
DA_price = pd.read_csv(r"C:\Users\GioBr\Desktop\DTU\RIEM - Renewables in Electricity Market\Assignment 2\Day-Ahead-Price-01_03_26-20_30_26.csv", sep=";")
DA_price["Eur/MWh"] = DA_price["Eur/MWh"].str.replace(',', '.').astype(float)
prices           = DA_price["Eur/MWh"].values[:24 * N_price]
price_scenarios  = prices.reshape(N_price, T)

wind_prod      = pd.read_excel(r"C:\Users\GioBr\Desktop\DTU\RIEM - Renewables in Electricity Market\Assignment 2\ninja-wind-country-DK-current_onshore-merra2.xlsx", header=3)
wind_scenarios = wind_prod["DK02_factor [1]"].values[:24 * N_wind] * capacity
wind_scenarios = wind_scenarios.reshape(N_wind, T)

np.random.seed(42)
si_scenarios = bernoulli.rvs(0.5, size=(N_si, T))

# Build full scenario arrays  (shape: W × T)
real_prod = np.zeros((W, T))
DA_lambda = np.zeros((W, T))
SI        = np.zeros((W, T))

w = 0
for i in range(N_wind):
    for j in range(N_price):
        for k in range(N_si):
            real_prod[w] = wind_scenarios[i]
            DA_lambda[w] = price_scenarios[j]
            SI[w]        = si_scenarios[k]
            w += 1

lambda_bal  = np.where(SI == 1, 1.25 * DA_lambda, 0.85 * DA_lambda)
lambda_up   = 1.25 * DA_lambda
lambda_down = 0.85 * DA_lambda

# ─────────────────────────────────────────────
# 2.  Helper: compute profit for a fixed p_DA
#     under the one-price settlement
# ─────────────────────────────────────────────
def compute_profit_1price(p_DA_sol, idx):
    """
    p_DA_sol : array (T,) with DA offers
    idx      : scenario indices to evaluate over
    returns  : expected profit (scalar) and per-scenario profits (array)
    """
    pi = 1 / len(idx)
    profits = np.sum(
        DA_lambda[idx] * p_DA_sol
        + lambda_bal[idx] * (real_prod[idx] - p_DA_sol),
        axis=1
    )
    return pi * profits.sum(), profits


def compute_profit_2price(p_DA_sol, idx):
    """Two-price settlement using the same deviation logic as Model 1.2."""
    delta_up   = np.maximum(real_prod[idx] - p_DA_sol, 0)
    delta_down = np.maximum(p_DA_sol - real_prod[idx], 0)

    revenues = (
        DA_lambda[idx] * p_DA_sol
        + delta_up * (
            SI[idx] * lambda_up[idx] + (1 - SI[idx]) * DA_lambda[idx]
        )
        - delta_down * (
            SI[idx] * DA_lambda[idx] + (1 - SI[idx]) * lambda_down[idx]
        )
    )
    profits = revenues.sum(axis=1)
    pi = 1 / len(idx)
    return pi * profits.sum(), profits


# ─────────────────────────────────────────────
# 3.  Helper: solve one-price model on a subset
# ─────────────────────────────────────────────
def solve_1price(idx):
    pi = 1 / len(idx)
    m  = gp.Model()
    m.setParam("OutputFlag", 0)

    p_DA = m.addVars(T, lb=0, ub=capacity)

    m.setObjective(
        gp.quicksum(
            pi * (
                DA_lambda[w, t] * p_DA[t]
                + lambda_bal[w, t] * (real_prod[w, t] - p_DA[t])
            )
            for w in idx for t in range(T)
        ),
        gp.GRB.MAXIMIZE,
    )
    m.optimize()
    return np.array([p_DA[t].X for t in range(T)])


# ─────────────────────────────────────────────
# 4.  Helper: solve two-price model on a subset
# ─────────────────────────────────────────────
def solve_2price(idx):
    pi   = 1 / len(idx)
    n    = len(idx)
    m    = gp.Model()
    m.setParam("OutputFlag", 0)

    p_DA      = m.addVars(T, lb=0, ub=capacity)
    Delta_up  = m.addVars(n, T, lb=0, ub=capacity)
    Delta_down = m.addVars(n, T, lb=0, ub=capacity)

    for ii, w in enumerate(idx):
        for t in range(T):
            m.addConstr(Delta_up[ii, t]   >= real_prod[w, t] - p_DA[t])
            m.addConstr(Delta_down[ii, t] >= p_DA[t] - real_prod[w, t])

    m.setObjective(
        gp.quicksum(
            pi * (
                DA_lambda[idx[ii], t] * p_DA[t]
                + Delta_up[ii, t] * (
                    SI[idx[ii], t] * lambda_up[idx[ii], t]
                    + (1 - SI[idx[ii], t]) * DA_lambda[idx[ii], t]
                )
                - Delta_down[ii, t] * (
                    SI[idx[ii], t] * DA_lambda[idx[ii], t]
                    + (1 - SI[idx[ii], t]) * lambda_down[idx[ii], t]
                )
            )
            for ii in range(n) for t in range(T)
        ),
        gp.GRB.MAXIMIZE,
    )
    m.optimize()
    return np.array([p_DA[t].X for t in range(T)])


# ─────────────────────────────────────────────
# 5.  8-Fold Cross-Validation
# ─────────────────────────────────────────────
all_idx = np.arange(W)

# Storage: rows = folds, cols = [in-sample EP, OOS EP]
results_1p = np.zeros((n_folds, 2))
results_2p = np.zeros((n_folds, 2))

# Also keep per-fold decisions for optional inspection
decisions_1p = []
decisions_2p = []

print(f"{'Fold':>5}  {'1P IS EP':>10}  {'1P OOS EP':>10}  "
      f"{'2P IS EP':>10}  {'2P OOS EP':>10}")
print("-" * 55)

for fold in range(n_folds):
    # in-sample: current fold (200 scenarios)
    is_idx  = all_idx[fold * fold_size : (fold + 1) * fold_size]
    # out-of-sample: all other folds (1400 scenarios)
    oos_idx = np.concatenate([
        all_idx[f * fold_size : (f + 1) * fold_size]
        for f in range(n_folds) if f != fold
    ])

    # ── One-price ──
    p1 = solve_1price(is_idx)
    decisions_1p.append(p1)
    ep_is_1p,  _ = compute_profit_1price(p1, is_idx)
    ep_oos_1p, _ = compute_profit_1price(p1, oos_idx)
    results_1p[fold] = [ep_is_1p, ep_oos_1p]

    # ── Two-price ──
    p2 = solve_2price(is_idx)
    decisions_2p.append(p2)
    ep_is_2p,  _ = compute_profit_2price(p2, is_idx)
    ep_oos_2p, _ = compute_profit_2price(p2, oos_idx)
    results_2p[fold] = [ep_is_2p, ep_oos_2p]

    print(f"{fold+1:>5}  {ep_is_1p:>10.0f}  {ep_oos_1p:>10.0f}  "
          f"{ep_is_2p:>10.0f}  {ep_oos_2p:>10.0f}")

print("-" * 55)
print(f"{'Mean':>5}  "
      f"{results_1p[:,0].mean():>10.0f}  {results_1p[:,1].mean():>10.0f}  "
      f"{results_2p[:,0].mean():>10.0f}  {results_2p[:,1].mean():>10.0f}")

# ─────────────────────────────────────────────
# 6.  Plots
# ─────────────────────────────────────────────
fold_labels = [f"Fold {i+1}" for i in range(n_folds)]
x = np.arange(n_folds)
width = 0.35

# ── Figure 1: In-sample vs OOS expected profit per fold (One-Price) ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.bar(x - width/2, results_1p[:, 0], width, label="In-sample",
       color="steelblue", alpha=0.85)
ax.bar(x + width/2, results_1p[:, 1], width, label="Out-of-sample",
       color="coral", alpha=0.85)
ax.axhline(results_1p[:, 0].mean(), color="steelblue", linestyle="--",
           linewidth=1, label=f"Avg IS = {results_1p[:,0].mean():.0f} €")
ax.axhline(results_1p[:, 1].mean(), color="coral", linestyle="--",
           linewidth=1, label=f"Avg OOS = {results_1p[:,1].mean():.0f} €")
ax.set_xticks(x)
ax.set_xticklabels(fold_labels, rotation=30)
ax.set_ylabel("Expected Profit (€)")
ax.set_title("Task 1.3 – One-Price: IS vs OOS Expected Profit")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)

# ── Figure 2: In-sample vs OOS expected profit per fold (Two-Price) ──
ax = axes[1]
ax.bar(x - width/2, results_2p[:, 0], width, label="In-sample",
       color="steelblue", alpha=0.85)
ax.bar(x + width/2, results_2p[:, 1], width, label="Out-of-sample",
       color="coral", alpha=0.85)
ax.axhline(results_2p[:, 0].mean(), color="steelblue", linestyle="--",
           linewidth=1, label=f"Avg IS = {results_2p[:,0].mean():.0f} €")
ax.axhline(results_2p[:, 1].mean(), color="coral", linestyle="--",
           linewidth=1, label=f"Avg OOS = {results_2p[:,1].mean():.0f} €")
ax.set_xticks(x)
ax.set_xticklabels(fold_labels, rotation=30)
ax.set_ylabel("Expected Profit (€)")
ax.set_title("Task 1.3 – Two-Price: IS vs OOS Expected Profit")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("task_1_3_cv_profits.png", dpi=150)
plt.show()

# ── Figure 3: DA offer profiles across folds (both models) ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
hours = np.arange(1, T + 1)

for fold in range(n_folds):
    axes[0].plot(hours, decisions_1p[fold], alpha=0.5, linewidth=0.9)
axes[0].set_title("Task 1.3 – One-Price: DA Offers Across Folds")
axes[0].set_xlabel("Hour")
axes[0].set_ylabel("DA Offer (MW)")
axes[0].set_xlim(1, T)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[0].grid(linestyle="--", alpha=0.3)

for fold in range(n_folds):
    axes[1].plot(hours, decisions_2p[fold], alpha=0.5, linewidth=0.9)
axes[1].set_title("Task 1.3 – Two-Price: DA Offers Across Folds")
axes[1].set_xlabel("Hour")
axes[1].set_ylabel("DA Offer (MW)")
axes[1].set_xlim(1, T)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
axes[1].grid(linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("task_1_3_offer_profiles.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 7.  Summary statistics
# ─────────────────────────────────────────────
print("\n=== Summary ===")
for label, res in [("One-price", results_1p), ("Two-price", results_2p)]:
    gap = (res[:, 0].mean() - res[:, 1].mean()) / res[:, 1].mean() * 100
    print(f"{label}:")
    print(f"  Avg IS  expected profit : {res[:,0].mean():,.0f} €")
    print(f"  Avg OOS expected profit : {res[:,1].mean():,.0f} €")
    print(f"  Optimism gap (IS vs OOS): {gap:+.2f}%")
