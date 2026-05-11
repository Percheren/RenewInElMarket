"""
Task 1.4 - Risk-Averse Stochastic Offering via Mean-CVaR (alpha = 0.90)
========================================================================
Following Lecture 9 (Kazempour, DTU 46755), the risk-neutral models of
Tasks 1.1 and 1.2 are extended with a mean-CVaR objective.
"""
from pathlib import Path
from scipy.stats import bernoulli
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 0.  Parameters
# =============================================================================
T        = 24
capacity = 500          # MW  (= P_nom)
N_wind   = 20
N_price  = 20
N_si     = 4
W        = N_wind * N_price * N_si   # 1 600 equiprobable scenarios
ALPHA    = 0.90


# =============================================================================
# 1.  Load and construct scenarios
# =============================================================================
base_path = Path(__file__).resolve().parent.parent / "data"
DA_price = pd.read_csv(base_path / "Day-Ahead-Price-01_03_26-20_30_26.csv", sep=";")
#DA_price = pd.read_csv(
#    r"C:\Users\GioBr\Desktop\DTU\RIEM - Renewables in Electricity Market\Assignment 2\Day-Ahead-Price-01_03_26-20_30_26.csv",
#    sep=";"
#)
DA_price["Eur/MWh"] = DA_price["Eur/MWh"].str.replace(',', '.').astype(float)
prices          = DA_price["Eur/MWh"].values[:24 * N_price]
price_scenarios = prices.reshape(N_price, T)

wind_prod = pd.read_excel(base_path / "ninja-wind-country-DK-current_onshore-merra2.xlsx")
#wind_prod      = pd.read_excel(
#    r"C:\Users\GioBr\Desktop\DTU\RIEM - Renewables in Electricity Market\Assignment 2\ninja-wind-country-DK-current_onshore-merra2.xlsx",
#    header=3
#)
wind_scenarios = wind_prod["DK02_factor [1]"].values[:24 * N_wind] * capacity
wind_scenarios = wind_scenarios.reshape(N_wind, T)

np.random.seed(42)
si_scenarios = bernoulli.rvs(0.5, size=(N_si, T))

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

# One-price balancing: lBP = 1.25*lDA (SI=1, deficit) or 0.85*lDA (SI=0, surplus)
lambda_bal = np.where(SI == 1, 1.25 * DA_lambda, 0.85 * DA_lambda)

# Two-price coefficients
# SI=1 (deficit):  over-gen rewarded at lDA, under-gen penalised at lBP
# SI=0 (surplus):  over-gen penalised at lBP, under-gen rewarded at lDA
c_up = SI * DA_lambda  + (1 - SI) * lambda_bal   # revenue coeff for Delta_up
c_dn = SI * lambda_bal + (1 - SI) * DA_lambda    # cost   coeff for Delta_down

pi_w = 1.0 / W

print(f"Scenarios: {W}  |  Hours: {T}  |  alpha = {ALPHA}")
print(f"Mean DA price : {DA_lambda.mean():.2f} EUR/MWh")
print(f"Mean wind cap.: {real_prod.mean():.1f} MW")

# Pre-solve analytics: one-price marginal revenue E[lDA - lBP] per hour
# dE[Profit_w]/dp_DA_t = E[lDA_t - lBP_t]
# > 0 → bid P_nom at hour t;  < 0 → bid 0 at hour t
MR_1p = np.array([(DA_lambda[:, t] - lambda_bal[:, t]).mean() for t in range(T)])
print(f"\nOne-price E[lDA-lBP]: mean={MR_1p.mean():.4f} | "
      f"hours MR>0: {(MR_1p>0).sum()}/24 | hours MR<0: {(MR_1p<0).sum()}/24")
print(f"Expected corner solution: p_DA=P_nom at {(MR_1p>0).sum()} hours, "
      f"p_DA=0 at {(MR_1p<0).sum()} hours")


# =============================================================================
# 2.  Numpy helpers for ex-post evaluation
# =============================================================================

def scenario_profits_1p(p_DA_sol, idx):
    """One-price profits, imbalances from closed-form (no simultaneous +/+)."""
    du = np.maximum(real_prod[idx] - p_DA_sol, 0)
    dd = np.maximum(p_DA_sol - real_prod[idx], 0)
    return np.sum(DA_lambda[idx] * p_DA_sol
                  + lambda_bal[idx] * (du - dd), axis=1)


def scenario_profits_2p(p_DA_sol, idx):
    """Two-price profits, imbalances from closed-form."""
    du = np.maximum(real_prod[idx] - p_DA_sol, 0)
    dd = np.maximum(p_DA_sol - real_prod[idx], 0)
    return np.sum(DA_lambda[idx] * p_DA_sol
                  + c_up[idx] * du - c_dn[idx] * dd, axis=1)


def cvar_from_profits(profits, alpha=ALPHA):
    """CVaR_alpha: expected profit in the worst (1-alpha) fraction."""
    var  = np.quantile(profits, 1 - alpha)
    tail = profits[profits <= var]
    return float(tail.mean()) if len(tail) > 0 else float(var)


# =============================================================================
# 3.  One-price mean-CVaR LP  (Lecture 9 formulation)
# =============================================================================

def solve_1price_cvar(idx, beta, alpha=ALPHA):
    """
    Risk-averse one-price LP following Lecture 9, slide 65.

    Delta_up and Delta_down have NO upper bound — only lb=0.
    The economics (lBP != lDA) prevent simultaneous positivity at optimality.

    Parameters
    ----------
    idx  : scenario indices (in-sample set)
    beta : risk-aversion weight (>= 0);  beta=0 recovers Task 1.1
    """
    n  = len(idx)
    pi = 1.0 / n
    # Finite bounds for zeta prevent GRB.INF_OR_UNBD at beta=0
    zeta_lb = float(np.minimum(DA_lambda[idx], lambda_bal[idx]).min()) * (-capacity) * T
    zeta_ub = float(np.maximum(DA_lambda[idx], lambda_bal[idx]).max()) *  capacity  * T

    m = gp.Model()
    m.setParam("OutputFlag", 0)
    m.setParam("DualReductions", 0)

    # First-stage: DA offer in [0, P_nom]
    p_DA = m.addVars(T, lb=0, ub=capacity, name="pDA")

    # Second-stage: imbalance split variables, bounded by P_nom
    # Physical upper bound: cannot over/under-produce more than installed capacity
    Delta_up   = m.addVars(n, T, lb=0, ub=capacity, name="Dup")
    Delta_down = m.addVars(n, T, lb=0, ub=capacity, name="Ddn")

    # CVaR auxiliary: zeta (free, endogenous VaR), eta[w] >= 0
    zeta = m.addVar(lb=zeta_lb, ub=zeta_ub, name="zeta")
    eta  = m.addVars(n, lb=0, name="eta")

    # Imbalance definition: Delta_up - Delta_down = p_real - p_DA
    for ii in range(n):
        for t in range(T):
            m.addConstr(
                Delta_up[ii, t] - Delta_down[ii, t]
                == real_prod[idx[ii], t] - p_DA[t]
            )

    # Scenario profit: lDA*p_DA + lBP*(Delta_up - Delta_down)
    Pi = {
        ii: gp.quicksum(
            DA_lambda[idx[ii], t] * p_DA[t]
            + lambda_bal[idx[ii], t] * (Delta_up[ii, t] - Delta_down[ii, t])
            for t in range(T)
        )
        for ii in range(n)
    }

    # CVaR shortfall: eta[w] >= zeta - Profit_w  (lecture slide 65)
    for ii in range(n):
        m.addConstr(eta[ii] >= zeta - Pi[ii])

    # Objective: E[Pi] + beta * CVaR  (lecture slide 57, approach 1)
    E_Pi     = gp.quicksum(pi * Pi[ii] for ii in range(n))
    CVaR_lin = zeta - (1.0 / (1 - alpha)) * gp.quicksum(pi * eta[ii] for ii in range(n))
    m.setObjective(E_Pi + beta * CVaR_lin, GRB.MAXIMIZE)
    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError(f"1p solver: status {m.Status} for beta={beta}")

    return np.array([p_DA[t].X for t in range(T)])


# =============================================================================
# 4.  Two-price mean-CVaR LP  (Lecture 9 formulation, two-price profit)
# =============================================================================

def solve_2price_cvar(idx, beta, alpha=ALPHA):
    """
    Risk-averse two-price LP following Lecture 9 structure.

    Same as one-price solver but uses c_up / c_dn coefficients for the
    asymmetric two-price balancing settlement (Task 1.2).
    Delta_up and Delta_down have NO upper bound — only lb=0.
    """
    n  = len(idx)
    pi = 1.0 / n
    zeta_lb = float(np.minimum(DA_lambda[idx], lambda_bal[idx]).min()) * (-capacity) * T
    zeta_ub = float(np.maximum(DA_lambda[idx], lambda_bal[idx]).max()) *  capacity  * T

    m = gp.Model()
    m.setParam("OutputFlag", 0)
    m.setParam("DualReductions", 0)

    p_DA       = m.addVars(T, lb=0, ub=capacity, name="pDA")
    Delta_up   = m.addVars(n, T, lb=0, ub=capacity, name="Dup")
    Delta_down = m.addVars(n, T, lb=0, ub=capacity, name="Ddn")
    zeta       = m.addVar(lb=zeta_lb, ub=zeta_ub, name="zeta")
    eta        = m.addVars(n, lb=0, name="eta")

    for ii in range(n):
        for t in range(T):
            m.addConstr(
                Delta_up[ii, t] - Delta_down[ii, t]
                == real_prod[idx[ii], t] - p_DA[t]
            )

    # Two-price profit: lDA*p_DA + c_up*Delta_up - c_dn*Delta_down
    Pi = {
        ii: gp.quicksum(
            DA_lambda[idx[ii], t] * p_DA[t]
            + c_up[idx[ii], t] * Delta_up[ii, t]
            - c_dn[idx[ii], t] * Delta_down[ii, t]
            for t in range(T)
        )
        for ii in range(n)
    }

    for ii in range(n):
        m.addConstr(eta[ii] >= zeta - Pi[ii])

    E_Pi     = gp.quicksum(pi * Pi[ii] for ii in range(n))
    CVaR_lin = zeta - (1.0 / (1 - alpha)) * gp.quicksum(pi * eta[ii] for ii in range(n))
    m.setObjective(E_Pi + beta * CVaR_lin, GRB.MAXIMIZE)
    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError(f"2p solver: status {m.Status} for beta={beta}")

    return np.array([p_DA[t].X for t in range(T)])


# =============================================================================
# 5.  Beta sweep over all 1600 scenarios
# =============================================================================
BETAS   = [0, 0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
all_idx = np.arange(W)
results = {"1p": [], "2p": []}

print(f"\n{'beta':>5}  {'E[Pi] 1P':>12}  {'CVaR 1P':>12}  "
      f"{'E[Pi] 2P':>12}  {'CVaR 2P':>12}")
print("-" * 62)

for beta in BETAS:
    pDA_1p = solve_1price_cvar(all_idx, beta)
    Pi_1p  = scenario_profits_1p(pDA_1p, all_idx)
    ep_1p  = float(Pi_1p.mean())
    cv_1p  = cvar_from_profits(Pi_1p)

    pDA_2p = solve_2price_cvar(all_idx, beta)
    Pi_2p  = scenario_profits_2p(pDA_2p, all_idx)
    ep_2p  = float(Pi_2p.mean())
    cv_2p  = cvar_from_profits(Pi_2p)

    results["1p"].append(dict(beta=beta, pDA=pDA_1p, Pi=Pi_1p, E=ep_1p, CVaR=cv_1p))
    results["2p"].append(dict(beta=beta, pDA=pDA_2p, Pi=Pi_2p, E=ep_2p, CVaR=cv_2p))

    print(f"{beta:>5.2f}  {ep_1p:>12,.0f}  {cv_1p:>12,.0f}  "
          f"{ep_2p:>12,.0f}  {cv_2p:>12,.0f}")


# =============================================================================
# 6.  Plots
# =============================================================================
C     = {"1p": "steelblue", "2p": "coral"}
L     = {"1p": "One-price", "2p": "Two-price"}
SEL   = [0, 1.0, 5.0]
hours = np.arange(1, T + 1)

# Figure 1: Efficient frontier
fig, ax = plt.subplots(figsize=(7, 5))
for key in ("1p", "2p"):
    ep   = [r["E"]    for r in results[key]]
    cvar = [r["CVaR"] for r in results[key]]
    ax.plot(cvar, ep, "-o", color=C[key], ms=7, lw=1.8, label=L[key])
    for i, b in enumerate(BETAS):
        if b in (0, BETAS[-1]):
            ax.annotate(f"β={b}", (cvar[i], ep[i]),
                        textcoords="offset points", xytext=(6, 3),
                        fontsize=8, color=C[key])
ax.set_xlabel("CVaR$_{0.90}$ (EUR/day)", fontsize=11)
ax.set_ylabel("$\\mathbb{E}[\\Pi]$ (EUR/day)", fontsize=11)
ax.set_title("Task 1.4 — Efficient Frontier  (α=0.90)", fontsize=12)
ax.legend(fontsize=10); ax.grid(linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("task1_4_efficient_frontier.png", dpi=150)
plt.show()

# Figure 2: E[Pi] and CVaR vs beta
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for key in ("1p", "2p"):
    ep   = [r["E"]    for r in results[key]]
    cvar = [r["CVaR"] for r in results[key]]
    axes[0].plot(BETAS, ep,   "-o", color=C[key], ms=6, lw=1.6, label=L[key])
    axes[1].plot(BETAS, cvar, "-s", color=C[key], ms=6, lw=1.6, label=L[key])
for ax, yl, ttl in zip(
    axes,
    ["$\\mathbb{E}[\\Pi]$ (EUR/day)", "CVaR$_{0.90}$ (EUR/day)"],
    ["Expected Profit vs $\\beta$", "CVaR vs $\\beta$"]
):
    ax.set_xlabel("$\\beta$", fontsize=11); ax.set_ylabel(yl, fontsize=11)
    ax.set_title(f"Task 1.4 — {ttl}", fontsize=12)
    ax.legend(fontsize=10); ax.grid(linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("task1_4_beta_tradeoff.png", dpi=150)
plt.show()

# Figure 3: DA offer profiles
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for key, ax in zip(("1p", "2p"), axes):
    for r in results[key]:
        in_sel = r["beta"] in SEL
        ax.plot(hours, r["pDA"],
                lw=(2.2 if in_sel else 0.7),
                alpha=(0.9 if in_sel else 0.3),
                label=(f"β={r['beta']}" if in_sel else None))
    ax.axhline(capacity, color="gray", ls=":", lw=1.2,
               label=f"$P^{{\\rm nom}}$={capacity} MW")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("$p_t^{DA}$ (MW)")
    ax.set_title(f"Task 1.4 — {L[key]}: DA Offers vs $\\beta$", fontsize=12)
    ax.set_xlim(1, T); ax.set_xticks(range(1, T+1, 3))
    ax.legend(fontsize=9); ax.grid(linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("task1_4_da_offers.png", dpi=150)
plt.show()

# Figure 4: Profit distributions for beta in {0, 1, 5}
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for col, beta in enumerate(SEL):
    for row, key in enumerate(("1p", "2p")):
        ax  = axes[row][col]
        r   = results[key][BETAS.index(beta)]
        Pi  = r["Pi"]; ep = r["E"]; cv = r["CVaR"]
        var = float(np.quantile(Pi, 1 - ALPHA))
        ax.hist(Pi, bins=60, color=C[key], alpha=0.75, edgecolor="white", lw=0.3)
        ax.axvline(ep,  color="navy",       lw=2.0, ls="-",
                   label=f"$\\mathbb{{E}}[\\Pi]$={ep:,.0f}")
        ax.axvline(cv,  color="crimson",    lw=2.0, ls="--",
                   label=f"CVaR={cv:,.0f}")
        ax.axvline(var, color="darkorange", lw=1.4, ls=":",
                   label=f"VaR={var:,.0f}")
        ax.set_title(f"{L[key]},  β={beta}", fontsize=10)
        ax.set_xlabel("Daily Profit (EUR)"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=8, framealpha=0.85); ax.grid(linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.suptitle("Task 1.4 — Profit Distributions  (α=0.90)", fontsize=13)
plt.tight_layout()
plt.savefig("task1_4_profit_distributions.png", dpi=150)
plt.show()

# =============================================================================
# 7.  Summary tables
# =============================================================================
print("\n=== Task 1.4 — Full Results ===")
print(f"\n{'beta':>5}  {'E[Pi] 1P':>14}  {'CVaR 1P':>14}  "
      f"{'E[Pi] 2P':>14}  {'CVaR 2P':>14}")
print("-" * 70)
for i, beta in enumerate(BETAS):
    r1 = results["1p"][i]; r2 = results["2p"][i]
    print(f"{beta:>5.2f}  {r1['E']:>14,.0f}  {r1['CVaR']:>14,.0f}  "
          f"{r2['E']:>14,.0f}  {r2['CVaR']:>14,.0f}")

print("\n=== Risk-neutral (β=0) vs Max Risk-Averse (β=5) ===")
for key, lbl in [("1p","One-price"), ("2p","Two-price")]:
    r0 = results[key][0]; r5 = results[key][-1]
    ep_drop   = r0["E"]    - r5["E"]
    cvar_gain = r5["CVaR"] - r0["CVaR"]
    print(f"\n{lbl}:")
    print(f"  E[Pi] sacrifice  : {ep_drop:+,.0f} EUR  ({ep_drop/r0['E']*100:+.2f}%)")
    print(f"  CVaR improvement : {cvar_gain:+,.0f} EUR")
