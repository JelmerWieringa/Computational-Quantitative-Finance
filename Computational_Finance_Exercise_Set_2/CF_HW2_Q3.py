########################################################################################################################
##################################### COMPUTATIONAL FINANCE - EXERCISE SET 2 ###########################################
################################################## EXERCISE 3 #########################################################
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt


############################################# Setting Model Parameters #################################################
sigma_1 = 0.4  # Volatility of asset 1
sigma_2 = 0.15  # Volatility of asset 2
r       = 0.01  # Risk-free rate
S_1_0   = 1  # Initial price of asset 1
S_2_0   = 1  # Initial price of asset 2
T       = 1  # Maturity
t_0     = 0  # Initial time

Rho_Values = [-0.9, 0.9]  # Correlation values

NoOfPaths = 100000  # Number of Monte Carlo paths
NoOfSteps = 1  # Only one step suffices, since GBM has an exact solution



###################### Simulate Two Correlated GBM Assets via Cholesky Decomposition ###################################
def SimulateTwoAssetGBM(S_1_0, S_2_0, r, sigma_1, sigma_2, rho, T, NoOfPaths):
    """
    GOAL = Simulate S_1(T) and S_2(T) under correlated GBM using the exact solution.
    """
    np.random.seed(23)

    # Generate two independent standard normal vectors
    Z_1 = np.random.normal(0, 1, [NoOfPaths])
    Z_2 = np.random.normal(0, 1, [NoOfPaths])

    # Cholesky decomposition to create correlated Brownian increments
    W_1_T = np.sqrt(T) * Z_1
    W_2_T = np.sqrt(T) * (rho * Z_1 + np.sqrt(1 - rho**2) * Z_2)

    # Exact GBM solution for S_1(T) and S_2(T)
    S_1_T = S_1_0 * np.exp((r - 0.5 * sigma_1**2) * T + sigma_1 * W_1_T)
    S_2_T = S_2_0 * np.exp((r - 0.5 * sigma_2**2) * T + sigma_2 * W_2_T)

    return S_1_T, S_2_T



################################## Compute Option Price and Confidence Interval ########################################
def Payoff_CI(S_1_T, S_2_T, r, T):
    r"""
    GOAL = Compute the discounted price of the two-asset option and the corresponding 95% confidence interval (CI).
    """
    Payoff = S_1_T * np.maximum(S_1_T, S_2_T)  # Payoff: S_1(T) * max(S_1(T), S_2(T))
    DiscountedPayoff = np.exp(-r * T) * Payoff

    V_hat = np.mean(DiscountedPayoff)  # Monte Carlo estimator

    StdError = np.std(DiscountedPayoff) / np.sqrt(len(DiscountedPayoff))  # Standard error

    # 95% confidence interval
    CI_lower = V_hat - 1.96 * StdError
    CI_upper = V_hat + 1.96 * StdError

    return V_hat, StdError, CI_lower, CI_upper



######################################## Printing Important Values #####################################################
print(f"\nParameters: sigma_1 = {sigma_1}, sigma_2 = {sigma_2}, r = {r}, S_1(0) = {S_1_0}, S_2(0) = {S_2_0}, T = {T}, NoOfPaths = {NoOfPaths}\n")

Results = {}

for rho in Rho_Values:
    # Simulate the two correlated assets
    S_1_T, S_2_T = SimulateTwoAssetGBM(S_1_0, S_2_0, r, sigma_1, sigma_2, rho, T, NoOfPaths)

    # Compute the option price and confidence interval
    V_hat, StdError, CI_lower, CI_upper = Payoff_CI(S_1_T, S_2_T, r, T)

    Results[rho] = {"V_hat": V_hat, "StdError": StdError, "CI_lower": CI_lower, "CI_upper": CI_upper}

    print(f"rho = {rho:+.1f}:")
    print(f"  Option Price     = {V_hat:.6f}")
    print(f"  Standard Error   = {StdError:.6f}")
    print(f"  95% CI           = [{CI_lower:.6f}, {CI_upper:.6f}]")
    print()



######################################### Analysis of the Payoff #######################################################
print("-" * 80)
print("Analysis:")
print("-" * 80)

V_neg = Results[-0.9]["V_hat"]
V_pos = Results[0.9]["V_hat"]

print(f"\n  V(rho = -0.9) = {V_neg:.6f}")
print(f"  V(rho = +0.9) = {V_pos:.6f}")
print(f"  Difference    = {V_neg - V_pos:.6f}")
print(f"\n  The option is MORE expensive for rho = -0.9 than for rho = +0.9.")



###################################### Plotting Histograms of the Payoff ###############################################
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, rho in enumerate(Rho_Values):
    S_1_T, S_2_T = SimulateTwoAssetGBM(S_1_0, S_2_0, r, sigma_1, sigma_2, rho, T, NoOfPaths)
    Payoff = S_1_T * np.maximum(S_1_T, S_2_T)

    axes[idx].hist(Payoff, bins='fd', edgecolor='grey', density=True, alpha=0.6, color='steelblue')
    axes[idx].axvline(x=np.mean(Payoff), color='red', linewidth=2, linestyle='--',
                      label=rf'Expected Payoff = ${np.mean(Payoff):.4f}$')
    axes[idx].set_title(rf'Payoff distribution, $\rho = {rho}$')
    axes[idx].set_xlabel(r'$S_1(T) \cdot \max(S_1(T), S_2(T))$')
    axes[idx].set_ylabel('Density')
    axes[idx].legend()
    axes[idx].grid(True, which='both', linestyle='--', alpha=0.5)
    axes[idx].set_xlim([0, 8])

plt.tight_layout()
plt.savefig("CF_HW2_Q4_PayoffHistograms.png", dpi=300, bbox_inches='tight')
plt.show()



################################# Plotting Scatter Plot of (S_1(T), S_2(T)) ############################################
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, rho in enumerate(Rho_Values):
    S_1_T, S_2_T = SimulateTwoAssetGBM(S_1_0, S_2_0, r, sigma_1, sigma_2, rho, T, NoOfPaths)

    N_plot = 2000
    axes[idx].scatter(S_1_T[:N_plot], S_2_T[:N_plot], alpha=0.3, s=3, color='steelblue')
    axes[idx].set_title(rf'Scatter plot of $(S_1(T), S_2(T))$, $\rho = {rho}$')
    axes[idx].set_xlabel(r'$S_1(T)$')
    axes[idx].set_ylabel(r'$S_2(T)$')
    axes[idx].grid(True, which='both', linestyle='--', alpha=0.5)
    axes[idx].set_xlim([0, 4])
    axes[idx].set_ylim([0, 2])
    axes[idx].set_aspect('equal')

plt.tight_layout()
plt.savefig("CF_HW2_Q4_ScatterPlots.png", dpi=300, bbox_inches='tight')
plt.show()