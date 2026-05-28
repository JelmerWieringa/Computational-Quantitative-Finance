########################################################################################################################
############################################ COMPUTATIONAL FINANCE - PROJECT ###########################################
###################################### QUESTION 3 - STOCHASTIC STRIKE K=S(T_{1/2}) #####################################
########################################################################################################################

import numpy as np
from scipy.stats import norm
from scipy.stats import iqr
import matplotlib.pyplot as plt


############################################# Model Parameters & Time Grid #############################################
########################################################################################################################

#------------------------------------------------ Set Model Parameters -------------------------------------------------
S_0 = 1  # S(T_0)
# K = S_0  # K = S(t_0) := S_0
r = 0.06  # Equivalent to LSM paper
Sigma = 0.2

N = 50000   # Number of paths
m = 120   # Number of steps = number of time points in grid

T_0 = 0  # Initial time.
T_1 = 3  # Start of exercise period (& end of vesting period, not included in the vesting period)
T_2 = 10  # Maturity/expiration date
# \mathcal{T} = \mathcal{T}_{vesting} \cup \mathcal{T}_{exercise}, where T_1 is the first entry of \mathcal{T}_{exercise}.
Delta_t = (T_2 - T_0) / m  # (Uniform) distance between two grid points


#------------------------------------------------ Construct Time Grid --------------------------------------------------
from CF_Project_2 import ConstructTimeGrid
TimeGrid = ConstructTimeGrid(T_2 = T_2, m = m)


#------------------------------------------------- Obtain Index of T_1 -------------------------------------------------
from CF_Project_2 import IndexExerciseStart
Index_T_1 = IndexExerciseStart(time_grid = TimeGrid, T_1 = T_1)


#---------------------------------- Define Payoff Function of Call Option: H(S(t_i)) -----------------------------------
from CF_Project_2 import H_PayoffFunction



#-------------------------------------------- Define ITM Checking Function ---------------------------------------------
from CF_Project_2 import ITM_Paths_Index


#------------------------------------- Define Functions for Stochastic Strike Price ------------------------------------
def IndexStrikeSetDate(time_grid, T_half):
    r"""
    GOAL: Find the index in time_grid corresponding to T_{1/2} = T_1 / 2 \in [T_0, T_1].
          Raises ValueError if T_{1/2} does not coincide with a grid point.
    :param time_grid:  Equidistant time grid {t_i}_{i=0}^{m}.
    :param T_half:     Strike-setting date T_{1/2} = T_1 / 2.
    """
    Index_StrikeDate = np.argmin(np.abs(time_grid - T_half))  # Closest point in terms of minimised absolute difference

    # Checking whether the points lie within the default tolerance of 1e-05
    if np.isclose(time_grid[Index_StrikeDate], T_half) == False:
        raise ValueError(rf"T_{{1/2}}={T_half:.4f} does not fall on a grid point.",
                         rf"Nearest grid point is {time_grid[Index_StrikeDate]:.4f}. Adjust m.")
    return Index_StrikeDate


def StochasticStrikeVector(S, index_T_half):
    r"""
    GOAL: Compute the path-specific strike vector K^{(j)} = S^{(j)}(T_{1/2}), for j \in {1,...,N}.
    :param S:              Stock price matrix of shape (N, m+1).
    :param index_T_half:   Column index corresponding to T_{1/2} in S.
    :return:               K_vec of shape (N,), where K_vec[n] = S[n, index_T_half].
    """
    K_vector = S[:, index_T_half].copy()   # dim=N✕1 & .copy() avoids modifying S
    return K_vector  # K_vector[j] = S[j, index_T_half]




############################################## Generate Stock Price Paths ##############################################
########################################################################################################################
from CF_Project_2 import SimulateGBM
StockPricePaths = SimulateGBM(S_0 = S_0, drift = r, sigma = Sigma, delta_t = Delta_t, N = N, m = m)




################################################ Define Basis Functions ################################################
########################################################################################################################
from CF_Project_2 import BasisFunction_Monomial

from CF_Project_2 import BasisFunction_LaguerrePolynomials_Unstandardized

from CF_Project_2 import BasisFunction_LaguerrePolynomials_Standardized

from CF_Project_2 import BasisFunction_LegendrePolynomials

from CF_Project_2 import BasisFunction_ChebyshevPolynomials

from CF_Project_2 import BasisFunction_Trigonometric




######################################## Regression at Single Exercise Date t_i ########################################
########################################################################################################################
from CF_Project_2 import RegressionContinuation
# RegressionContinuation(S_i, Y, K, basis_function, n_Terms)




############################################## Backward Induction Routine ##############################################
########################################################################################################################
from CF_Project_2 import BackwardInductionRoutine
# BackwardInductionRoutine(S, K, r, delta_t, index_T_1, basis_function, n_Terms)




############################################## Compute ESO Price at Grant ##############################################
########################################################################################################################
from CF_Project_2 import ESO_PriceAtGrant
# ESO_PriceAtGrant(CashFlowMatrix, r, time_grid)




############################## Estimate V_0 with Stochastic Strike & Different Regressors ##############################
########################################################################################################################
T_Half = T_1 / 2  # Assigning a specific value to T_{1/2}
Index_T_Half   = IndexStrikeSetDate(time_grid = TimeGrid, T_half = T_Half)
print(f"T_{{1/2}} = {T_Half}:  Grid-Index = {Index_T_Half}  &  Grid-Value = {TimeGrid[Index_T_Half]:.4f}")

#---------------------------------------------- Fixed Strike: K = S(T_0) -----------------------------------------------
K_FixedStrike = S_0
CF_FixedStrike = BackwardInductionRoutine(S = StockPricePaths, K = K_FixedStrike, r = r, delta_t = Delta_t,
                                          index_T_1 = Index_T_1, basis_function = BasisFunction_Monomial, n_Terms = 3)
Result_FixedStrike = ESO_PriceAtGrant(CashFlowMatrix = CF_FixedStrike, r = r, time_grid = TimeGrid)


#------------------------------------------- Stochastic Strike: Regress on S -------------------------------------------
K_Vector = StochasticStrikeVector(S = StockPricePaths, index_T_half = Index_T_Half)  # dim=N✕1

CF_StochasticStrike_Regressor_S = BackwardInductionRoutine(S = StockPricePaths, K = K_Vector, r = r, delta_t = Delta_t,
                                                           index_T_1 = Index_T_1,
                                                           basis_function = BasisFunction_Monomial, n_Terms = 3)
Result_StochasticStrike_Regressor_S = ESO_PriceAtGrant(CashFlowMatrix = CF_StochasticStrike_Regressor_S,
                                                       r = r, time_grid = TimeGrid)

print(Result_StochasticStrike_Regressor_S)


#--------------------------------------- Stochastic Strike: Regress on M := S/K ----------------------------------------
CF_StochasticStrike_Regressor_M = BackwardInductionRoutine(S= StockPricePaths / K_Vector.reshape(N, 1),
                                                           K = 1, r = r, delta_t = Delta_t, index_T_1 = Index_T_1,
                                                           basis_function = BasisFunction_Monomial, n_Terms = 3)
Result_StochasticStrike_Regressor_M = ESO_PriceAtGrant(CashFlowMatrix = CF_StochasticStrike_Regressor_M,
                                                       r = r, time_grid = TimeGrid)

print(Result_StochasticStrike_Regressor_M)


#------------------------------------------------ Print Results --------------------------------------------------------
print(f"Fixed Strike K=S(T_0):              "
      f"V_0_Hat = {Result_FixedStrike["Price"]:.4f}  &  "
      f"CI = [{Result_FixedStrike["CI"][0]:.4f}, {Result_FixedStrike["CI"][1]:.4f}]")

print(f"Stochastic Strike [Regress on S]:   "
      f"V_0_Hat = {Result_StochasticStrike_Regressor_S["Price"]:.4f}  &  "
      f"CI = [{Result_StochasticStrike_Regressor_S["CI"][0]:.4f}, {Result_StochasticStrike_Regressor_S["CI"][1]:.4f}]")

print(f"Stochastic Strike [Regress on M]:   "
      f"V_0_Hat = {Result_StochasticStrike_Regressor_M["Price"]:.4f}  &  "
      f"CI = [{Result_StochasticStrike_Regressor_M["CI"][0]:.4f}, {Result_StochasticStrike_Regressor_M["CI"][1]:.4f}]")


print(f"\nK statistics:  Mean={np.mean(K_Vector):.4f},  SD={np.std(K_Vector):.4f},  "
      f"Min={np.min(K_Vector):.4f},  Max={np.max(K_Vector):.4f}")




#################################### ESO Price - T_{1/2} \in [T_0, T_1) Sensitivity ####################################
########################################################################################################################

#------------------------------ Compute ESO Price with Stochastic Strike on T_{1/2} Grid -------------------------------
T_Half_Values = TimeGrid[TimeGrid < T_1]   # All grid points strictly before T_1
V_0_Hat_Regressor_S = []
V_0_Hat_Regressor_M = []

for T_Half_i in T_Half_Values:
    Index_T_Half = IndexStrikeSetDate(time_grid = TimeGrid, T_half = T_Half_i)
    K_Vector_T_Half_i = StochasticStrikeVector(S = StockPricePaths, index_T_half = Index_T_Half)

    CF_Regressor_S = BackwardInductionRoutine(S = StockPricePaths,
                                              K = K_Vector_T_Half_i,  # This changes
                                              r = r, delta_t = Delta_t, index_T_1 = Index_T_1,
                                              basis_function = BasisFunction_Monomial, n_Terms = 3)
    CF_Regressor_M = BackwardInductionRoutine(S = StockPricePaths / K_Vector_T_Half_i.reshape(N, 1),  # This changes
                                              K = 1,
                                              r = r, delta_t = Delta_t, index_T_1 = Index_T_1,
                                              basis_function = BasisFunction_Monomial, n_Terms = 3)

    V_0_Hat_Regressor_S.append(ESO_PriceAtGrant(CF_Regressor_S, r = r, time_grid = TimeGrid)["Price"])
    V_0_Hat_Regressor_M.append(ESO_PriceAtGrant(CF_Regressor_M, r = r, time_grid = TimeGrid)["Price"])

V_0_Hat_Regressor_S_Array = np.array(V_0_Hat_Regressor_S)
V_0_Hat_Regressor_M_Array = np.array(V_0_Hat_Regressor_M)


#------------------------------------------------ Sensitivity Plot ----------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))  # Necessary because we want to plot multiple lines on a single figure

ax.plot(T_Half_Values, V_0_Hat_Regressor_S_Array, color = 'steelblue',  linewidth = 2,
        label = r"Stochastic Strike [Regress on $S$]")
ax.plot(T_Half_Values, V_0_Hat_Regressor_M_Array, color = 'darkorange', linewidth = 2,
        label = r"Stochastic Strike [Regress on $M=S/K$]")

# Add the K=S(T_0) case for comparison
ax.axhline(Result_FixedStrike["Price"], color = 'black', linestyle = '--', linewidth = 1.5,
           label = rf"Fixed Strike K=S(T_0):  $\hat{{V}}_0 \approx${Result_FixedStrike["Price"]:.4f}")
# Add a vertical line to indicate the T_Half used above
ax.axvline(T_Half, color = 'greenyellow', linestyle = ':', linewidth = 1.2,
           label = rf"Default $T_{{1/2}}=T_1/2={T_Half}$")

ax.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax.set_title(rf"ESO Price $-$ Sensitivity to $T_{{1/2}}$"
             "\n"
             rf"($T_0$={T_0}, $T_1={T_1}$, $T_2$={T_2}, $\sigma$={Sigma}, $N$={N}, $m$={m})")
ax.set_xlabel(r"Strike Determination Date $T_{1/2}$")
ax.set_ylabel(r"$\hat{V}_0$")
ax.set_xlim([T_Half_Values[0], T_Half_Values[-1]])
ax.legend()
plt.tight_layout()
plt.savefig(f"CF_Project_3_SensitivityT_Half.png", dpi = 300, bbox_inches = 'tight')
plt.show()




########################################### Distribution of K^{(j)} ####################################################
########################################################################################################################
# log(K) ~ N(\mu_K, \sigma_K^2), with:
#       - \mu_K = log(S_0) + (r - 0.5*\sigma^2) * T_{1/2}
#       - \sigma_K^2 = \sigma^K * T_{1/2}
Mu_K = np.log(S_0) + (r - 0.5*Sigma**2) * T_Half
Sigma_K = Sigma * np.sqrt(T_Half)

fig, ax = plt.subplots(figsize=(10, 6))  # Necessary because we want to plot multiple lines on a single figure

# Plot the empirical distribution of K^{(j)} by means of a histogram
OptimalBinWidth = 2 * iqr(K_Vector) * N**(-1/3)  # Use the Freedman-Diaconis rule for optimal bin width
NoOfBins = int(np.ceil((K_Vector.max() - K_Vector.min()) / OptimalBinWidth))
ax.hist(K_Vector, bins = NoOfBins, edgecolor = 'grey', density = True, color = 'steelblue', alpha = 0.6,
        label = r"Empirical Distribution of $K^{(j)}$")


# Overlay the histogram by the theoretical log-normal PDF
x_Range = np.linspace(K_Vector.min(), K_Vector.max(), 400)
PDF_LogNormal = ((1 / (x_Range * Sigma_K * np.sqrt(2 * np.pi))) *  # Log-Normal PDF
                 np.exp(-0.5 * ((np.log(x_Range / S_0) - Mu_K) / Sigma_K)**2))
ax.plot(x_Range, PDF_LogNormal, color = 'black', linewidth = 2,  # Plot Log-Normal PDF
        label = rf"Log-Normal PDF $[\mu={Mu_K:.2f}, \ \sigma={Sigma_K:.2f}]$")
ax.axvline(np.mean(K_Vector), color = 'darkorange', linestyle="--", linewidth=1.5,  # Plot sample mean of K
           label = rf"$\mathbb{{E}}[K] \approx${np.mean(K_Vector):.3f}")

ax.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax.set_title(rf"Distribution of Stochastic Strike $K^{{(j)}} = S^{{(j)}}(T_{{1/2}})$, with $T_{{1/2}}={T_Half}$"
              "\n"
              rf"($T_0$={T_0}, $T_1$={T_1}, $T_2$={T_2}, $\sigma$={Sigma}, $N$={N:,}, $m$={m})")
ax.set_xlabel(r"$K^{(j)} = S^{(j)}(T_{1/2})$")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig("CF_Project_3_StrikeDistribution.png", dpi = 300, bbox_inches = 'tight')
plt.show()