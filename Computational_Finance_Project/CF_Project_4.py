########################################################################################################################
############################################ COMPUTATIONAL FINANCE - PROJECT ###########################################
################################################ QUESTION 4 - DIVIDENDS ################################################
########################################################################################################################

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


############################################# Model Parameters & Time Grid #############################################
########################################################################################################################

#------------------------------------------------ Set Model Parameters -------------------------------------------------
S_0 = 1  # S(T_0)
K = S_0  # K = S(t_0) := S_0
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




############################################## Generate Stock Price Paths ##############################################
########################################################################################################################
from CF_Project_2 import SimulateGBM




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




###################################################### Simulation ######################################################
########################################################################################################################

#-------------------------------------------- ESO Price with Fixed Dividend --------------------------------------------
d = 0.03  # Dividend
StockPricePaths_Dividend = SimulateGBM(S_0 = S_0, drift = r - d, sigma = Sigma, delta_t = Delta_t, N = N, m = m)
# Note: The drift is r-d.

CF_Dividend = BackwardInductionRoutine(S = StockPricePaths_Dividend, K = K, r = r, delta_t = Delta_t,
                                       index_T_1 = Index_T_1, basis_function = BasisFunction_Monomial, n_Terms = 3)

Result_Dividend = ESO_PriceAtGrant(CashFlowMatrix = CF_Dividend, r = r, time_grid = TimeGrid)

print(f"Dividend d={d}:  V_0_Hat_Dividend: {Result_Dividend["Price"]:.4f}  &  "
      f"CI = [{Result_Dividend["CI"][0]:.4f}, {Result_Dividend["CI"][1]:.4f}]")


#------------------------------------------ Plot ESO Price - d Sensitivity ---------------------------------------------
DividendValues = np.linspace(0, r, 30)
V_0_Hat_Dividend = np.zeros(len(DividendValues))

for i in range(len(DividendValues)):
    # Sampling GBM paths is the only component that depends on dividend d through the drift.
    StockPricePaths_Dividend = SimulateGBM(S_0 = S_0, drift = r - DividendValues[i], sigma = Sigma, delta_t = Delta_t,
                                        N = N, m = m)

    CashFlowMatrix_Dividend = BackwardInductionRoutine(S = StockPricePaths_Dividend, K = K, r = r, delta_t = Delta_t,
                                                    index_T_1 = Index_T_1,
                                                    basis_function = BasisFunction_Monomial, n_Terms = 3)
    V_0_Hat_Dividend[i] = ESO_PriceAtGrant(CashFlowMatrix = CashFlowMatrix_Dividend,
                                           r = r, time_grid = TimeGrid)["Price"]

# Plot \hat{V}_0 over dividend grid
plt.figure(figsize = (10, 6))

plt.plot(DividendValues, V_0_Hat_Dividend)
plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.title(rf"Estimated ESO Price $-$ Sensitivity to Dividend $d$"
          "\n"
          rf"($T_0$={T_0}, $T_1$={T_1}, $T_2$={T_2}, $\sigma$={Sigma}, $N$={N:,}, $m$={m})")
plt.xlabel(r"Dividend $d$")
plt.ylabel(r"$\hat{V}_0$")
plt.xlim([DividendValues[0], DividendValues[-1]])
plt.tight_layout()
plt.savefig(f"CF_Project_4_SensitivityDividend.png", dpi = 300, bbox_inches = 'tight')
plt.show()


#--------------------------------------- Plot ESO Price - (d, r) Sensitivity -------------------------------------------
rValues = np.linspace(0.01, 0.1, 15)
dValues = np.linspace(0., 0.1, 15)

# Compute ESO price over the grid
rGrid, dGrid = np.meshgrid(rValues, dValues)  # Create 2D grids, with rGrid[i,j]=rValues[j], dGrid[i,j]=dValues[i]
V_0_Hat_Grid = np.zeros_like(rGrid)  # To store ESO prices

for i, d_value in enumerate(dValues):  # Loop over rows, i.e., d-value
    for j, r_value in enumerate(rValues):  # Loop over columns, i.e., r-value
        # Note: enumerate() returns both the index and value of each element.

        S = SimulateGBM(S_0 = S_0, drift = r_value - d_value, sigma = Sigma, delta_t = Delta_t, N = N, m = m)

        CF = BackwardInductionRoutine(S = S, K = K, r = r_value, delta_t = Delta_t, index_T_1 = Index_T_1,
                                      basis_function = BasisFunction_Monomial, n_Terms = 3)

        # Store ESO price at point (i,j)
        V_0_Hat_Grid[i, j] = ESO_PriceAtGrant(CF, r = r_value, time_grid = TimeGrid)["Price"]


# Create 3D plot of (x,y,z) = (d, r, ESO price)
fig = plt.figure(figsize=(6, 6))              # Create figure of size 10x7 inches
ax  = fig.add_subplot(111, projection = '3d')    # Add a single 3D subplot

surface = ax.plot_surface(dGrid, rGrid, V_0_Hat_Grid, cmap = 'cool')   # Plot surface: x=d, y=r, z=ESO price
fig.colorbar(surface, shrink = 0.5, pad = 0.1)  # Adds colourbar to the right

ax.set_xlabel(r"Dividend $d$")
ax.set_ylabel(r"Risk-free rate $r$")
ax.set_zlabel(r"$\hat{V}_0$")
ax.set_title(r"ESO Price over $(r, d)$-Grid")

# Make axes planes transparant
ax.xaxis.pane.set_facecolor((1,1,1,0))
ax.yaxis.pane.set_facecolor((1,1,1,0))
ax.zaxis.pane.set_facecolor((1,1,1,0))
# Make lines on axes planes dashed & lightgrey
ax.xaxis._axinfo['grid'].update({'linestyle': '--', 'linewidth': 0.5, 'color': 'lightgrey'})
ax.yaxis._axinfo['grid'].update({'linestyle': '--', 'linewidth': 0.5, 'color': 'lightgrey'})
ax.zaxis._axinfo['grid'].update({'linestyle': '--', 'linewidth': 0.5, 'color': 'lightgrey'})

plt.tight_layout()
plt.savefig("CF_Project_4_Surface_r_d.png", dpi=300, bbox_inches='tight')
plt.show()