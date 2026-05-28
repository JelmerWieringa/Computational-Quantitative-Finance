########################################################################################################################
######################################## COMPUTATIONAL FINANCE - EXERCISE SET 1 ########################################
###################################################### QUESTION 3 ######################################################
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


################################## Black-Scholes Closed-Form Solution --> Exact Price ##################################
def BS_CallPrice(S_0, K, r, sigma, T_Maturity):
    """
    GOAL = To obtain the exact price of a European call option by the Black-Scholes formula.
    """
    d_1 = (np.log(S_0 / K) + (r + 0.5 * sigma**2) * T_Maturity) / (sigma * np.sqrt(T_Maturity))
    d_2 = d_1 - sigma * np.sqrt(T_Maturity)

    V_0 = S_0 * norm.cdf(d_1) - np.exp(-r * T_Maturity) * K * norm.cdf(d_2)  # norm.cdf() gives the normal CDF
    return V_0



################################## MC Call Price: With & Without Standardization of Z ##################################
def MC_CallPrice(S_0, K, r, sigma, T_Maturity, NoOfPaths, Standardize):
    """
    GOAL = To obtain an estimate of the call option price by Monte Carlo simulation, using the Feynman-Kac Theorem.
    """
    np.random.seed(21)  # Fix the randomness
    Z = np.random.normal(0, 1, NoOfPaths)  # Z_i \sim N(0,1)

    if Standardize == True and NoOfPaths > 1:
        Z = (Z - np.mean(Z)) / np.std(Z)  # Making sure that samples from normal have mean 0 and variance 1

    S_T = S_0 * np.exp((r - 0.5 * sigma**2) * T_Maturity + sigma * np.sqrt(T_Maturity) * Z)
    Payoff = np.maximum(S_T - K, 0)  # E^Q[ max{S(T_Maturity) − K,0} ]
    Price = np.exp(-r * T_Maturity) * np.mean(Payoff)  # Discounted payoff
    return Price



############################################## Choose the Model Parameters #############################################
S_0 = 100  # Initial value of stock
K = 100  # Strike price
r = 0.03  # Risk-free interest rate
sigma = 0.21  # Volatility
T_Maturity = 1  # Maturity Date

ExactPrice = BS_CallPrice(S_0, K, r, sigma, T_Maturity)

NoOfPaths = 50000



############################# Obtaining the Monte Carlo Estimates over a Range of NoOfPaths ############################
# Create an array with all the number of paths/trajectories considered in the MC estimates
PathRange = np.arange(100, NoOfPaths + 1, 100)  # Array [100, 200, ..., NoOfPaths]

Prices_Standardized = np.zeros(len(PathRange))
Prices_NOT_Standardized = np.zeros(len(PathRange))
AbsoluteErrors_Standardized = np.zeros(len(PathRange))
AbsoluteErrors_NOT_Standardized = np.zeros(len(PathRange))

for index, n_Paths in enumerate(PathRange):  # enumerate() provides both the index and the element simultaneously.
    Prices_Standardized[index] = MC_CallPrice(S_0, K, r, sigma, T_Maturity, n_Paths, Standardize = True)
    Prices_NOT_Standardized[index] = MC_CallPrice(S_0, K, r, sigma, T_Maturity, n_Paths, Standardize = False)
    AbsoluteErrors_Standardized[index] = np.abs(Prices_Standardized[index] - ExactPrice)
    AbsoluteErrors_NOT_Standardized[index] = np.abs(Prices_NOT_Standardized[index] - ExactPrice)



########################## Plotting the Monte Carlo Estimates over Different NoOfPaths Values ##########################
plt.figure(figsize=(10, 6))

plt.plot(PathRange, Prices_NOT_Standardized, label = "Without Standardization")
plt.plot(PathRange, Prices_Standardized, label = "With Standardization")
plt.axhline(y = ExactPrice, color = 'black', linestyle = '--', label = f"Exact BS price = {ExactPrice:.3f}")

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.title(r"Convergence of Monte Carlo Call Price Estimates")
plt.xlabel(r"Number of Paths")
plt.ylabel(r"Estimated Call Option Price  $\hat{V}(S_0, t_0)$")
plt.xlim([0, NoOfPaths])
plt.legend()
plt.tight_layout()
plt.savefig("CF_HW1_Q3_ConvergencePlot.png", dpi = 300, bbox_inches = 'tight')
plt.show()



############################### Plotting the Absolute Error of the Monte Carlo Estimates ###############################
plt.figure(figsize=(10, 6))

plt.plot(PathRange, AbsoluteErrors_NOT_Standardized, label = "Without Standardization")
plt.plot(PathRange, AbsoluteErrors_Standardized, label = "With Standardization")

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.title("Absolute Error of Monte Carlo Estimate")
plt.xlabel("Number of Paths")
plt.ylabel(r"Absolute Error  $|V(S_0, t_0) - \hat{V}(S_0, t_0)|$")
plt.xlim([0, NoOfPaths])
plt.legend()
plt.tight_layout()
plt.savefig("CF_HW1_Q3_AbsoluteError.png", dpi = 300, bbox_inches = 'tight')
plt.show()



############################ Comparing the True Price to the (NOT) Standardized MC Estimates ###########################
print(f"Exact BS Price of Call Option:                   {ExactPrice:.4f}")
print(f"MC Price, Standardized, NoOfPaths = {NoOfPaths}:       {Prices_Standardized[-1]:.4f}")
print(f"MC Price, Not Standardized, NoOfPaths = {NoOfPaths}:   {Prices_NOT_Standardized[-1]:.4f}")