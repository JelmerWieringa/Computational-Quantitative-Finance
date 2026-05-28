########################################################################################################################
##################################### COMPUTATIONAL FINANCE - EXERCISE SET 2 ###########################################
################################################## EXERCISE 1b #########################################################
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


############################################# Setting Model Parameters #################################################
S_0 = 1   # Initial stock price
r = 0.05  # Risk-free interest rate
t_0 = 0   # Initial time

Maturities = [1, 2, 3, 7, 10]  # Maturity times
Strikes = np.linspace(0.5, 2.5, 50)  # K
Sigma_0 = 0.3  # Initial guess/value for Newton-Raphson algorithm



###################################### Define Time-Dependent Volatility Functions ######################################
def Sigma(t):
    r"""
    GOAL = Compute the time-dependent volatility \sigma(t) = 0.6 - 0.2 * exp(-1.5 * t).
    """
    return 0.6 - 0.2 * np.exp(-1.5 * t)


def Sigma_Star(t_0, T):
    r"""
    GOAL = Compute the "constant" volatility sigma_* at maturity time T.

    sigma_* = sqrt( (1/T) * int_{t_0}^T sigma^2(t) dt ) =: sqrt( (1/T) * \tilde{\sigma}^2 )
    """
    tau = (T - t_0)

    SigmaTilde2 = 0.36 * tau + 0.16 * (np.exp(-1.5 * T) - np.exp(-1.5 * t_0)) - (0.04 / 3) * (np.exp(-3 * T) - np.exp(-3 * t_0))
    return np.sqrt(SigmaTilde2 / tau)



##################################### Black-Scholes Option Pricing Function ############################################
def BS_PutPrice(S_0, K, r, sigma, t_0, T_Maturity):
    r"""
    GOAL = To obtain the price of a European PUT option by the Black-Scholes formula.
    """
    tau = T_Maturity - t_0  # Time to maturity

    d_1 = (np.log(S_0 / float(K)) + (r + 0.5 * np.power(sigma, 2)) * tau) / float(sigma * np.sqrt(tau))
    d_2 = d_1 - sigma * np.sqrt(tau)

    V_0 = K * np.exp(-r * tau) * norm.cdf(-d_2) - S_0 * norm.cdf(-d_1)  # norm.cdf() gives the normal CDF
    return V_0



################################# Compute Vega = dV/dsigma for Newton-Raphson Algorithm ################################
def dV_dsigma(S_0, K, sigma, t_0, T, r):
    r"""
    GOAL = Compute the option vega, i.e., the derivative of the BS option price w.r.t. \sigma.

    vega = g'(sigma) = -K * exp(-r*tau) * f_{N(0,1)}(d2) * sqrt(tau)
    """
    tau = T - t_0
    d2 = (np.log(S_0 / float(K)) + (r - 0.5 * np.power(sigma, 2)) * tau) / float(sigma * np.sqrt(tau))
    return K * np.exp(-r * tau) * norm.pdf(d2) * np.sqrt(tau)



############################## Compute the Implied Volatility by Newton-Raphson Algorithm ##############################
def ImpliedVolatility(S_0, K, sigma, t_0, T, r, V_Market):
    r"""
    GOAL = Compute the Black-Scholes implied volatility using the Newton-Raphson method.

    \sigma_{k+1} = \sigma_k - g(\sigma_k) / g'(\sigma_k), where g(\sigma) := BS(\sigma) - V_market & g'(\sigma) = vega.
    """
    error = 1e10  # initial error

    PutOptionPrice = lambda sigma: BS_PutPrice(S_0 = S_0, K = K, r = r, sigma = sigma, t_0 = t_0, T_Maturity = T)
    vega     = lambda sigma: dV_dsigma(S_0 = S_0, K = K, sigma = sigma, t_0 = t_0, T = T, r = r)

    while error > 1e-10:
        g = PutOptionPrice(sigma) - V_Market
        g_prim = vega(sigma)

        if abs(g_prim) < 1e-12:
            break

        sigma_new = sigma - g / g_prim

        if sigma_new <= 0:
            sigma_new = sigma / 2.0

        error = abs(g)  # error = abs(sigma_new - sigma)
        sigma = sigma_new

    return sigma



####################################### Plotting \sigma(t) Function  over [0,10] #######################################
t_Values = np.linspace(0, 10, 500)

plt.figure(figsize=(10, 6))

plt.plot(t_Values, Sigma(t_Values), 'b-', linewidth=2)
plt.axhline(y=0.4, color='r', linestyle=':', alpha=0.5, label=r"$\sigma(0) = 0.4$")
plt.axhline(y=0.6, color='k', linestyle=':', alpha=0.5, label=r"$\lim_{x \to \infty} \ \sigma(x) = 0.6$")

plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.title(r'Time-Dependent Volatility $\sigma(t) = 0.6 - 0.2 e^{-1.5t}$')
plt.xlabel(r'Time $t$')
plt.ylabel(r'$\sigma(t)$ [%]')
plt.legend()
plt.tight_layout()
plt.savefig("CF_HW2_Q1b_SigmaFunction.png", dpi=300, bbox_inches='tight')
plt.show()



##################### Plotting Implied Volatility versus Strike Price for Different Maturities #########################
plt.figure(figsize=(10, 6))

for T in Maturities:
    IV_list = []

    for K in Strikes:
        # Compute the put option price analytically, by using subquestion 1a
        Price = BS_PutPrice(S_0 = S_0, K = K, r = r, sigma = Sigma_Star(t_0 = t_0, T = T), t_0 = t_0, T_Maturity = T)

        # Inverting using Newton-Raphson algorithm to find the implied volatility
        if Price > 1e-12:
            IV = ImpliedVolatility(S_0 = S_0, K = K, sigma = Sigma_0, t_0 = t_0, T = T, r = r, V_Market = Price)
            IV_list.append(IV)
        else:
            IV_list.append(np.nan)

    plt.plot(Strikes, IV_list, label = rf"$T = {T}$")

plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.title(r'Implied Volatility vs Strike $K$ for Different Maturities $T$')
plt.xlabel(r'Strike $K$')
plt.ylabel('Implied Volatility')
plt.xlim(np.min(Strikes), np.max(Strikes))
plt.legend()
plt.tight_layout()
plt.savefig("CF_HW2_Q1b_IVversusStrike.png", dpi=300, bbox_inches='tight')
plt.show()



############################ Plotting Implied Volatility & \sigma(t) Function over [0,15] #############################
T_range           = np.linspace(0.05, 15, 500)
SigmaStar_values  = np.array([Sigma_Star(t_0 = t_0, T = T) for T in T_range])
Sigma_t_values    = Sigma(T_range)

plt.figure(figsize=(10, 6))

plt.plot(T_range, SigmaStar_values, 'blue', linewidth=2, label=r"Implied Volatility")
plt.plot(T_range, Sigma_t_values, 'orange', linewidth=1.5, label=r"$\sigma(t) = 0.6 - 0.2 e^{-1.5t}$")

plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.title(r"Implied Volatility & Time-Dependent Volatility $\sigma(t)$")
plt.xlabel(r"Maturity $T$")
plt.ylabel("Volatility")
plt.xlim(np.min(T_range), np.max(T_range))
plt.legend()
plt.tight_layout()
plt.savefig("CF_HW2_Q1b_IVversusSigma.png", dpi=300, bbox_inches='tight')
plt.show()