########################################################################################################################
######################################## COMPUTATIONAL FINANCE - ANIMATION (GIF) #######################################
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from scipy.stats import norm


#=======================================================================================================================
#----------------------------------------------- Animation Settings ----------------------------------------------------
#=======================================================================================================================
N_Start        = 25            # Starting number of MC paths
N_End          = 700           # Final number of MC paths
N_Step         = 25            # Increment per frame
N_Grid         = 500           # Time grid resolution for GBM path simulation
Frame_Duration = 420           # Milliseconds per frame in the GIF
Random_Seed    = 21            # Fixed random seed
Gif_Filename   = f'CF_Animation_MC_Convergence_nStart={N_Start}_nEnd={N_End}_nStep={N_Step}.gif'




#=======================================================================================================================
#----------------------------------------------- Model Parameters ------------------------------------------------------
#=======================================================================================================================
S_0        = 100    # Initial stock price
K          = 100    # Strike price
r          = 0.03   # Risk-free interest rate
Sigma      = 0.21   # Volatility
T_Maturity = 1      # Time to maturity




#=======================================================================================================================
#------------------------------------------------- Black-Scholes Price -------------------------------------------------
#=======================================================================================================================
def BS_CallPrice(S_0, K, r, Sigma, T_Maturity):
    r"""
    GOAL: Compute the exact European call option price via the Black-Scholes formula.
    :param S_0:        Initial stock price.
    :param K:          Strike price.
    :param r:          Risk-free interest rate.
    :param Sigma:      Volatility.
    :param T_Maturity: Time to maturity.
    """
    d_1 = (np.log(S_0 / K) + (r + 0.5 * Sigma**2) * T_Maturity) / (Sigma * np.sqrt(T_Maturity))
    d_2 = d_1 - Sigma * np.sqrt(T_Maturity)
    return S_0 * norm.cdf(d_1) - np.exp(-r * T_Maturity) * K * norm.cdf(d_2)


ExactPrice = BS_CallPrice(S_0 = S_0, K = K, r = r, Sigma = Sigma, T_Maturity = T_Maturity)




#=======================================================================================================================
#--------------------------------------------- GBM Path Simulation -----------------------------------------------------
#=======================================================================================================================
def SimulateGBM_Paths(S_0, r, Sigma, T_Maturity, NoOfPaths, NoOfSteps):
    r"""
    GOAL: Simulate NoOfPaths GBM paths S(t) on [0, T_Maturity] using the exact solution.
    :param S_0:        Initial stock price.
    :param r:          Risk-free interest rate (drift under Q-measure).
    :param Sigma:      Volatility.
    :param T_Maturity: Time to maturity.
    :param NoOfPaths:  Number of paths to simulate.
    :param NoOfSteps:  Number of time steps in the discretisation of [0, T_Maturity].
    """
    np.random.seed(Random_Seed)
    Delta_t = T_Maturity / NoOfSteps
    Z       = np.random.normal(0, 1, [NoOfPaths, NoOfSteps])   # Z_i \sim N(0,1)

    S       = np.zeros([NoOfPaths, NoOfSteps + 1])
    S[:, 0] = S_0

    for i in range(NoOfSteps):
        S[:, i + 1] = S[:, i] * np.exp((r - 0.5 * Sigma**2) * Delta_t + Sigma * np.sqrt(Delta_t) * Z[:, i])

    return S   # Shape: (NoOfPaths, NoOfSteps + 1)




#=======================================================================================================================
#----------------------------------------- Pre-Compute Paths & MC Estimates --------------------------------------------
#=======================================================================================================================

#----------------------------------------- Simulate N_End paths once --------------------------------------------------
### Each frame uses the first N_Curr of these paths, so all frames are nested subsets of the same sample
TimeGrid    = np.linspace(0, T_Maturity, N_Grid + 1)
S_AllPaths  = SimulateGBM_Paths(S_0 = S_0, r = r, Sigma = Sigma, T_Maturity = T_Maturity,
                                NoOfPaths = N_End, NoOfSteps = N_Grid)
S_T_All     = S_AllPaths[:, -1]  # Terminal stock prices, dim=N_End✕1
Payoffs_All = np.maximum(S_T_All - K, 0)  # Terminal payoffs, dim=N_End✕1


#---------------------------------- Pre-compute running MC estimate per frame ------------------------------------------
PathCounts   = np.arange(N_Start, N_End + 1, N_Step)  # [N_Start, N_Start + N_Step, ..., N_End]
MC_Estimates = np.zeros(len(PathCounts))
MC_StdErrors = np.zeros(len(PathCounts))

for idx, N in enumerate(PathCounts):
    Payoffs_N = Payoffs_All[:N]
    MC_Estimates[idx] = np.exp(-r * T_Maturity) * np.mean(Payoffs_N)
    MC_StdErrors[idx] = np.exp(-r * T_Maturity) * np.std(Payoffs_N, ddof = 1) / np.sqrt(N)




#=======================================================================================================================
#-------------------------------------------------- Animation Loop -----------------------------------------------------
#=======================================================================================================================
### Create frames folder
Frames_Folder = 'Frames'
os.makedirs(Frames_Folder, exist_ok = True)

N_Frames = len(PathCounts)
Frame_Paths = []

for i, N_Curr in enumerate(PathCounts):

    V_Hat_Curr = MC_Estimates[i]
    SE_Curr = MC_StdErrors[i]
    AbsError = np.abs(V_Hat_Curr - ExactPrice)

    #--------------------------------------- Left Panel: GBM Paths ------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 5))

    for j in range(N_Curr):
        Color = 'green' if S_T_All[j] >= K else 'red'
        ax1.plot(TimeGrid, S_AllPaths[j, :], color = Color, linewidth = 0.7, alpha = 0.6)

    ax1.axhline(y = K, color = 'black', linestyle = '--', linewidth = 1.2, label = rf"Strike $K = {K}$")  # Strike
    ax1.set_title(rf"GBM Paths  ($N = {N_Curr}$)"
                  "\n"
                  rf"($S_0 = {S_0}$, $r = {r}$, $\sigma = {Sigma}$, $T = {T_Maturity}$)")
    ax1.set_xlabel(r"Time $t$")
    ax1.set_ylabel(r"$S(t)$")
    ax1.set_xlim([0, T_Maturity])
    ax1.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
    ax1.legend(loc = 'upper left')

    #--------------------------------------- Right Panel: MC Convergence ------------------------------------------
    ax2.plot(PathCounts[:i + 1], MC_Estimates[:i + 1], color = 'dodgerblue', linewidth = 1.5,
             label = r"MC estimate $\hat{V}(S_0, t_0)$")
    ax2.fill_between(PathCounts[:i + 1],
                     MC_Estimates[:i + 1] - 1.96 * MC_StdErrors[:i + 1],
                     MC_Estimates[:i + 1] + 1.96 * MC_StdErrors[:i + 1],
                     color = 'dodgerblue', alpha = 0.2, label = r"95% CI")
    ax2.plot(PathCounts[i], V_Hat_Curr, marker = 'o', markersize = 7, color = 'magenta', zorder = 5,
             label = rf"Current: $\hat{{V}}_0 = {V_Hat_Curr:.3f}$")
    ax2.axhline(y = ExactPrice, color = 'black', linestyle = '--', linewidth = 1.2,
                label = rf"Exact BS price $= {ExactPrice:.3f}$")
    ax2.set_title(rf"MC Convergence  ($N = {N_Curr}$)"
                  "\n"
                  rf"($|\hat{{V}}_0 - V_0| = {AbsError:.3f}$,  StandardError $= {SE_Curr:.3f}$)")
    ax2.set_xlabel(r"Number of Paths $N$")
    ax2.set_ylabel(r"Estimated Call Price $\hat{V}(S_0, t_0)$")
    ax2.set_xlim([PathCounts[0], PathCounts[-1]])
    ax2.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
    ax2.legend(loc = 'upper right')

    plt.tight_layout()

    #--------------------------------------- Save Frame -----------------------------------------------------------
    Frame_Path = os.path.join(Frames_Folder,
                              f"frame_MC_Convergence_nStart={N_Start}_nEnd={N_End}_nStep={N_Step}_{i:03d}.png")
    plt.savefig(Frame_Path, dpi = 150)
    plt.close(fig)
    Frame_Paths.append(Frame_Path)
    print(f"Frame {i + 1}/{N_Frames} completed  (N = {N_Curr},  |error| = {AbsError:.4f})")




#=======================================================================================================================
#-------------------------------------------------- Assemble GIF -------------------------------------------------------
#=======================================================================================================================
Images = [imageio.imread(fp) for fp in Frame_Paths]
imageio.mimsave(Gif_Filename, Images, duration = Frame_Duration, loop = 0)
print(f"GIF saved to {Gif_Filename}")