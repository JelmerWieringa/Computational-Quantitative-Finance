########################################################################################################################
######################################## COMPUTATIONAL FINANCE - ANIMATION (GIF) #######################################
############################################# BACKWARD INDUCTION (LSM) #################################################
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os


#=======================================================================================================================
#----------------------------------------------- Animation Settings ----------------------------------------------------
#=======================================================================================================================
N_Paths        = 2000          # Number of GBM paths
m              = 50            # Number of time steps in [T_0, T_2]
n_Terms        = 3             # Number of monomial basis terms for OLS regression
N_Grid_Smooth  = 300           # Resolution of smooth curve grid for left panel
Frame_Duration = 600           # Milliseconds per frame in the GIF
Random_Seed    = 23            # Fixed random seed
Gif_Filename   = f'CF_Animation_LSM_BackwardInduction_N={N_Paths}_m={m}.gif'




#=======================================================================================================================
#----------------------------------------------- Model Parameters ------------------------------------------------------
#=======================================================================================================================
S_0   = 1       # Initial stock price
K     = S_0     # Strike price  K = S(T_0) := S_0
r     = 0.06    # Risk-free interest rate
d     = 0.03    # Continuous dividend yield  (makes early exercise optimal for the call)
Sigma = 0.2     # Volatility
T_0   = 0       # Initial time
T_1   = 3       # End of vesting period / start of exercise period
T_2   = 10      # Maturity / expiration date




#=======================================================================================================================
#----------------------------------------------- Time Grid -------------------------------------------------------------
#=======================================================================================================================
Delta_t = (T_2 - T_0) / m
TimeGrid = np.linspace(T_0, T_2, m + 1)  # {t_0=0, t_1, ..., t_m=T_2}
Index_T_1 = int(round(T_1 / Delta_t))  # Index of T_1 in TimeGrid

if not np.isclose(TimeGrid[Index_T_1], T_1):
    raise ValueError(f"T_1={T_1} does not fall on a grid point. Adjust m.")




#=======================================================================================================================
#----------------------------------------------- Core Functions --------------------------------------------------------
#=======================================================================================================================

#----------------------------------------------- Payoff Function -------------------------------------------------------
def H_PayoffFunction(S, K):
    r"""
    GOAL: Define the payoff function H(S) := max{S - K, 0}.
    :param S:  Stock price(s).
    :param K:  Strike price.
    """
    return np.maximum(S - K, 0)


#--------------------------------------------- ITM Indicator -----------------------------------------------------------
def ITM_Paths_Index(StockPrices, K):
    r"""
    GOAL: Return boolean array indicating which paths are ITM at a given time step.
    :param StockPrices:  Array of stock prices across N paths at time t_i.
    :param K:            Strike price.
    """
    return H_PayoffFunction(S = StockPrices, K = K) > 0


#---------------------------------------------- GBM Simulation ---------------------------------------------------------
def SimulateGBM(S_0, r, d, Sigma, Delta_t, N, m):
    r"""
    GOAL: Simulate N exact GBM paths on a grid of m steps with dividend-adjusted drift r - d.
    :param S_0:     Initial stock price.
    :param r:       Risk-free interest rate.
    :param d:       Continuous dividend yield.
    :param Sigma:   Volatility.
    :param Delta_t: Time-step size.
    :param N:       Number of paths.
    :param m:       Number of time steps.
    """
    np.random.seed(Random_Seed)
    Z = np.random.standard_normal((N, m))

    S = np.ones((N, m + 1)) * S_0
    for i in range(m):
        S[:, i + 1] = S[:, i] * np.exp((r - d - 0.5 * Sigma**2) * Delta_t + Sigma * np.sqrt(Delta_t) * Z[:, i])
    return S


#----------------------------------------------- Monomial Basis --------------------------------------------------------
def BasisFunction_Monomial(S, n_Terms):
    r"""
    GOAL: Compute the monomial basis {1, S, S^2, ..., S^{n_Terms - 1}}. Shape: (N, n_Terms).
    :param S:        Stock prices at time t_i, shape (N_ITM,).
    :param n_Terms:  Number of basis terms.
    """
    return np.column_stack([S**k for k in range(n_Terms)])


#--------------------------------------------- Exercise Boundary -------------------------------------------------------
def ComputeExerciseBoundary(ExercisePaths, S_i):
    r"""
    GOAL: Estimate the exercise boundary S*(t_i) as the minimum stock price among paths exercising at t_i.
    :param ExercisePaths:  Boolean array of shape (N,) indicating which paths exercise at t_i.
    :param S_i:            Stock prices at t_i, shape (N,).
    """
    if ExercisePaths.sum() == 0:
        return None
    return S_i[ExercisePaths].min()




#=======================================================================================================================
#------------------------------------ Backward Induction with Animation Data Storage -----------------------------------
#=======================================================================================================================
def BackwardInductionRoutine_Animated(S, K, r, Delta_t, Index_T_1, n_Terms):
    r"""
    GOAL: Run the LSM backward induction routine and store per-step regression data for animation.
          At each exercise step t_i the following are stored: stock prices, payoffs, continuation
          values, exercise decisions, and OLS beta parameters for the smooth continuation curve.
    :param S:           GBM paths, shape (N, m+1).
    :param K:           Strike price.
    :param r:           Risk-free interest rate.
    :param Delta_t:     Time-step size.
    :param Index_T_1:   Column index of T_1 in the time grid.
    :param n_Terms:     Number of monomial basis terms.
    """
    N = S.shape[0]
    m = S.shape[1] - 1

    CashFlow           = np.zeros((N, m))
    CashFlow[:, m - 1] = H_PayoffFunction(S[:, m], K)
    Future_CashFlow    = CashFlow[:, m - 1].copy()

    StepData_List = []   # StepData_List[0] = data at t_{m-1}, StepData_List[-1] = data at t_{Index_T_1}

    for i in range(m - 1, 0, -1):   # i = m-1, m-2, ..., 1
        Future_CashFlow *= np.exp(-r * Delta_t)   # Discount back from t_{i+1} to t_i

        if i < Index_T_1:   # Vesting period: exercise is forbidden
            continue

        #------------------------------------------ Regression at t_i ------------------------------------------------
        ITM_Paths = ITM_Paths_Index(StockPrices = S[:, i], K = K)
        H_i       = H_PayoffFunction(S = S[:, i], K = K)

        if ITM_Paths.sum() >= n_Terms:
            X_ITM         = S[:, i][ITM_Paths]
            Psi           = BasisFunction_Monomial(X_ITM, n_Terms)
            Y_ITM         = Future_CashFlow[ITM_Paths]
            Beta, _, _, _ = np.linalg.lstsq(Psi, Y_ITM, rcond = None)
            c_Hat             = np.zeros(N)
            c_Hat[ITM_Paths]  = Psi @ Beta
        else:
            Beta  = np.zeros(n_Terms)
            c_Hat = np.zeros(N)

        #----------------------------------------- Exercise Decision at t_i ------------------------------------------
        ExercisePaths_ITM        = H_i[ITM_Paths] >= c_Hat[ITM_Paths]
        ExercisePaths            = np.zeros(N, dtype = bool)
        ExercisePaths[ITM_Paths] = ExercisePaths_ITM

        #------------------------------------------ Store Step Data ---------------------------------------------------
        StepData_List.append({
            'time_index':    i,
            'time':          TimeGrid[i],
            'S_i':           S[:, i].copy(),
            'H_i':           H_i.copy(),
            'c_Hat':         c_Hat.copy(),
            'ExercisePaths': ExercisePaths.copy(),
            'ITM_Paths':     ITM_Paths.copy(),
            'Beta':          Beta.copy(),
            'N_Exercise':    int(ExercisePaths.sum()),
        })

        #------------------------------------- Update Cash Flow Matrix & Future Value ----------------------------------
        CashFlow[:, i - 1]          = np.where(ExercisePaths, H_i, 0)
        CashFlow[ExercisePaths, i:] = 0
        Future_CashFlow             = np.where(ExercisePaths, H_i, Future_CashFlow)

    return CashFlow, StepData_List




#=======================================================================================================================
#--------------------------------------------- Pre-Compute Animation Data ----------------------------------------------
#=======================================================================================================================

#---------------------------------------------- Simulate GBM Paths ----------------------------------------------------
StockPricePaths = SimulateGBM(S_0 = S_0, r = r, d = d, Sigma = Sigma, Delta_t = Delta_t, N = N_Paths, m = m)


#--------------------------------------- Run Annotated Backward Induction ----------------------------------------------
CashFlowMatrix, StepData_List = BackwardInductionRoutine_Animated(S = StockPricePaths, K = K, r = r, Delta_t = Delta_t,
                                                                  Index_T_1 = Index_T_1, n_Terms = n_Terms)


#--------------------------------------------- Global Axis Limits -----------------------------------------------------
S_Min_Global  = min(Step['S_i'].min() for Step in StepData_List)
S_Max_Global  = max(Step['S_i'].max() for Step in StepData_List)
H_Max_Global  = max(Step['H_i'].max() for Step in StepData_List)
S_Grid_Smooth = np.linspace(max(S_Min_Global, 1e-6), S_Max_Global, N_Grid_Smooth)


#-------------------------------------- Pre-Compute Exercise Boundary S^*(t) ------------------------------------------
ExerciseBoundary = []  # List of {'time': t_i, 'S_Star': S^*(t_i)} for each exercise step
for Step in StepData_List:
    S_Star = ComputeExerciseBoundary(ExercisePaths = Step['ExercisePaths'], S_i = Step['S_i'])
    ExerciseBoundary.append({'time': Step['time'], 'S_Star': S_Star})




#=======================================================================================================================
#-------------------------------------------------- Animation Loop -----------------------------------------------------
#=======================================================================================================================
### Create frames folder
Frames_Folder = 'Frames'
os.makedirs(Frames_Folder, exist_ok = True)

N_Frames    = len(StepData_List)
Frame_Paths = []

for i, Step in enumerate(StepData_List):

    t_i           = Step['time']
    S_i           = Step['S_i']
    H_i           = Step['H_i']
    ITM_Paths     = Step['ITM_Paths']
    ExercisePaths = Step['ExercisePaths']
    Beta          = Step['Beta']
    N_Exercise    = Step['N_Exercise']

    #--------------------------------------- Left Panel: Regression at t_i ----------------------------------------
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 5))

    ### Smooth continuation value and payoff curves
    Psi_Smooth = BasisFunction_Monomial(S_Grid_Smooth, n_Terms)
    c_Smooth   = Psi_Smooth @ Beta
    H_Smooth   = np.maximum(S_Grid_Smooth - K, 0)

    ax1.plot(S_Grid_Smooth, H_Smooth, color = 'dodgerblue', linewidth = 2.0, linestyle = '-',
             label = r"Payoff $H(S) = \max\{S - K,\, 0\}$")
    ax1.plot(S_Grid_Smooth, c_Smooth, color = 'darkorange', linewidth = 2.0, linestyle = '--',
             label = r"Continuation $\hat{c}(t_i,\, S)$")

    ### Scatter: OTM (red), ITM-continue (magenta), ITM-exercise (green)
    OTM_Paths = ~ITM_Paths
    ax1.scatter(S_i[OTM_Paths], H_i[OTM_Paths], s = 5, color = 'red', alpha = 0.5, zorder = 2)
    ax1.scatter(S_i[ITM_Paths & ~ExercisePaths], H_i[ITM_Paths & ~ExercisePaths], s = 8,  color = 'magenta',
                alpha = 0.6, zorder = 3, label = "Continue")
    ax1.scatter(S_i[ExercisePaths], H_i[ExercisePaths], s = 8, color = 'green', alpha = 0.7, zorder = 4,
                label = "Exercise")

    ### Exercise boundary vertical line
    S_Star_i = ExerciseBoundary[i]['S_Star']
    if S_Star_i is not None:
        ax1.axvline(x = S_Star_i, color = 'black', linestyle = ':', linewidth = 1.2,
                    label = rf"$S^* \approx {S_Star_i:.3f}$")

    ax1.set_title(rf"Regression at $t_i = {t_i:.2f}$,  {N_Exercise} paths exercise"
                  "\n"
                  rf"($T_1={T_1}$, $T_2={T_2}$, $K={K}$, $r={r}$, $d={d}$, $\sigma={Sigma}$)")
    ax1.set_xlabel(r"Stock Price $S(t_i)$")
    ax1.set_ylabel(r"Value")
    ax1.set_xlim([S_Min_Global, S_Max_Global])
    ax1.set_ylim([-0.02, H_Max_Global * 1.15])
    ax1.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
    ax1.legend(loc = 'upper left', fontsize = 8)

    #--------------------------------------- Right Panel: Exercise Boundary Accumulating ------------------------------
    ### Collect boundary points revealed so far: steps 0 to i (t_{m-1} down to t_i)
    Times_Revealed   = [ExerciseBoundary[j]['time']   for j in range(i + 1)
                        if ExerciseBoundary[j]['S_Star'] is not None]
    S_Stars_Revealed = [ExerciseBoundary[j]['S_Star'] for j in range(i + 1)
                        if ExerciseBoundary[j]['S_Star'] is not None]

    ax2.axhline(y = K,   color = 'black', linestyle = '--', linewidth = 1.2, label = rf"Strike $K = {K}$")
    ax2.axvline(x = T_1, color = 'gray',  linestyle = ':',  linewidth = 1.0, label = rf"$T_1 = {T_1}$")
    ax2.axvline(x = t_i, color = 'purple', linestyle = '--', linewidth = 1.2, label = rf"Current $t_i = {t_i:.2f}$")

    if len(Times_Revealed) > 0:
        SortedPairs    = sorted(zip(Times_Revealed, S_Stars_Revealed))
        Times_Sorted   = [p[0] for p in SortedPairs]
        S_Stars_Sorted = [p[1] for p in SortedPairs]
        ax2.plot(Times_Sorted, S_Stars_Sorted, color = 'darkorange', linewidth = 1.5, alpha = 0.7)
        ax2.scatter(Times_Revealed, S_Stars_Revealed, color = 'darkorange', s = 20, zorder = 4, label = r"$S^*(t_i)$")

    ax2.set_title(rf"Exercise Boundary $S^*(t)$ — Backward from $T_2 = {T_2}$"
                  "\n"
                  rf"($N = {N_Paths}$, $m = {m}$, $n_{{\mathrm{{Terms}}}} = {n_Terms}$)")
    ax2.set_xlabel(r"Time $t$")
    ax2.set_ylabel(r"Exercise Boundary $S^*(t)$")
    ax2.set_xlim([T_1, T_2])
    ax2.set_ylim([S_Min_Global, S_Max_Global])
    ax2.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
    ax2.legend(loc = 'upper left', fontsize = 8)

    plt.tight_layout()

    #--------------------------------------- Save Frame -----------------------------------------------------------
    Frame_Path = os.path.join(Frames_Folder, f"frame_LSM_BackwardInduction_N={N_Paths}_m={m}_{i:03d}.png")
    plt.savefig(Frame_Path, dpi = 150)
    plt.close(fig)
    Frame_Paths.append(Frame_Path)
    print(f"Frame {i + 1}/{N_Frames} completed (t_i = {t_i:.2f},  N_Exercise = {N_Exercise})")




#=======================================================================================================================
#-------------------------------------------------- Assemble GIF -------------------------------------------------------
#=======================================================================================================================
Images = [imageio.imread(fp) for fp in Frame_Paths]
imageio.mimsave(Gif_Filename, Images, duration = Frame_Duration, loop = 0)
print(f"GIF saved to {Gif_Filename}")