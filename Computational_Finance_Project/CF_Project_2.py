########################################################################################################################
############################################ COMPUTATIONAL FINANCE - PROJECT ###########################################
######################################## QUESTION 2 - ADVANCED NUMERICAL EXAMPLE #######################################
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
def ConstructTimeGrid(T_2, m):
    r"""
    GOAL: Construct equidistant time grid {t_i}_{i=0}^{m} over [0, T2], where t_0=0 and t_m=T_2.
    :param T_2:  Time of maturity/expiration.
    :param m:    Number of time steps.
    """
    return np.linspace(0, T_2, m + 1)   # [t_0=0, t_1, ..., t_{m-1}, t_m=T2]

TimeGrid = ConstructTimeGrid(T_2 = T_2, m = m)


#------------------------------------------------- Obtain Index of T_1 -------------------------------------------------
def IndexExerciseStart(time_grid, T_1):
    r"""
    GOAL: Find the index in the time_grid corresponding to T_1. Ensures that T_1 falls exactly on a grid point.
    :param time_grid:  Grid of time-points on [T_0, T_2].
    :param T_1:        Time point at which the exercise period starts.
    """
    Index_T_1 = np.argmin(np.abs(time_grid - T_1))  # Closest time point in terms of minimised absolute difference

    # Checking whether the points lie within the default tolerance of 1e-05
    if np.isclose(time_grid[Index_T_1], T_1) == False:
        raise ValueError(rf"$T_1$={T_1} does not fall on a grid point. Adjust m.")

    return Index_T_1

Index_T_1 = IndexExerciseStart(time_grid = TimeGrid, T_1 = T_1)


#---------------------------------- Define Payoff Function of Call Option: H(S(t_i)) -----------------------------------
def H_PayoffFunction(S, K):
    r"""
    GOAL: Define the payoff function H(S) := max{S - K, 0}.
    :param S:  Stock price.
    :param K:  Strike price.
    """
    return np.maximum(S - K, 0)


#-------------------------------------------- Define ITM Checking Function ---------------------------------------------
def ITM_Paths_Index(StockPrices, K):
    r"""
    GOAL: Select stocks that are ITM (in-the-money) at a single time step t_i & return a boolean array.
    :param StockPrices:  Array of stock prices across N paths at time t_i.
    :param K:            Strike price.
    """
    # Select paths that are ITM, i.e., H(S(t_i)) = max{S(t_i) - K, 0} > 0
    return H_PayoffFunction(S = StockPrices, K = K) > 0  # dim=Nx1, when dim(StockPrices)=Nx1




############################################## Generate Stock Price Paths ##############################################
########################################################################################################################
def SimulateGBM(S_0, drift, sigma, delta_t, N, m, default_seed=True):
    r"""
    GOAL:  Simulates N exact GBM paths on a grid of m steps with time-step size delta_t.
           The dimension of S is N✕(m+1), with S_0 = S[:, 0].
    :param S_0:           Initial stock price.
    :param r:             Risk-free rate.
    :param sigma:         Volatility.
    :param delta_t:       Time-step size.
    :param N:             Number of paths.
    :param m:             Number of steps/time points in grid.
    :param default_seed:  A default seed of 23 is set for random number generation.
    """
    if default_seed == True:
        np.random.seed(23)

    Z = np.random.standard_normal((N, m))  # Generate the same realizations of standard BM
    S = np.ones((N, m + 1)) * S_0
    for i in range(m):  # Compute the exact solution of GBM at each t_1,...,t_m. For all paths simulateanously.
        S[:, i+1] = S[:, i] * np.exp((drift - 0.5 * sigma**2) * delta_t + sigma * np.sqrt(delta_t) * Z[:, i])

    return S

StockPricePaths = SimulateGBM(S_0 = S_0, drift = r, sigma = Sigma, delta_t = Delta_t, N = N, m = m)




################################################ Define Basis Functions ################################################
########################################################################################################################
def BasisFunction_Monomial(S, n_Terms):
    r"""
    GOAL: Compute the monomial basis function: {1, S, S^2,..., S^{n_Terms-1}}. dim=N✕n_Terms.
    :param S:        N paths of stock prices at time t_i.
    :param n_Terms:  Number of monomial terms. E.g.: n_Terms = 3 ==> {1, S, S^2}.
    """
    return np.column_stack([S**k for k in range(n_Terms)])

# BasisFunction_Monomial(StockPricePaths[:, 2], n_Terms = 3)


def BasisFunction_LaguerrePolynomials_Unstandardized(S, n_Terms):
    r"""
    GOAL: Compute the weighted Laguerre polynomials for the basis: {L_0(S), L_1(S), ..., L_{n_Terms-1}(S)}, with
          L_n(S) := exp(-S/2) * \tilde{L}_n(S) where \tilde{L}_n is the general Laguerre polynomial.
          Using the recurrence relation: \tilde{L}_0(S) = 1, \tilde{L}_1(S) = 1-S,
                                         \tilde{L}_{n+1}(S) = [(2n+1-S)*\tilde{L}_n(S) - n*\tilde{L}_{n-1}(S)] / (n+1).
    :param S:        N paths of stock prices at time t_i.
    :param n_Terms:  Number of Laguerre terms.
    """
    L_Tilde = np.zeros((len(S), n_Terms))  # dim=N✕n_Terms
    L_Tilde[:, 0] = 1  # \tilde{L}_0(S) = 1
    if n_Terms > 1:
        L_Tilde[:, 1] = 1 - S  # \tilde{L}_1(S) = 1 - S
    for k in range(1, n_Terms - 1):
        L_Tilde[:, k+1] = ((2*k + 1 - S) * L_Tilde[:, k] - k * L_Tilde[:, k-1]) / (k + 1)

    Weight = np.exp(-S / 2)   # Weighting factor (used in the LSM paper)
    return Weight[:, None] * L_Tilde  # Weighted Laguerre basis


def BasisFunction_LaguerrePolynomials_Standardized(S, n_Terms):
    r"""
    GOAL: Compute the weighted Laguerre polynomials for the basis: {L_0(S), L_1(S), ..., L_{n_Terms-1}(S)}, with
          L_n(S) := exp(-S/2) * \tilde{L}_n(S) where \tilde{L}_n is the general Laguerre polynomial.
          Using the recurrence relation: \tilde{L}_0(S) = 1, \tilde{L}_1(S) = 1-S,
                                         \tilde{L}_{n+1}(S) = [(2n+1-S)*\tilde{L}_n(S) - n*\tilde{L}_{n-1}(S)] / (n+1).
    :param S:        N paths of stock prices at time t_i.
    :param n_Terms:  Number of Laguerre terms.
    """
    # Normalise by cross-sectional mean, because the weight implies that the basis functions are nearly zero for large   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # stock prices. Hence, the design matrix Psi becomes unreliable for regression.
    S_Scaled = S / np.mean(S)

    L_Tilde = np.zeros((len(S_Scaled), n_Terms))  # dim=N✕n_Terms
    L_Tilde[:, 0] = 1  # \tilde{L}_0(S) = 1
    if n_Terms > 1:
        L_Tilde[:, 1] = 1 - S_Scaled   # \tilde{L}_1(S) = 1 - S
    for k in range(1, n_Terms - 1):
        L_Tilde[:, k+1] = ((2*k + 1 - S_Scaled) * L_Tilde[:, k] - k * L_Tilde[:, k-1]) / (k + 1)

    Weight = np.exp(-S_Scaled / 2)   # Weighting factor (used in the LSM paper)
    return Weight[:, None] * L_Tilde  # Weighted Laguerre basis


def BasisFunction_LegendrePolynomials(S, n_Terms):
    """
    GOAL: Compute the Legendre basis on normalised S in [-1, 1]: {P_0(S), P_1(S), ..., P_{n_Terms-1}(S)}.
          Using the recurrence relation: P_0(x)=1, P_1(x)=x, P_{n+1}(x) = [(2n+1)*x*P_n(x) - n*P_{n-1}(x)] / (n+1).
    :param S:        N paths of stock prices at time t_i.
    :param n_Terms:  Number of Legendre terms.
    """
    # First, normalise S to [-1, 1]
    S_Min = S.min()
    S_Max = S.max()
    S_Scaled = 2 * (S - S_Min) / (S_Max - S_Min) - 1  # Maps to [-1, 1]

    P = np.zeros((len(S_Scaled), n_Terms))  # dim=N✕n_Terms
    P[:, 0] = 1
    if n_Terms > 1:
        P[:, 1] = S_Scaled
    for k in range(1, n_Terms - 1):
        P[:, k+1] = ((2*k + 1) * S_Scaled * P[:, k] - k * P[:, k-1]) / (k + 1)
    return P


def BasisFunction_ChebyshevPolynomials(S, n_Terms):
    r"""
    GOAL: Compute the Chebyshev polynomials (of first kind) for the basis: {T_0(S), T_1(S), ..., T_{n_Terms-1}(S)}.
          Using the recurrence relation: T_0(x) = 1, T_1(x) = x, T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x) for n>1.
    :param S:        N paths of stock prices at time t_i.
    :param n_Terms:  Number of Chebyshev terms.
    """
    T = np.zeros((len(S), n_Terms))  # dim=N✕n_Terms
    T[:, 0] = 1  # T_0(x) = 1
    if n_Terms > 1:
        T[:, 1] = S  # T_1(x) = x
    for k in range(2, n_Terms):
        T[:, k] = 2 * S * T[:, k - 1] - T[:, k - 2]  # Recurrence relation
    return T


def BasisFunction_Trigonometric(S, n_Terms):
    r"""
    GOAL: Compute a trigonometric basis: {1, sin(S), cos(S), sin(2S), cos(2S), ...}.
    :param S:        N paths of stock prices at time t_i.
    :param n_Terms:  Number of terms.
    Note: Angular frequency is implicitly set to one.
    """
    # First, normalise S to [0, 2*pi]
    S_Min = S.min()
    S_Max = S.max()
    S_scaled = 2 * np.pi * (S - S_Min) / (S_Max - S_Min)  # Maps to [0, 2*pi]

    Columns_Basis = [np.ones(len(S))]
    for k in range(1, (n_Terms + 1) // 2 + 1):
        Columns_Basis.append(np.sin(k * S_scaled))
        Columns_Basis.append(np.cos(k * S_scaled))

    return np.column_stack(Columns_Basis[:n_Terms])




######################################## Regression at Single Exercise Date t_i ########################################
########################################################################################################################

def RegressionContinuation(S_i, Y, K, basis_function, n_Terms):
    r"""
    GOAL: Estimate the continuation value c(t_i) for all N paths by fitting a linear combination of basis functions to
          the discounted future cash flows of ITM paths via OLS (cross-sectional linear regression).
    :param S_i:             N paths of stock prices at time t_i.
    :param Y:               Discounted future cash flows for N paths.
    :param K:               Strike price.
    :param basis_function:  Basis function \psi used in linear regression at time t_i.
                            Note: We assume that c(t_i)=f(X_i)=E[Y_i|X_i] is a linear function in {\psi_1(X),..., \psi_M}.
    :param n_Terms:         Number of basis functions, i.e., M.
    """
    ITM = ITM_Paths_Index(StockPrices = S_i, K = K)  # Index of the paths that are ITM

    # Check if the system is underdetermined, because OLS requires at least as many observations as (beta) parameters
    if ITM.sum() < n_Terms:  # N_{ITM} = number of ITM paths = ITM.sum()
        return np.zeros(len(S_i))  # Skip regression, return zero continuation values

    X_ITM = S_i[ITM]  # Only perform the regression on the ITM paths
    Psi = basis_function(X_ITM, n_Terms)  # Matrix with basis functions (columns) evaluated at the ITM paths (rows).
    Y_ITM = Y[ITM]  # (Y_1,..., Y_{N_{ITM}})^T
    EstimatedBetaParameters, _, _, _ = np.linalg.lstsq(Psi, Y_ITM, rcond=None)  # \hat{\beta}
    # np.linalg.lstsq() returns the least-squares solution to a linear matrix equation. I.e., it finds/approximates the
    # \hat{beta} that minimizes || Psi @ beta - Y ||^2.
    # Note: We do not use the analytical solution of the OLS estimator, because np.linalg.inv() is numerically unstable.
    #       Further, we prefer np.linalg.lstsq() over np.linalg.solve() because it is numerically more stable.

    c_Hat = np.zeros(len(S_i))
    c_Hat[ITM] = Psi @ EstimatedBetaParameters  # Fitted value: \hat{Y} = Psi \hat{\beta}
    return c_Hat




############################################## Backward Induction Routine ##############################################
########################################################################################################################

def BackwardInductionRoutine(S, K, r, delta_t, index_T_1, basis_function, n_Terms):
    r"""
    GOAL: Perform the LSM backward induction over [T_0, T_2]. In the exercising period, the optimal strategy is
          determined for each path by comparing immediate exercise against the regression-estimated continuation value
          at each exercise date.
    :param S:               N paths of stock prices on {t_0, ..., t_m} time grid.
    :param K:               Strike price.
    :param r:               Risk-free rate.
    :param delta_t:         Time-step size.
    :param index_T_1:       Index of T_1, i.e., start of the exercising period.
    :param basis_function:  Basis function \psi used in linear regression.
    :param n_Terms:         Number of basis functions, i.e., M.
    """
    N = S.shape[0]  # Number of paths
    m = S.shape[1] - 1  # Number of steps. S has m+1 columns (including t_0), hence m steps

    CashFlow = np.zeros((N, m))  # Cash flow matrix (no t_0 column)
    # CF[j, i] = undiscounted payoff received at t_{i+1} (0 if not exercised there), for path j

    # When t_i=t_m=T_2 (i.e., at maturity), the value of the option V(t_i, S(t_i)) at t_i is the payoff function at
    #   maturity: H(S(T_2)) = max{S(T_2) - K, 0}.
    CashFlow[:, m - 1] = H_PayoffFunction(S[:, m], K)


    #-------------------------------- Computing the Cash Flow with Backward Induction ----------------------------------
    # At the start of iteration i, we use Future_CashFlow[j] to store the optimal future cash flow of path j, discounted
    #   to time t_i.
    # Note: We do not scan all the columns to the right of the t_i column in the CashFlow matrix, find the non-zero
    #       entry for each path, and discount it back to t_i. Instead, we use Future_CashFlow.
    # Note: To determine the optimal strategy at t_i, we need to know the optimal strategy at t_{i+1},...,t_m

    Future_CashFlow = CashFlow[:, m - 1].copy()  # Copy the columns with terminal payoff H(S(T_2))
    # Note: We do not want to modify CashFlow.

    for i in range(m - 1, 0, -1):  # i = m-1, m-2, ..., 1
        Future_CashFlow *= np.exp(-r * delta_t)  # Discounting the cash flows back from t_{i+1} to t_i

        if i < index_T_1:  # Check: In the vesting period, exercising is not allowed
            continue

        #--------------------------------------- Regression & Exercise Decision ----------------------------------------
        c_Hat = RegressionContinuation(S_i = S[:, i], Y = Future_CashFlow, K = K, basis_function = basis_function,
                                       n_Terms = n_Terms)
        H_i = H_PayoffFunction(S = S[:, i], K = K)

        ITM_Paths = ITM_Paths_Index(StockPrices = S[:, i], K = K)  # Select ITM paths
        H_i = H_PayoffFunction(S = S[:, i], K = K)
        # Select paths that are ITM, and for which immediate exercise is at least as good as continuation.
        ExercisePaths_ITM = H_i[ITM_Paths] >= c_Hat[ITM_Paths]
        ExercisePaths = np.zeros(N, dtype=bool)
        ExercisePaths[ITM_Paths] = ExercisePaths_ITM

        #---------------------------------------- Updating the Optimal Strategy ----------------------------------------
        # For exercising paths: store the immediate payoff H(S(t_i)) in the t_i column (i.e., column i-1) of the
        #                       CashFlow matrix.
        # For non-exercising paths: store a zero.
        CashFlow[:, i - 1] = np.where(ExercisePaths, H_i, 0)

        # For paths that exercise at t_i, zero out all columns to the right in the corresponding rows.
        CashFlow[ExercisePaths, i:] = 0

        # For exercising path j: replace Future_CashFlow[j] with H(S(t_i)).
        # Keep the rows in Future_CashFlow corresponding to non-exercising paths unchanged.
        Future_CashFlow = np.where(ExercisePaths, H_i, Future_CashFlow)

    return CashFlow




############################################## Compute ESO Price at Grant ##############################################
########################################################################################################################

def ESO_PriceAtGrant(CashFlowMatrix, r, time_grid):
    r"""
    GOAL: Compute the (estimated) ESO price at grant (T_0 = t_0), i.e., the valuation at t_0: \hat{V}_0.
    :param CashFlowMatrix:  The cash flow matrix, with columns corresponding to {t_1,...,t_m}.
    :param r:               Risk-free rate.
    :param time_grid:       Grid of time-points on [T_0, T_2].
    """
    DiscountFactors = np.exp(-r * time_grid[1:])  # [e^{-r*1}, ..., e^{-r*m}], dim=m✕1
    # Note: [1:] excludes the first column corresponding to t_0.

    # Next, discount each column of the cash flow matrix back to time t_0.
    # In particular, multiply the i^{th} column (corresponding to t_i) by e^{-r*t_i}
    PresentsValue_Paths = CashFlowMatrix @ DiscountFactors  # dim=N✕1
    V_0_Hat = np.mean(PresentsValue_Paths)

    SE = np.std(PresentsValue_Paths, ddof=1) / np.sqrt(len(PresentsValue_Paths))  # standard error !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Note: "ddof=1" ensures that 1/(N-1) instead of 1/N is used to ensure unbiasedness of the estimator.
    CriticalValue = norm.ppf(0.975)  # Inverse CDF evaluated at 0.975, approximately 1.96
    CI_LowerBound = V_0_Hat - CriticalValue * SE
    CI_UpperBound = V_0_Hat + CriticalValue * SE

    return {"Price": V_0_Hat, "CI": [CI_LowerBound, CI_UpperBound]}





###################################################### Simulation ######################################################
########################################################################################################################

#----------------------------------------------- Compare Different Bases -----------------------------------------------
def CompareBases():

    Basis = [("Monomial",                   BasisFunction_Monomial,                             3),
             ("Monomial",                   BasisFunction_Monomial,                             5),
             ("Laguerre",                   BasisFunction_LaguerrePolynomials_Unstandardized,   3),
             ("Laguerre",                   BasisFunction_LaguerrePolynomials_Unstandardized,   5),
             ("Laguerre (standardized)",    BasisFunction_LaguerrePolynomials_Standardized,     3),
             ("Laguerre (standardized)",    BasisFunction_LaguerrePolynomials_Standardized,     5),
             ("Legrende",                   BasisFunction_LegendrePolynomials,                  3),
             ("Legrende",                   BasisFunction_LegendrePolynomials,                  5),
             ("Chebyshev",                  BasisFunction_ChebyshevPolynomials,                 3),
             ("Chebyshev",                  BasisFunction_ChebyshevPolynomials,                 5)]

    for Name, BasisFunction, NoOfTerms in Basis:
        CashFlowMatrix = BackwardInductionRoutine(S = StockPricePaths, K = K, r = r, delta_t = Delta_t,
                                                  index_T_1 = Index_T_1,
                                                  basis_function = BasisFunction, n_Terms = NoOfTerms)  # These change
        V_0_Hat_Bases = ESO_PriceAtGrant(CashFlowMatrix = CashFlowMatrix, r = r, time_grid = TimeGrid)

        print(f"{Name} basis with {NoOfTerms} terms: {V_0_Hat_Bases["Price"]:.4f}  &  "
              f"CI = [{V_0_Hat_Bases["CI"][0]:.4f}, {V_0_Hat_Bases["CI"][1]:.4f}]")




#---------------------------------------- Plot Sample GBM Paths over [T_0, T_2] ----------------------------------------
def PlotGBMPaths():
    plt.figure(figsize = (10, 6))

    plt.plot(TimeGrid, np.transpose(StockPricePaths))
    plt.axhline(K, linestyle = '--', linewidth = 1.5, color = 'black', label = "K")  # Indicating the strike price K
    plt.axvline(T_1, linestyle = '--', linewidth = 1.5, color = 'blue', label = r"$T_1$")  # Indicating vesting period

    plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
    plt.title(rf"{N} paths of $S(t)$ on $[T_0, T_2]$ = [{T_0}, {T_2}]")
    plt.xlabel(r"Time $t$")
    plt.ylabel(r"$S(t)$")
    plt.xlim([0, T_2])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"CF_Project_2_GBM_{N}_Paths.png", dpi = 300, bbox_inches = 'tight')
    plt.show()




#-------------------------------------- Convergence Plot & Compare Different Bases -------------------------------------
def ConvergencePlot_CompareBases():
    N_Values = [1000, 5000, 10000, 25000, 50000, 100000]
    n_Terms_List = [1, 3, 5]
    Colours = {1: "steelblue", 3: "darkorange", 5: "green"}  # Corresponding to the elements of n_Terms_List

    Bases = [("Monomials",                   BasisFunction_Monomial),
             ("Legendre",                    BasisFunction_LegendrePolynomials),
             ("Laguerre (unstandardized)",   BasisFunction_LaguerrePolynomials_Unstandardized),
             ("Laguerre (standardized)",     BasisFunction_LaguerrePolynomials_Standardized),
             ("Chebyshev",                   BasisFunction_ChebyshevPolynomials),
             ("Trigonometric",               BasisFunction_Trigonometric)]

    # Simulate with the largest N and re-use subsets. This ensures differences across N are purely due to sample size,
    # not different random seeds.
    StockPricePaths_Max = SimulateGBM(S_0 = S_0, drift = r, sigma = Sigma, delta_t = Delta_t, N = max(N_Values), m = m)

    # Create an empty dictionary to store the data for the plots
    Results = {}  # Results[basis_name][n_terms] = {"V_0": [...], "CI_Lower": [...], "CI_Upper": [...]}


    # Next, compute the ESO price at grant for the different bases (basis function + number of terms)
    #                                    & for different N values.
    for Name, BasisFunction in Bases:
        Results[Name] = {}

        for n_Terms in n_Terms_List:
            V_0_list = []
            CI_Lower_list = []
            CI_Upper_list = []

            for N_i in N_Values:
                S_subset = StockPricePaths_Max[:N_i, :]  # Select N_i paths, for N_i \in N_Values
                CF = BackwardInductionRoutine(S = S_subset, K = K, r = r, delta_t = Delta_t, index_T_1 = Index_T_1,
                                              basis_function = BasisFunction, n_Terms = n_Terms)
                Result = ESO_PriceAtGrant(CashFlowMatrix = CF, r = r, time_grid = TimeGrid)

                # Add ESO price (and corresponding confidence interval) to Results dictionary, for specific N_1
                V_0_list.append(Result["Price"])
                CI_Lower_list.append(Result["CI"][0])
                CI_Upper_list.append(Result["CI"][1])

            # Add ESO price (and corresponding confidence interval) to Results dictionary, for all N_Values
            Results[Name][n_Terms] = {"V_0": np.array(V_0_list),
                                      "CI_Lower": np.array(CI_Lower_list), "CI_Upper": np.array(CI_Upper_list),}


    # Create the final plot
    fig, axes = plt.subplots(3, 2, figsize = (10, 10), sharey = True)  # Figure with 6 subplots
    axes = axes.flatten()  # Transform it into an 1D array of 6 axes, easier to loop over

    for i in range(len(Bases)):
        ax = axes[i]  # Determine the location of the subplot
        Name = Bases[i][0]  # Extract the name of the basis

        for n_Terms in n_Terms_List:
            SpecificBasisResult = Results[Name][n_Terms]  # Select a specific basis: basis function + number of terms
            Colour = Colours[n_Terms]  # Select the colour corresponding to the specific n_Terms value

            ax.plot(N_Values, SpecificBasisResult['V_0'], color = Colour, linewidth = 2, marker = 'o',
                    label = rf"$n_{{\mathrm{{Terms}}}}={n_Terms}$")
            ax.fill_between(N_Values, SpecificBasisResult["CI_Lower"], SpecificBasisResult["CI_Upper"],
                            color = Colour, alpha = 0.15)  # Confidence interval region

        # Add the \hat{V}_0 corresponding to Monomial basis function with 3 terms for comparison
        ax.axhline(y = Results["Monomials"][3]['V_0'][-1], color = 'red', linestyle = '--', linewidth = 1,
                   label = "Monomial Benchmark")

        ax.set_title(Name, fontsize = 12)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.set_xlabel(r"Number of Paths $N$", fontsize = 10)
        ax.set_ylabel(r"$\hat{V}_0$", fontsize = 10)
        ax.set_xscale('log')
        ax.set_xlim([N_Values[0], N_Values[-1]])
        ax.legend(fontsize = 8)

    plt.tight_layout()
    plt.savefig(f"CF_Project_2_ConvergencePlot_BasisComparison.png", dpi = 300, bbox_inches = 'tight')
    plt.show()





#----------------------------------------- ESO Price - Volatility Sensitivity -----------------------------------------
def PlotSensitivity_Volatility():
    SigmaValues = np.linspace(0.05, 1, int(round((1-0.05)/0.05, 0)))
    V_0_Hat_Sigma = np.zeros(len(SigmaValues))

    for i in range(len(SigmaValues)):
        # Sampling GBM paths is the only component that depends on \sigma.
        StockPricePaths_Sigma = SimulateGBM(S_0 = S_0, drift = r, sigma = SigmaValues[i], delta_t = Delta_t,
                                            N = N, m = m)

        CashFlowMatrix_Sigma = BackwardInductionRoutine(S = StockPricePaths_Sigma, K = K, r = r, delta_t = Delta_t,
                                                        index_T_1 = Index_T_1,
                                                        basis_function = BasisFunction_Monomial, n_Terms = 3)
        V_0_Hat_Sigma[i] = ESO_PriceAtGrant(CashFlowMatrix = CashFlowMatrix_Sigma, r = r, time_grid = TimeGrid)["Price"]


    plt.figure(figsize = (10, 6))

    plt.plot(SigmaValues, V_0_Hat_Sigma)
    plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
    plt.title(rf"Estimated ESO Price $-$ Sensitivity to Volatility $\sigma$"
              "\n"
              rf"($T_0$={T_0}, $T_1$={T_1}, $T_2$={T_2}, $N$={N:,}, $m$={m})")
    plt.xlabel(r"Sigma $\sigma$")
    plt.ylabel(r"$\hat{V}_0$")
    plt.xlim([SigmaValues[0], SigmaValues[-1]])
    plt.tight_layout()
    plt.savefig(f"CF_Project_2_SensitivityVolatility.png", dpi = 300, bbox_inches = 'tight')
    plt.show()




#--------------------------------------------- ESO Price - T_1 Sensitivity ---------------------------------------------
def PlotSensitivity_T_1():
    T_1_Values = TimeGrid  # Include T_0 and T_2
    V_0_Hat_T_1 = np.zeros(len(T_1_Values))

    for i in range(len(T_1_Values)):
        # BackwardInductionRoutine() depends on T_1 through index_T_1 = IndexExerciseStart(...)
        Index_T_1_i = IndexExerciseStart(time_grid = TimeGrid, T_1 = T_1_Values[i])

        CashFlowMatrix_T_1 = BackwardInductionRoutine(S = StockPricePaths, K = K, r = r, delta_t = Delta_t,
                                                        index_T_1 = Index_T_1_i,  # This changes
                                                        basis_function = BasisFunction_Monomial, n_Terms = 3)
        V_0_Hat_T_1[i] = ESO_PriceAtGrant(CashFlowMatrix = CashFlowMatrix_T_1, r = r, time_grid = TimeGrid)["Price"]


    plt.figure(figsize = (10, 6))

    plt.plot(T_1_Values, V_0_Hat_T_1)
    plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
    plt.title(rf"Estimated ESO Price $-$ Sensitivity to $T_1$"
              "\n"
              rf"($T_0$={T_0}, $T_2$={T_2}, $\sigma$={Sigma}, $N$={N:,}, $m$={m})")
    plt.xlabel(r"$T_1$")
    plt.ylabel(r"$\hat{V}_0$")
    plt.xlim([T_1_Values[0], T_1_Values[-1]])
    plt.tight_layout()
    plt.savefig(f"CF_Project_2_SensitivityT_1.png", dpi = 300, bbox_inches = 'tight')
    plt.show()


#----------------------------------------- Running the Simulation in This File -----------------------------------------
if __name__ == "__main__":
    # This block only executes when the CF_Project_2.py file is run directly.
    # Every Python file has a built-in variable __name__:
    #       - When the file is run directly, Python sets: __name__ = "__main__"
    #       - When the file is imported by another file, Python sets: __name__ = "NameOtherFile". Hence, this block
    #         is skipped entirely.

    CompareBases()
    PlotGBMPaths()
    ConvergencePlot_CompareBases()
    PlotSensitivity_Volatility()
    PlotSensitivity_T_1()
