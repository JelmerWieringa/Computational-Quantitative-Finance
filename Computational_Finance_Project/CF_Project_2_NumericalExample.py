########################################################################################################################
############################################ COMPUTATIONAL FINANCE - PROJECT ###########################################
######################################### QUESTION 2 - SIMPLE NUMERICAL EXAMPLE ########################################
########################################################################################################################

import numpy as np


############################################## Generate Stock Price Paths ##############################################
########################################################################################################################

#------------------------------------------------ Set Model Parameters -------------------------------------------------
S_0 = 1
K = 1  # K = S(t_0) := S_0
r = 0.06  # Equivalent to LSM paper
sigma = 0.2
delta_t = 1

NoOfPaths = 5   # N parameter
NoOfSteps = 3   # m parameter

#------------------------------------ Generate the same realizations of standard BM ------------------------------------
np.random.seed(23)
Z = np.random.standard_normal((NoOfPaths, NoOfSteps))

#--------------------------------------------- Compute the exact S-values ----------------------------------------------
S = np.ones((NoOfPaths, NoOfSteps + 1))
for i in range(NoOfSteps):
    # Exact solution of GBM:
    S[:, i+1] = S[:, i] * np.exp((r - 0.5 * sigma**2) * delta_t + sigma * np.sqrt(delta_t) * Z[:, i])

#--------------------------------------- Print the Matrix with stock price paths ---------------------------------------
StockPricePaths = np.round(S, 2)
print(StockPricePaths)




######################################## Optimal Choice at Maturity t = T_2 = 3 ########################################
########################################################################################################################
def H_PayoffFunction(S, K):
    """
    GOAL: Define the payoff function H(S) := max{S - K, 0}.
    :param S: Stock price.
    :param K: Strike price.
    """
    return np.maximum(S - K, 0)

CashFlowMatrix_t_3 = H_PayoffFunction(StockPricePaths[:, 3], K = K)
# Similar to European option, i.e., only evaluation of payoff function at maturity




######################################## Optimal Choice at time t = T_{3/2} = 2 ########################################
########################################################################################################################
def ITM_Paths(StockPrices, K):
    """
    GOAL: Select stocks that are ITM (in-the-money) at a single time step t_i & return a boolean array.
    :param StockPrices:  Array of stock prices across N paths at time t_i.
    :param K: Strike price.
    """
    ITM_Indices = H_PayoffFunction(S = StockPrices, K = K) > 0  # Select paths that are ITM, i.e., H > 0
    return {"Values": StockPrices[ITM_Indices], "Indices": ITM_Indices}


#------------------------------------ Construct the regression data matrix for t=2 -------------------------------------
DiscountFactor = np.exp(-r)  # For a single period, i.e., \delta t = 1

ITM_Indices_t_2 = ITM_Paths(StockPrices = StockPricePaths[:, 2], K = K)["Indices"]
RegressionData_t_2 = np.zeros((sum(ITM_Indices_t_2), 2))  # Matrix containing observations for linear regression
# Note: We only consider ITM paths

RegressionData_t_2[:, 0] = DiscountFactor * CashFlowMatrix_t_3[ITM_Indices_t_2]  # Y_{t=2} column
RegressionData_t_2[:, 1] = ITM_Paths(StockPrices = StockPricePaths[:, 2], K = K)["Values"]  # S_{t=2} column
# Note: Equivalently, "= StockPricePaths[:, 2][ITM_Indices_t_2]"


#----------------------------------- Linear regression with 1,S,S^2 basis functions ------------------------------------
Y_t_2 = RegressionData_t_2[:, 0]  # Discounted continuation cash flows
S_t_2 = RegressionData_t_2[:, 1]  # Stock prices of ITM paths

Psi_t_2 = np.column_stack([  # Design matrix with basis functions {1, S, S^2}, dim=N_TIM✕3
    np.ones(len(S_t_2)),   # psi_1(S) = 1
    S_t_2,                 # psi_2(S) = S
    S_t_2**2               # psi_3(S) = S^2
])

beta_Hat_t_2, _, _, _ = np.linalg.lstsq(Psi_t_2, Y_t_2, rcond=None)  # OLS estimator
# np.linalg.lstsq() returns the least-squares solution to a linear matrix equation. I.e., it finds/approximates the
# \hat{beta} that minimizes || Psi @ beta - Y ||^2.
# Note: We do not use the analytical solution of the OLS estimator, because np.linalg.inv() is numerically unstable.
#       Further, we prefer np.linalg.lstsq() over np.linalg.solve() because it is numerically more stable.

c_Hat_t_2 = Psi_t_2 @ beta_Hat_t_2  # Estimated continuation values \hat{c} for each ITM path

print("Regression coefficients:", beta_Hat_t_2)
print(r"Estimated continuation values (c_Hat)", c_Hat_t_2)


#--------------------------------------- Optimal early exercise decision at t=2 ----------------------------------------
PayoffValues_ITMPaths_t_2 = H_PayoffFunction(StockPricePaths[:, 2], K = K)[ITM_Indices_t_2]
# Payoff values of stocks that are ITM at t=2

OptimalEarlyExerciseDecision_t_2 = np.column_stack([PayoffValues_ITMPaths_t_2, c_Hat_t_2])


#----------------------------------------------- Cash flow matrix at t=2 -----------------------------------------------
ITMPaths_OptimalToExercise_t_2 = OptimalEarlyExerciseDecision_t_2[:, 0] > OptimalEarlyExerciseDecision_t_2[:, 1]
# Note: These are only the ITM paths.
OTMPaths_Indices_t_2 = np.where(ITM_Indices_t_2 == False)[0]
# The indices of elements that were OTM at t=2, and hence not included in ITMPaths_OptimalToExercise_t_2.
Indices_OptimalToExercise_t_2 = np.insert(arr = ITMPaths_OptimalToExercise_t_2, obj = OTMPaths_Indices_t_2,
                                          values = np.zeros(np.size(OTMPaths_Indices_t_2)), axis=0)
# This includes all the N paths. OTM paths are set to False.

CashFlowMatrix_t_2 = np.zeros((NoOfPaths, 2))  # Cash flow matrix at t=2
CashFlowMatrix_t_2[:, 0] = H_PayoffFunction(S = StockPricePaths[:, 2], K = K) * Indices_OptimalToExercise_t_2
# I.e., exercise at t=2, when payoff from exercising at t=2 is > the continuation value at t=2.
CashFlowMatrix_t_2[:, 1] = H_PayoffFunction(StockPricePaths[:, 3], K = K) * [Indices_OptimalToExercise_t_2 == False]
# Only keeping the payoff values at maturity t=3, if it is not optimal to exercise at t=2.




######################################## Optimal Choice at Maturity t = T_1 = 1 ########################################
########################################################################################################################

#------------------------------------ Construct the regression data matrix for t=1 -------------------------------------
ITM_Indices_t_1 = ITM_Paths(StockPrices = StockPricePaths[:, 1], K = K)["Indices"]  # Paths that are ITM at t=1
RegressionData_t_1 = np.zeros((sum(ITM_Indices_t_1), 2))  # Matrix containing observations for linear regression
# Note: We only consider ITM paths.

# Total discounted future cash flow, from whichever date it pays
Y_future = (DiscountFactor * CashFlowMatrix_t_2[:, 0]       # Cash flow at t=2, discounted 1 step
          + DiscountFactor**2 * CashFlowMatrix_t_2[:, 1])   # Cash flow at t=3, discounted 2 steps

RegressionData_t_1[:, 0] = Y_future[ITM_Indices_t_1]  # Y_{t=1} column
RegressionData_t_1[:, 1] = ITM_Paths(StockPrices = StockPricePaths[:, 1], K = K)["Values"]  # S_{t=1} column
# Note: In the regression step, we exclude OTM paths.


#----------------------------------- Linear regression with 1,S,S^2 basis functions ------------------------------------
Y_t_1 = RegressionData_t_1[:, 0]  # Discounted continuation cash flows
S_t_1 = RegressionData_t_1[:, 1]  # Stock prices of ITM paths

Psi_t_1 = np.column_stack([  # Design matrix with basis functions {1, S, S^2}, dim=N_TIM✕3
    np.ones(len(S_t_1)),   # psi_1(S) = 1
    S_t_1,                 # psi_2(S) = S
    S_t_1**2               # psi_3(S) = S^2
])

beta_Hat_t_1, _, _, _ = np.linalg.lstsq(Psi_t_1, Y_t_1, rcond=None)  # OLS estimator
c_Hat_t_1 = Psi_t_1 @ beta_Hat_t_1  # Estimated continuation values \hat{c} for each ITM path

print("Regression coefficients:", beta_Hat_t_1)
print(r"Estimated continuation values (c_Hat)", c_Hat_t_1)


#--------------------------------------- Optimal early exercise decision at t=1 ----------------------------------------
PayoffValues_ITMPaths_t_1 = H_PayoffFunction(StockPricePaths[:, 1], K = K)[ITM_Indices_t_1]
# Payoff values of stocks that are ITM at t=1.

OptimalEarlyExerciseDecision_t_1 = np.column_stack([PayoffValues_ITMPaths_t_1, c_Hat_t_1])


#----------------------------------------------- Cash flow matrix at t=1 -----------------------------------------------
ITMPaths_OptimalToExercise_t_1 = OptimalEarlyExerciseDecision_t_1[:, 0] > OptimalEarlyExerciseDecision_t_1[:, 1]
# Note: These are only the ITM paths.
OTMPaths_Indices_t_1 = np.where(ITM_Indices_t_1 == False)[0]
# The indices of elements that were OTM at t=1, and hence not included in ITMPaths_OptimalToExercise_t_1.
Indices_OptimalToExercise_t_1 = np.insert(arr = ITMPaths_OptimalToExercise_t_1, obj = OTMPaths_Indices_t_1,
                                          values = np.zeros(np.size(OTMPaths_Indices_t_1)), axis=0)
# This includes all the N paths. OTM paths are set to False.

CashFlowMatrix_t_1 = np.zeros((NoOfPaths, 3))  # Fill it with zeros first, then only change the entries that should be non-zero
CashFlowMatrix_t_1[:, 0] = H_PayoffFunction(S = StockPricePaths[:, 1], K = K) * Indices_OptimalToExercise_t_1
# I.e., exercise at t=1, when payoff from exercising at t=1 is > the continuation value at t=1.
CashFlowMatrix_t_1[:, 1] = CashFlowMatrix_t_2[:, 0] * [Indices_OptimalToExercise_t_1 == False]
# Only keeping the cash flow values at t=2 and t=3, if it is not optimal to exercise at t=1
CashFlowMatrix_t_1[:, 2] = CashFlowMatrix_t_2[:, 1]  # Cash flow at t=3
# However, when exercising at t=1 is optimal, we need to ensure that all the other entries to the right in that row are set to zero.
CashFlowMatrix_t_1[:, 2] = CashFlowMatrix_t_1[:, 2] * (CashFlowMatrix_t_1[:, 0] == 0)
print(CashFlowMatrix_t_1)

StoppingRuleMatrix = (CashFlowMatrix_t_1 > 0) * 1  # 1 = exercise that path at that time.




################################################ ESO Price at Grant t=0 ################################################
########################################################################################################################
TimeGrid = np.array([1, 2, 3])  # t_1, t_2, t_3
DiscountFactors = np.exp(-r * TimeGrid)   # [e^{-r*1}, e^{-r*2}, e^{-r*3}]
V_0 = np.mean(CashFlowMatrix_t_1 @ DiscountFactors)

print(f"ESO price at t=0: {V_0:.4f}")