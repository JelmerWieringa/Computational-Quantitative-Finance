########################################################################################################################
######################################## COMPUTATIONAL FINANCE - EXERCISE SET 1 ########################################
###################################################### QUESTION 4 ######################################################
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt


########################## Define a Function to Generate Trajectories of Brownian Motion W(t) ##########################
def BrownianMotionPaths(T_Max, NoOfPaths, NoOfSteps):
    """
    GOAL = Generate paths/trajectories of Brownian Motion W(t) over [0,T_Max].

    T_Max: Maximum of time interval [0,T], i.e., T
    NoOfPaths: The number of trajectories/paths
    NoOfSteps: The number of time steps, i.e., [0,T] gets partitioned into NoOfSteps-1 subintervals
    """

    Time = np.zeros([NoOfSteps + 1])  # Assume the discrete time grid are non-negative integers
    # Note: "+ 1" is because W(0) needs to be included
    Delta_t = T_Max / float(NoOfSteps)  # Equidistant partition

    np.random.seed(21)  # Fix the randomness
    Z = np.random.normal(0, 1, [NoOfPaths, NoOfSteps])  # Z_i \sim N(0,1)

    W = np.zeros([NoOfPaths, NoOfSteps + 1])  # Array to store Brownian Motion W(t), 0<=t<=T_Max.
    # Note: W(0) is automatically set to zero

    for i in range(0, NoOfSteps):
        # Making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])

        W[:, i + 1] = W[:, i] + np.sqrt(Delta_t) * Z[:, i]
        Time[i + 1] = Time[i] + Delta_t

    Paths = {"Time": Time, "W": W}
    return Paths



################################## Define a Function to Generate Trajectories of X(t) ##################################
def X_Paths(T_Max, NoOfPaths, NoOfSteps, W):
    """
    GOAL = Generate paths/trajectories of X(t) over [0,T_Max].

    T_Max: Maximum of time interval [0,T], i.e., T
    NoOfPaths: The number of trajectories/paths
    NoOfSteps: The number of time steps, i.e., [0,T] gets partitioned into NoOfSteps-1 subintervals
    """

    Time = np.zeros([NoOfSteps + 1])  # Assume the discrete time grid are non-negative integers
    # Note: "+ 1" is because W(0) needs to be included
    Delta_t = T_Max / float(NoOfSteps)  # Equidistant partition

    X = np.zeros([NoOfPaths, NoOfSteps + 1])  # Array to store X(t) values, 0<=t<=T_Max.
    # Note: At t=0, X(t)=0.

    for i in range(0, NoOfSteps):

        Time[i + 1] = Time[i] + Delta_t  # t_1, ..., t_{NoOfSteps}
        X[:, i + 1] = W[:, i + 1] - (Time[i + 1] / T_Max) * W[:, NoOfSteps - (i + 1)]

    Paths = {"Time": Time, "X": X}
    return Paths



#################################### Plotting Some Trajectories of Brownian Motion #####################################
NoOfPaths = 21
NoOfSteps = 1000
T = 10

# Paths of Brownian Motion W(t)
Paths_BM = BrownianMotionPaths(T_Max = T, NoOfPaths = NoOfPaths, NoOfSteps = NoOfSteps)
TimeGrid_W = Paths_BM["Time"]
W = Paths_BM["W"]

# Paths of Process X(t)
Paths_X = X_Paths(T_Max = T, NoOfPaths = NoOfPaths, NoOfSteps = NoOfSteps, W = W)
TimeGrid_X = Paths_X["Time"]
X = Paths_X["X"]


plt.figure(figsize = (10, 6))

plt.plot(TimeGrid_X, np.transpose(X))

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.title(rf"{NoOfPaths} trajectories of $X(t)$ on $[0, {T}]$")
plt.xlabel(r"Time $t$")
plt.ylabel(r"$X(t)$")
plt.xlim([0, T])
plt.tight_layout()
plt.savefig(f"CF_HW1_Q4_BMTrajectories_{NoOfPaths}_Paths.png", dpi = 300, bbox_inches = 'tight')
plt.show()



################################## Plotting the Sample & Theoretical Variance of X(t) ##################################
# Computing the variance of X(t), for each t in the boundary points of the used partition of [0,T]
Var_X = np.var(X, axis=0)  # axis=0 ensures that the variance is computed per column

def TheoreticalVariance_X(t, T_Max):  # Derived in analytically Q4
    """
    GOAL = Computes the theoretical variance of X(t) over [0,T].
    """

    if 0 <= t <= (T_Max / 2):
        return t - (t**2) / T_Max - (t**3) / (T_Max**2)
    elif (T_Max / 2) < t <= T_Max:
        return -t + (3 * t**2) / T_Max - (t**3) / (T_Max**2)
    else:
        print("Please, ensure that t is inside the interval [0,T].")
        return None


plt.figure(figsize = (10, 6))

plt.plot(TimeGrid_X, Var_X, label = "Sample Variance")  # Sample variance

Var_X_Theoretical = np.zeros(len(TimeGrid_X))
for i in range(len(TimeGrid_X)):
    Var_X_Theoretical[i] = TheoreticalVariance_X(t = TimeGrid_X[i], T_Max = T)

plt.plot(TimeGrid_X, Var_X_Theoretical, label = "Theoretical Variance")  # Theoretical variance

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.title(rf"Sample & Theoretical Variance of X(t) on $[0, {T}]$, NoOfPaths$= {NoOfPaths}$")
plt.xlabel(r"Time $t$")
plt.ylabel(r"(Sample) Variance")
plt.xlim([0, T])
plt.legend()
plt.tight_layout()
plt.savefig(f"CF_HW1_Q4_VariancePlot_{NoOfPaths}_Paths.png", dpi = 300, bbox_inches = 'tight')
plt.show()



############################## Plotting the Absolute Error of the Sample Variance of X(t) ##############################
plt.figure(figsize = (10, 6))

plt.plot(TimeGrid_X, np.abs(Var_X_Theoretical - Var_X))  # Absolute error of (sample) variance

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.title(rf"Absolute Error of Sample Variance of X(t) on $[0, {T}]$, NoOfPaths$= {NoOfPaths}$")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Absolute Error Sample Variance")
plt.xlim([0, T])
plt.tight_layout()
plt.savefig(f"CF_HW1_Q4_AbsoluteErrorVariance_{NoOfPaths}_Paths.png", dpi = 300, bbox_inches = 'tight')
plt.show()