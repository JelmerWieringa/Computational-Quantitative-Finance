########################################################################################################################
######################################## COMPUTATIONAL FINANCE - EXERCISE SET 1 ########################################
###################################################### QUESTION 1 ######################################################
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt


############################ Define a Function to Generate Trajectories of Brownian Motion #############################
def BrownianMotionPaths(T_Max, NoOfPaths, NoOfSteps):
    """
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



#################################### Plotting Some Trajectories of Brownian Motion #####################################
NoOfPaths = 21
NoOfSteps = 1000
T = 5

Paths_BM = BrownianMotionPaths(T_Max = T, NoOfPaths = NoOfPaths, NoOfSteps = NoOfSteps)
TimeGrid = Paths_BM["Time"]
W = Paths_BM["W"]


plt.figure(figsize = (10, 6))

plt.plot(TimeGrid, np.transpose(W))

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.title(rf"{NoOfPaths} trajectories of Brownian motion $W$ on $[0, {T}]$")
plt.xlabel(r"Time $t$")
plt.ylabel(r"$W(t)$")
plt.xlim([0, T])
plt.tight_layout()
plt.savefig("CF_HW1_Q1_BMTrajectories.png", dpi = 300, bbox_inches = 'tight')
plt.show()



################################ Approximate the LHS Ordinary Integral by a Riemann Sum ################################
nMax = 150
RiemannSumApproximation = np.zeros(nMax)  # Store approximations for varying n

Paths_BM = BrownianMotionPaths(T_Max = T, NoOfPaths = 1, NoOfSteps = NoOfSteps)
TimeGrid = Paths_BM["Time"]
W = Paths_BM["W"][0, :]  # Single path, shape (NoOfSteps+1,)


for n in range(1, nMax + 1):
    SumIndices = np.linspace(0, NoOfSteps, n + 1, dtype=int)
    W_n = W[SumIndices]

    TimeGrid_n = TimeGrid[SumIndices]  # The partition of [0,T] into n-1 intervals
    TimeIncrement_n = T / n  # Delta t

    RiemannSumApproximation[n - 1] = np.sum(np.exp(-(T - TimeGrid_n[:-1])) * W_n[:-1] * TimeIncrement_n)



################################# Approximate the RHS Stochastic Integral by a Ito Sum #################################
ItoSumApproximation = np.zeros(nMax)

for n in range(1, nMax + 1):
    SumIndices = np.linspace(0, NoOfSteps, n + 1, dtype=int)
    W_n = W[SumIndices]
    TimeGrid_n = TimeGrid[SumIndices]

    Increments_W_n = np.diff(W_n)  # W(t_{i+1}) - W(t_i)

    ItoSumApproximation[n - 1] = np.sum( (1 - np.exp(-1 * (T - TimeGrid_n[:-1]))) * Increments_W_n )



######################################## Plotting the Estimates of the Integral ########################################
plt.figure(figsize = (10, 6))

plt.plot(range(1, nMax + 1), RiemannSumApproximation)
plt.plot(range(1, nMax + 1), ItoSumApproximation)

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.title(rf"Approximation of Integral over the Number of Subintervals $n$")
plt.xlabel(r"n")
plt.ylabel(r"Approximation of Integral")
plt.xlim([1, nMax])
plt.legend(["Riemann Sum", r"It$\hat{o}$ Sum"])
plt.tight_layout()
plt.savefig("CF_HW1_Q1_IntegralApproximation.png", dpi = 300, bbox_inches = 'tight')
plt.show()
plt.show()

print(f"Riemann Sum Approximation at n={nMax}:  {RiemannSumApproximation[-1]}")
print(f"Ito Sum Approximation at n={nMax}:      {ItoSumApproximation[-1]}")