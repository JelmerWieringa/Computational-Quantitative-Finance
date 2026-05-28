########################################################################################################################
##################################### COMPUTATIONAL FINANCE - EXERCISE SET 2 ###########################################
################################################## EXERCISE 2b #########################################################
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt


################################# Define the Characteristic Function of S(T) ###########################################
def CF_S(u, S_0, gamma, t_0, T):
    r"""
    GOAL = Compute the characteristic function of S(T) derived in subquestion 2a:
            \phi_{S(T)}(u) = exp( i*u*S_0 / (1 - 0.5*i*gamma^2*tau*u) )
    """
    tau = T - t_0
    i = complex(0, 1)  # Imaginary number
    return np.exp( -u * S_0 / (0.5 * gamma**2 * tau * u + i) )



#################################### Define Function to Recover PDF with COS Method ####################################
def COS_PDF_Recovery(CF, x, N, a, b):
    r"""
    GOAL = Recover the PDF by using the COS method.

    The Fourier-cosine expansion approximation of the PDF on [a, b] is given by:
            f_X(x) = \sum_{k=0}^{N-1}' F_k * cos( k*pi*(x - a) / (b - a) ),
    with F_k = (2/(b-a)) * Re[ CF(k*pi/(b-a)) * exp(-i*k*pi*a/(b-a)) ].
    """
    i = complex(0, 1)  # Imaginary number

    k = np.linspace(0, N - 1, N)  # k indices
    u = k * np.pi / (b - a)  # Defined on page 28 of lecture 8

    # Next, compute the F_k coefficients
    F_k = 2.0 / (b - a) * np.real(CF(u) * np.exp(-i * u * a))
    F_k[0] = F_k[0] * 0.5  # Scale the first term (i.e., the prime in the sum notation (capital sigma))

    f_X_Approximation = np.matmul(F_k, np.cos(np.outer(u, x - a)))  # Matrix multiplication: F_k times cos(...)
    return f_X_Approximation



######################### Define Function for Monte Carlo Simulation with Euler Discretization #########################
def MonteCarloEuler(S_0, gamma, T, NoOfPaths, NoOfSteps):
    r"""
    GOAL = Simulate S(T) using the Euler discretization of dS(t) = gamma*sqrt(S(t))*dW(t):
            S_{i+1} = S_i + gamma * sqrt(max{S_i, 0}) * (W(t_{i+1}) - W(t_i))
    """
    Delta_t = T / float(NoOfSteps)  # Equidistant time steps

    np.random.seed(23)
    Z = np.random.normal(0, 1, [NoOfPaths, NoOfSteps])  # Matrix of standard normal realizations

    S = np.zeros([NoOfPaths, NoOfSteps + 1])  # Empty matrix to store all MC paths (row-wise)
    S[:, 0] = S_0

    for i in range(NoOfSteps):
        Sqrt_S_i = np.sqrt(np.maximum(S[:, i], 0))  # Ensure existence of square root, i.e., no imaginary numbers
        S[:, i + 1] = S[:, i] + gamma * Sqrt_S_i * np.sqrt(Delta_t) * Z[:, i]

    return S[:, -1]  # Return S(T)



############################################# Setting Model Parameters #################################################
gamma = 0.25
S_0   = 1  # Initial stock price
T     = 5  # Maturity time
t_0   = 0  # Initial time



############################### Plotting the COS Recovered Density versus the MC Density ###############################
### COS method settings
CF = lambda u: CF_S(u, S_0, gamma, t_0, T)

a = 0  # Lower bound of truncated domain. S(t_0) > 0.
b = 4  # Upper bound of truncated domain
N_values = [2**n for n in range(2, 9)]  # Values for N
s_grid = np.linspace(0.001, b, 1000)  # Points at which we evaluate the PDF f_{S(T)}


### Monte Carlo settings
NoOfPaths = 100000
NoOfSteps = 5000
S_T_MC = MonteCarloEuler(S_0, gamma, T, NoOfPaths, NoOfSteps)


plt.figure(figsize=(10, 6))

# Plot the MC PDF
plt.hist(S_T_MC, bins='fd', edgecolor='grey', density=True, alpha=0.4, color='gray', label='MC Euler histogram')

for N in N_values:
    f_COS = COS_PDF_Recovery(CF = CF, x = s_grid, N = N, a = a, b = b)  # COS recovered PDF for specific N
    plt.plot(s_grid, f_COS, '--', label=rf'COS, $N = {N}$')

plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.title(rf"COS Recovered PDF of $S(T)$ versus Monte Carlo PDF, NoOfPaths = {NoOfPaths}, NoOfSteps = {NoOfSteps}")
plt.xlabel(r'$s$')
plt.ylabel(r'$f_{S(T)}(s)$')
plt.xlim([a, b])
plt.legend()
plt.tight_layout()
plt.savefig(f"CF_HW2_Q2b_COSvsAnalytical_NoOfPaths={NoOfPaths}_NoOfSteps={NoOfSteps}.png", dpi=300, bbox_inches='tight')
plt.show()