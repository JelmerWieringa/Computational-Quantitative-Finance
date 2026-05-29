# Computational & Quantitative Finance

Course materials, exercises, and projects for two master-level finance courses at the University of Groningen and Mastermath:

- **Computational Finance** — Mastermath (instructors: L.A. Grzelak & K.W. Oosterlee, based on *Mathematical Modeling and Computation in Finance*, Oosterlee & Grzelak, World Scientific, 2019)
- **Quantitative Finance for MSc EORAS** — MSc EORAS, University of Groningen


Topics span stochastic calculus, Black-Scholes, characteristic functions, Fourier/COS methods, Monte Carlo simulation, the Heston model, and derivative pricing of vanilla and exotic instruments.

---

## Repository Structure

```
Computational-Quantitative-Finance/
│
├── Computational_Finance_Exercise_Set_1/
│   ├── CF_HW1_Q1.py                # Itô integration & Brownian motion
│   ├── CF_HW1_Q3.py                # MC option pricing: effect of sample standardization
│   ├── CF_HW1_Q4.py                # Variance of X(t) = W(t) − (t/T)·W(T−t)
│   └── Computational_Finance_Exercise_Set_1.pdf
│
├── Computational_Finance_Exercise_Set_2/
│   ├── CF_HW2_Q1.py                # Implied volatility of put option
│   ├── CF_HW2_Q2.py                # Characteristic functions & COS density
│   ├── CF_HW2_Q3.py                # Two-asset correlated GBM (Monte Carlo)
│   └── Computational_Finance_Exercise_Set_2.pdf
│
├── Computational_Finance_Project/
│   ├── CF_Project_2_NumericalExample.py
│   ├── CF_Project_2.py             # LSM backward induction routine 
│   ├── CF_Project_3.py             # Stochastic strike K = S(T_{1/2}) 
│   ├── CF_Project_4.py             # ESO pricing with dividends 
│   └── Computational_Finance_Project_Jelmer_Wieringa.pdf
│
├── Quantitative_Finance/
│   └── (to be added)
│
└── Animations
    └── CF_Animation_MC.py
    └── CF_Animation_LSM.py
```

---

## Animations


**Monte Carlo convergence** — starting from $N = 25$ paths and incrementally growing to $N = 700$ in steps of 25, at each step the left panel adds GBM paths coloured by terminal moneyness (green: ITM, red: OTM), and the right panel shows convergence of the running MC estimate $\hat{V}(S_0, t_0)$ toward the exact Black-Scholes price (dashed), with the 95% confidence interval narrowing as $N$ grows.

<img width="2100" height="750" alt="CF_Animation_MC_Convergence_nStart=25_nEnd=700_nStep=25" src="https://github.com/user-attachments/assets/705b1aa1-a6ca-42a7-ba2d-1c04b21af6ec" />


<p align="center">

**LSM backward induction** — starting from $t_{m-1}$ just before maturity $t_m = T_2 = 10$ and sweeping backward to the vesting boundary $T_1 = 3$, each frame shows one exercise date $t_i$ of the backward induction routine. The left panel displays the scatter of all $N = 2000$ paths split into OTM (red), ITM-continue (magenta), and ITM-exercise (green), overlaid with the immediate payoff $H(S) = \max \\{ S - K,\, 0 \\}$ (solid) and the OLS-fitted continuation value $\hat{c}(t_i, S) = \hat{\beta}_1 \cdot 1+ \hat{\beta}_2 \cdot S + \hat{\beta}_3 \cdot S^2$ (dashed) estimated on ITM paths. The dotted vertical line marks the estimated exercise boundary $S^*(t_i)$, the minimum stock price at which $H(S) \geq \hat{c}(S)$. The right panel accumulates these boundary estimates from right to left, revealing the full exercise boundary surface as the algorithm unwinds.
 
<img width="2100" height="750" alt="CF_Animation_LSM_BackwardInduction_N=2000_m=50" src="https://github.com/user-attachments/assets/89c6f07f-8ff5-46c7-a042-b8ba59d6dc82" />




---

## Courses

### Computational Finance (Mastermath)
A master-level course on numerical methods for derivative pricing. Core topics: stochastic processes and Itô calculus, the Black-Scholes framework, implied volatility, affine diffusion models, characteristic functions, Fourier/COS pricing, Monte Carlo simulation (Euler, Milstein, Heston), hedging and Greeks, and exotic derivatives (barriers, Asians, forward-start options).

**Final project:** Pricing Employee Stock Options (ESOs) via regression Monte Carlo (Longstaff-Schwartz), including vesting periods, stochastic strikes, and dividends.

### Quantitative Finance (MSc EORAS)
*(Materials to be added.)*

---

## References

- Oosterlee, C.W. & Grzelak, L.A. (2019). *Mathematical Modeling and Computation in Finance*. World Scientific.
- Longstaff, F.A. & Schwartz, E.S. (2001). Valuing American options by simulation: A simple least-squares approach. *Review of Financial Studies*, 14(1), 113–147.
- Computational Finance course website: https://compfinance.ddns.net/wordpress/computational-finance/
