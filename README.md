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
└── Quantitative_Finance/
    └── (to be added)
```

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
