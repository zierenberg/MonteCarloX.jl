# Monte Carlo Fundamentals

Monte Carlo methods estimate expectations by sampling.
Let ``X`` be a random variable with target density or mass function ``\pi(x)``,
and let ``f`` be an observable. The central quantity is
```math
\mathbb{E}_{\pi}[f(X)] = \int f(x)\,\pi(x)\,\mathrm{d}x,
```

with the integral understood as a sum in discrete spaces.
Given samples ``x_1, \dots, x_n`` distributed according to ``\pi``, Monte Carlo uses
```math
\mathbb{E}_{\pi}[f(X)] \approx \frac{1}{n}\sum_{i=1}^n f(x_i).
```

## 1) What is being sampled?

Monte Carlo workflows differ mainly in the object whose distribution you sample.

- **State sampling:** distributions over states ``x``
- **Trajectory sampling:** distributions over paths ``\omega = (x_t)_{t \in [0,T]}``

## 2) State sampling: equilibrium ensembles

For equilibrium systems, the target distribution ``\pi(x)`` arises from a Hamiltonian or energy functional.

### Canonical ensemble

```math
\pi(x) = \frac{1}{Z(\beta)} e^{-\beta E(x)},
```

where ``\beta = 1/(k_B T)`` is the inverse temperature and ``Z(\beta) = \sum_x e^{-\beta E(x)}`` is the partition function.
Monte Carlo sampling produces configurations ``x`` distributed according to this Boltzmann weight.

### Generalized ensembles

For rugged energy landscapes or to access rare events, generalized ensembles modify the sampling measure.

**Multicanonical:** ``\pi_{\text{muca}}(x) \propto e^{-\beta_{\text{ref}} E(x)} / w(E(x))``, where ``w(E)`` is a tuned weight to flatten the energy histogram.

**Wang-Landau:** iteratively updates ``w(E)`` to achieve flat histogram sampling, estimating the density of states ``\Omega(E)``.

### Extended ensembles

**Parallel tempering** (replica exchange) introduces multiple replicas at different temperatures ``\{\beta_i\}`` and allows exchanges between them.
The extended state is ``(\beta_i, x_i)``, and the target distribution factorizes as
```math
\Pi(\{(\beta_i, x_i)\}) = \prod_i \frac{1}{Z(\beta_i)} e^{-\beta_i E(x_i)}.
```
Exchange moves ``(\beta_i, x_i) \leftrightarrow (\beta_j, x_j)`` enable sampling of low-temperature states via high-temperature replicas.

## 3) Trajectory sampling: kinetic processes

For non-equilibrium or time-dependent processes, the target is a distribution over entire trajectories ``\omega = (x_t)_{t\in[0,T]}``.

### Event-driven sampling

Given state-dependent event rates ``\{r_k(x)\}``, the total rate is ``r_{\text{tot}} = \sum_k r_k(x)``.
The waiting time ``\tau`` to the next event is exponentially distributed:
```math
p(\tau) = r_{\text{tot}}(x) e^{-r_{\text{tot}}(x) \tau}.
```
After drawing ``\tau``, an event ``k`` is selected with probability ``r_k(x)/r_{\text{tot}}(x)``.
The process then advances: ``t \to t + \tau`` and ``x \to x'`` (modified according to event ``k``).

This defines a path-space distribution where each trajectory ``\omega`` has probability weight proportional to the product of exponential waiting times and conditional event probabilities.

## 4) Markov chain convergence

For ergodic Markov chains:

- **Detailed balance:** ``\pi(x') T(x' \leftarrow x) = \pi(x) T(x \leftarrow x')`` ensures ``\pi`` is stationary.
- **Ergodiciy:** every state can be reached from every other state (eventually).

Under these conditions, the chain converges to ``\pi`` regardless of the initial state.
The *rate* of convergence depends on the algorithm and the target distribution's geometry (e.g., barrier heights, correlations).

## 5) Sampling vs. integration

Monte Carlo gives *stochastic estimates* of expectations:
```math
\hat{f} = \frac{1}{n}\sum_{i=1}^n f(x_i), \quad x_i \sim \pi.
```

The statistical error scales as ``1/\sqrt{n}``, independent of dimensionality — this is the key advantage over deterministic quadrature.
However, *correlated samples* from a Markov chain increase the asymptotic variance:
```math
\sigma^2_{\text{asymp}} = \sigma^2 \left(1 + 2\sum_{k=1}^\infty \rho_k\right),
```
where ``\rho_k`` is the autocorrelation at lag ``k``.
This is quantified by the *autocorrelation time* ``\tau_{\text{int}}``, so the effective sample size is ``n / \tau_{\text{int}}``.
