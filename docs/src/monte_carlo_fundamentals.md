# Monte Carlo Fundamentals

Monte Carlo methods estimate expectations by sampling.

Let \(X\) be a random variable with target density or mass function \(\pi(x)\), and let \(f\) be an observable.
The central quantity is

\[
\mathbb{E}_{\pi}[f(X)] = \int f(x)\,\pi(x)\,\mathrm{d}x,
\]

with the integral understood as a sum in discrete spaces.
Given samples \(x_1,\dots,x_n\) distributed according to \(\pi\), Monte Carlo uses

\[
\mathbb{E}_{\pi}[f(X)] \approx \frac{1}{n}\sum_{i=1}^n f(x_i).
\]

## 1) What is being sampled?

Monte Carlo workflows differ mainly in the object whose distribution you sample.

- **State sampling:** distributions over states \(x\)
- **Trajectory sampling:** distributions over paths \(\omega = (x_t)_{t\in[0,T]}\)

Examples:

- Bayesian inference:

\[
\pi(\theta) = p(\theta\mid \mathcal{D}) \propto p(\mathcal{D}\mid\theta)\,p(\theta).
\]

- Statistical mechanics (microstates):

\[
\pi(x) = p(x\mid\beta) = \frac{e^{-\beta E(x)}}{Z(\beta)}.
\]

- Statistical mechanics (energy variable):

\[
p(E\mid\beta) = \frac{\Omega(E)e^{-\beta E}}{Z(\beta)},
\]

where \(\Omega(E)\) is the density of states.

Examples of trajectory sampling:

- chemical reaction networks
- epidemic/contact processes
- kinetic spin dynamics

## 2) Unified view: trajectories are states in path space

Conceptually, trajectory sampling is still state-distribution sampling, but in a higher-dimensional object space (path space).

Why keep a separate class in practice?

- trajectory methods preserve explicit physical/simulation time
- event times and event identities are sampled jointly
- APIs are event-driven rather than proposal/reject loops

So the mathematics is unified, while computational interfaces are specialized.

Path-space notation:

- \(\omega\): entire trajectory
- \(\pi(\omega)=p(\omega\mid\lambda)\): path distribution under model parameters \(\lambda\)

## 3) Two algorithmic interfaces

## A. Discrete-step samplers (importance sampling / MCMC)

Loop:

1. propose a local change
2. evaluate a log-ratio or local weight difference
3. accept/reject

Used in Bayesian inference, equilibrium statistical mechanics, and many optimization/sampling hybrids.

## B. Continuous-time event samplers (kinetic Monte Carlo)

Loop:

1. compute event rates
2. sample waiting time dt
3. sample event index
4. advance time and apply event

Used when timing of events is part of the model output.

## 4) Mapping to MonteCarloX

Across both interfaces, MonteCarloX keeps the same decomposition:

1. **System**: state and model-specific updates
2. **Weight/rates**: target density or event intensities
3. **Algorithm**: transition sampler

`Measurements` are optional and independent of the core transition logic.

## 5) Practical simulation recipe

For most workflows, implementation follows one of two loops.

### A. Discrete-step sketch

```julia
using Random
using MonteCarloX

rng = MersenneTwister(1)
logdensity(x) = -0.5 * x^2
alg = Metropolis(rng, logdensity)

x = 0.0
for _ in 1:10_000
	x_new = x + randn(alg.rng)
	x = accept!(alg, x_new, x) ? x_new : x
end
```

### B. Continuous-time sketch

```julia
using Random
using MonteCarloX

alg = Gillespie(MersenneTwister(2))
rates = [0.2, 0.8]

for _ in 1:10_000
	t, event = step!(alg, rates)
	# update your state using `event`
end
```

## 6) Where to continue

- For discrete-step methods and concrete examples:
	- [Importance Sampling Algorithms](importance_sampling_algorithms.md)
- For event-driven continuous-time methods and examples:
	- [Continuous-Time Sampling Algorithms](continuous_time_sampling_algorithms.md)
- Then implement your own model in [Build Your Own System](build_your_own_system.md).
