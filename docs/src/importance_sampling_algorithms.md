# Importance Sampling Algorithms

Importance sampling in MonteCarloX is built around **local proposals + acceptance rules**.
The defining feature is a **discrete-step** update loop (proposal → decision → update).

While this is most commonly used for equilibrium sampling, the same machinery can be used in non-equilibrium protocols by changing parameters or target weights over steps.

## Mental model

Each step is:

1. propose a local change
2. compute a local log-ratio (or local energy difference)
3. accept/reject via the algorithm
4. update counters and measure if needed

The core API function is `accept!`.

## Target distribution and acceptance rule

Let \(\pi(x)\) be the target density (or mass function) on the state space.

- Bayesian example:

\[
\pi(\theta)=p(\theta\mid\mathcal D).
\]

- Statistical-mechanics microstate example:

\[
\pi(x)=p(x\mid\beta)=\frac{e^{-\beta E(x)}}{Z(\beta)}.
\]

Metropolis-Hastings acceptance is

\[
\alpha(x\to x')=\min\!\left(1,\frac{\pi(x')\,q(x\mid x')}{\pi(x)\,q(x'\mid x)}\right).
\]

For symmetric local proposals \(q\), this reduces to \(\pi(x')/\pi(x)\).

### Unified view: Bayesian and statistical-physics targets

The same sampler is used in both domains by changing only the callable
log target score:

- Bayesian inference: `logweight(theta) = logposterior(theta) = loglikelihood(theta) + logprior(theta)`
- Statistical mechanics: `logweight(x) = -beta * E(x)`

In MonteCarloX this callable is carried by the algorithm as the `ensemble`
object, and can also be accessed as `logweight(alg)`.

Both views are important:
- `ensemble(alg)` is the architecture-level object
- `logweight(alg)` is the acceptance-rule interpretation

They refer to the same callable value.

## Metropolis

### When to use it

- default first choice for equilibrium sampling
- simple and robust

### Acceptance intuition

- always accept moves toward larger target weight
- accept less favorable moves with probability `exp(log_ratio)`

### Minimal usage

```julia
using Random
using MonteCarloX

rng = MersenneTwister(1)
logdensity(x) = -0.5 * x^2
alg = Metropolis(rng, logdensity)

x = 0.0
for _ in 1:20_000
    x_new = x + randn(alg.rng)
    x = accept!(alg, x_new, x) ? x_new : x
end

println(acceptance_rate(alg))
```

### Bayesian example (primary)

Coin-flip posterior with local random-walk proposals on theta:

- \(\pi(\theta)=p(\theta\mid\text{data})\propto p(\text{data}\mid\theta)p(\theta)\)
- implementation target: `logposterior(theta)`

Repository example: `examples/bayesian_coin_flip.ipynb`

### Statistical-mechanics example (secondary)

Ising microstate sampling:

- state \(x\) = spin configuration
- target \(\pi(x)=e^{-\beta E(x)}/Z(\beta)\)
- local proposal: spin flip

Repository example: `examples/spin_systems/metropolis_ising2D.ipynb`

Energy-variable view (same physics):

\[
p(E\mid\beta)=\frac{\Omega(E)e^{-\beta E}}{Z(\beta)},
\]

where \(\Omega(E)\) is the density of states.

## Glauber

Same proposal style as Metropolis, but uses logistic acceptance.
Useful when that acceptance rule is the natural one for your dynamics/modeling convention.

## HeatBath

Draws from local conditional probabilities instead of accept/reject.
For Ising-like models this often means directly sampling local spin values from conditional weights.

## Generalized ensembles

These methods adapt or use non-canonical weights to improve exploration.

### Multicanonical

- keeps a histogram of visited bins
- updates tabulated log-weights from histogram information
- useful for broad energy exploration / barrier crossing

```julia
using Random
using MonteCarloX

lw = BinnedObject(-20:2:20, 0.0)
alg = Multicanonical(MersenneTwister(2), lw)

set!(ensemble(alg), -10:2:10, x -> 0.0)
# run your update loop with accept!(alg, x_new, x_old)
# then call update!(ensemble(alg))
```

### Wang-Landau

- updates log-density-of-states estimate at visited bins
- progressively refines modification factor (`logf` via `update!(ensemble(alg))` between stages)

```julia
using Random
using MonteCarloX

lw = BinnedObject(-20:2:20, 0.0)
alg = WangLandau(MersenneTwister(3), lw; logf=1.0)

# in your loop: accept!(alg, x_new, x_old)
# between stages: update!(alg)
```

## Choosing quickly

- Start with `Metropolis` for standard equilibrium sampling.
- Use `HeatBath` when conditional local probabilities are natural and cheap.
- Use `Multicanonical`/`WangLandau` when canonical sampling gets stuck or explores too narrowly.

## Example map (algorithm ↔ application)

- **Bayesian scalar posterior** (`Metropolis`): `examples/bayesian_coin_flip.ipynb`
- **Bayesian regression posterior** (`Metropolis`): `examples/house_price_prediction.ipynb`
- **Canonical spin sampling** (`Metropolis`): `examples/spin_systems/metropolis_ising2D.ipynb`
- **Generalized-ensemble exploration** (`Multicanonical`, `WangLandau`):
    - `examples/spin_systems/muca_ising2D.ipynb`
    - `examples/muca_LDT_gaussian_rngs.ipynb`

## API reference

```@docs
AbstractImportanceSampling
AbstractMetropolis
AbstractHeatBath
ImportanceSampling
ensemble(alg::AbstractImportanceSampling)
logweight(alg::AbstractImportanceSampling)
logweight(ens::AbstractEnsemble)
logweight(ens::AbstractEnsemble, x)
Metropolis
Glauber
HeatBath
accept!
acceptance_rate
reset!(alg::AbstractImportanceSampling)
Multicanonical
WangLandau
update!(ens::AbstractEnsemble, args...)
update!(e::WangLandauEnsemble; power::Real=0.5)
```