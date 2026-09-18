# External/DynamicalSystems

Examples connecting MonteCarloX to [DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystems.jl/stable/),
following [issue #77](https://github.com/zierenberg/MonteCarloX.jl/issues/77). The issue's literal
proposal — bias sampling of initial conditions toward a specific `Attractors.jl` attractor `id` — is
discussed and split into two examples that each match a different part of MCX to a different order
parameter:

- `forgetting_time.jl`: the initial condition `u0` of a chaotic map is the MCX state, and the
  **time to forget** `u0` (how long a tangent vector takes to grow past a threshold, computed via
  `DynamicalSystems.jl`'s `TangentDynamicalSystem`) is a well-behaved discrete order parameter — a
  direct fit for MCX's `WangLandauAlgorithm`. This reproduces the approach of Kitajima & Iba,
  *Multicanonical Sampling of Rare Trajectories in Chaotic Dynamical Systems*
  ([arXiv:1003.2013](https://arxiv.org/abs/1003.2013)), and efficiently harvests the otherwise
  exponentially rare near-regular initial conditions hiding inside a chaotic sea.
- `riddled_basins.jl`: attractor `id` from `Attractors.jl`'s `BasinMapRecurrences` is categorical,
  and for this system the two basins are **riddled** through each other — fractal, and fine-grained
  at every scale. A local search confined to one attractor (MCX's `MetropolisAlgorithm` with a
  0/1 "on-target" energy) starts rejecting proposals at any step size, however small — riddling made
  concrete — and a finite-temperature `BoltzmannEnsemble` turns that hard confinement into a
  continuous knob on how much wrong-attractor visitation to tolerate.

Each script activates this folder's dedicated environment (`Project.toml`) so
package-compatibility constraints stay isolated from the main examples env.
