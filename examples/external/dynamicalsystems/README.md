# External/DynamicalSystems

Examples connecting MonteCarloX to [DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystems.jl/stable/)
and [Attractors.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/attractors/stable/), each
matching a different MCX algorithm to a different dynamical-systems order parameter:

- `forgetting_time.jl`: the initial condition `u0` of a 4D coupled standard map is the MCX state,
  and the **time to forget** `u0` (how long a tangent vector takes to grow past a threshold,
  computed via `DynamicalSystems.jl`'s `TangentDynamicalSystem`) is a well-behaved discrete order
  parameter — a direct fit for MCX's `WangLandauAlgorithm`. Reproduces the actual system and
  parameters from Kitajima & Iba, *Multicanonical Sampling of Rare Trajectories in Chaotic
  Dynamical Systems* ([arXiv:1003.2013](https://arxiv.org/abs/1003.2013)): a pair of genuinely
  rare regular islands (probability ~`10⁻⁵`–`10⁻⁶`, not just "uncommon"), where naive sampling is
  demonstrably starved of hits in the tail and Wang-Landau resolves it from a single training run.
- `riddled_basins.jl`: attractor `id` from `Attractors.jl`'s `BasinMapRecurrences` is categorical,
  and for this system the two basins are **riddled** through each other — fractal, and fine-grained
  at every scale. A local search confined to one attractor (MCX's `MetropolisAlgorithm` with a
  0/1 "on-target" energy) starts rejecting proposals at any step size, however small — riddling made
  concrete — and a finite-temperature `BoltzmannEnsemble` turns that hard confinement into a
  continuous knob on how much wrong-attractor visitation to tolerate.

Each script activates this folder's dedicated environment (`Project.toml`) so
package-compatibility constraints stay isolated from the main examples env.
