# MCXSpins.jl

A companion package of MonteCarloX.jl providing spin systems.

## Overview

MCXSpins provides spin systems for MonteCarloX.jl's Monte Carlo algorithms. This
separation lets MonteCarloX focus on algorithms while MCXSpins handles model-specific
details.

A system is composed of a **spin type** (the local degree of freedom) and a tuple of
**interactions** (the energy terms):

```julia
SpinSystem(Spin(1), (PairInteraction(J, partners), CrystalField(Δ)))
```

- **Spin types**: `Spin(S)` (discrete, σ-convention: `Spin(1//2)` ↦ {−1,+1}, `Spin(1)` ↦
  {−1,0,+1}, any `Spin(S)` for free), `XYSpin()` (unit phasors), `HeisenbergSpin()`
  (unit 3-vectors).
- **Interactions**: `PairInteraction` (uniform J on a local neighborhood),
  `PairInteractionMatrix` (sparse J_ij: spin glasses, Hopfield, directed couplings),
  `ExternalField` (uniform or site-dependent h), `CrystalField` (+Δ Σσ²),
  `VisionConeInteraction` (nonreciprocal).

Each interaction owns its couplings, its index structure, and an exact cache — total
energies are O(1) and integer-exact for discrete spins.

## The classic models as one-liners

The topology argument is a dims vector (periodic hypercubic lattice), a Graphs.jl
`SimpleGraph`, or a sparse coupling matrix:

```julia
IsingSystem([L, L]; J=1, h=0)                 # periodic lattice
IsingSystem(graph; J=1)                       # arbitrary graph
IsingSystem(J_sparse)                         # spin glass / directed couplings
BlumeCapelSystem([L, L]; J=1, D=0.5, h=0)
XYSystem([L, L]; J=1)                        # rotation width at the update: spin_flip!(sys, alg; Δθ)
HeisenbergSystem([L, L, L]; J=1)
VisionConeIsingSystem([L, L]; κ=0.5)          # vision cone, no full Hamiltonian
VisionConeBlumeCapelSystem([L, L]; κ=0.5, D=0.5)
HopfieldSystem(patterns)
EdwardsAndersonSystem([L, L]; rng=rng, dist=:bimodal)
```

## Example

```julia
using Random
using MonteCarloX
using MCXSpins

rng = MersenneTwister(42)
sys = IsingSystem([8, 8]; J=1)
init!(sys, :random, rng=rng)

alg = MetropolisAlgorithm(rng; β=0.4)

measurements = Measurements([
    :energy => energy => Float64[],
    :magnetization => magnetization => Float64[]
], interval=10)

for i in 1:10_000
    spin_flip!(sys, alg)
    measure!(measurements, sys, i)
end

println("Acceptance rate: ", acceptance_rate(alg))
```

## API

### Initialization
- `init!(sys, :up)` / `init!(sys, :down)`: uniform reference states
- `init!(sys, :zero)`: all σ = 0 (discrete spin types containing 0)
- `init!(sys, :random, rng=rng)`: uniform random configuration
- `set_spins!(sys, spins)`: set a configuration, rebuild all caches

### Observables
- `energy(sys; full=false)`: total energy, O(1) from the caches (throws for
  non-Hamiltonian systems); `full=true` recomputes
- `magnetization(sys; full=false)`: Σσ (Int for discrete, ComplexF64 for XY, `SVector{3,Float64}` for
  Heisenberg)
- `hamiltonian_energy(sys)`: energy of the symmetric terms only (valid multicanonical
  coordinate for nonreciprocal systems)
- `structure_factor(sys, d)`, `correlation_length(sys)`: lattice observables via the
  passive `geometry(sys)` metadata
- `delta_energy(sys, i, s_new)`: energy change of a proposed update

### Updates
- `spin_flip!(sys, alg)`: local update — Metropolis/Glauber (`MetropolisAlgorithm`,
  `GlauberAlgorithm`) or heat bath (`HeatBathAlgorithm`, generic over any discrete spin
  type)
- `cluster_update!(sys, alg)`: `Wolff(rng; β)` or `SwendsenWang(rng; β)` cluster moves
  (embedded-Ising reflections for XY/Heisenberg)
- `spin_exchange!(sys, alg)`: Kawasaki two-site exchange (conserved Σσ)
- rejection-free: `SiteEvents(sys, NFoldRates(β=β))` under `Gillespie` — the n-fold way
  through the core local-states interface (standard KMC `step!`/`modify!` pair, or
  `advance!` with time-weighted observation)

### Defining your own system
Subtype `AbstractSpinSystem` and provide the `spin_flip!` hooks — or, usually simpler,
compose a `SpinSystem` from existing interactions and add new interaction types
implementing `delta`/`delta_energy`/`commit!`/`energy`/`recompute!`.

## Benchmarks

`benchmarks/benchmark_mcxspins.jl` compares the composed system against a hand-optimized
kernel (the "custom C" speed ceiling); cross-framework comparisons live in the repository's
top-level `benchmarks/`.
