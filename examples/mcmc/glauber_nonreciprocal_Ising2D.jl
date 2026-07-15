# # Glauber Dynamics of the Nonreciprocal (Vision-Cone) Ising Model
#
# This example runs the nonreciprocal 2D Ising model, where each spin couples more
# strongly to the neighbors *ahead* of its polarization
# ``\hat p_i = \sigma_i(\hat x+\hat y)/\sqrt2`` (a "vision cone"):
# ```math
# E_i = -\sigma_i\sum_j J_{ij}\sigma_j,\qquad
# J_{ij} = \begin{cases} J + \kappa, & \hat p_i\cdot\hat e_{ij}>0\\ J,&\text{otherwise.}\end{cases}
# ```
# The coupling is asymmetric (``J_{ij}\neq J_{ji}``), so reciprocity and detailed
# balance are broken and there is no global Hamiltonian — the model is defined by
# its Glauber update. Nonreciprocity leaves the transition continuous but shifts the
# critical temperature upward with ``\kappa``.
# Reference: Garcés & Levis, J. Stat. Mech. 2025 043205 (cf. arXiv:2606.06981, which
# uses a temperature-scaled coupling ``J+\lambda/\beta``).
#
# The coupling ``\kappa`` is temperature-independent, so the local energy is β-free and
# the nonreciprocal system — a composed `SpinSystem{model, nr, topo}` — plugs into the
# same `spin_flip!`/`accept!` interface as the equilibrium models, with no special update.

import Pkg; Pkg.activate(joinpath(@__DIR__, "..")); Pkg.instantiate()  #src

using Random, Statistics, Plots, BenchmarkTools
using MonteCarloX, MCXSpins

# ## Parameters

L    = 32;
seed = 42;

# ## System construction and per-flip benchmark
#
# We build the reciprocal (``\kappa=0``) and nonreciprocal (``\kappa=1``) systems
# and time a single `spin_flip!`. For reference we also time the optimized
# equilibrium `Ising` fast path, to see the cost of the general composed system.

alg = Glauber(Xoshiro(seed); β=1/2.5)

sys_eq = IsingSystem([L, L])
init!(sys_eq, :random, rng=MersenneTwister(seed))

sys_rec = VisionConeIsingSystem([L, L]; κ=0.0)
init!(sys_rec, :random, rng=MersenneTwister(seed))

sys_nr = VisionConeIsingSystem([L, L]; κ=1.0)
init!(sys_nr, :random, rng=MersenneTwister(seed))

println("Equilibrium Ising (specialized fast path):")
@btime spin_flip!($sys_eq, $alg)
println("Nonreciprocal Ising, κ=0 (composed SpinSystem):")
@btime spin_flip!($sys_rec, $alg)
println("Nonreciprocal Ising, κ=1:")
@btime spin_flip!($sys_nr, $alg)

# The composed `SpinSystem` is somewhat slower than the hand-specialized `Ising`
# lattice, and turning on ``\kappa`` adds negligible cost — the vision cone is just
# a directional partition of the same neighbor sum.

# ## Simulation helper
#
# A sweep is `N` single-spin `spin_flip!` calls, exactly as for the equilibrium
# models. We measure the mean absolute magnetization per spin.

function mean_abs_m(κ, T; L=L, warmup=100_000, samples=100_000, seed=seed)
    rng = MersenneTwister(seed)
    sys = VisionConeIsingSystem([L, L]; κ=κ*T)
    init!(sys, :up)
    alg = Glauber(rng; β=1/T)
    N = L * L
    for _ in 1:warmup, _ in 1:N
        spin_flip!(sys, alg)
    end
    acc = 0.0
    for _ in 1:samples
        for _ in 1:N
            spin_flip!(sys, alg)
        end
        acc += abs(magnetization(sys)) / N
    end
    return acc / samples
end

# ## Nonreciprocity raises the critical temperature
#
# Scanning temperature for several ``\kappa`` shows the ordering crossover moving to
# higher ``T`` as nonreciprocity increases, while ``T_c(0)\approx2.269`` recovers the
# equilibrium 2D Ising value.

Ts = 2.0:0.5:10.0
p = plot(xlabel="Temperature T", ylabel="⟨|m|⟩",
         title="Nonreciprocity raises Tc", legend=:topright)
for κ in (0.0, 0.5, 1.0)
    plot!(p, Ts, [mean_abs_m(κ, T) for T in Ts]; marker=:circle, lw=2, label="κ = $κ")
end
vline!(p, [2.269]; ls=:dash, color=:gray, label="Tc(κ=0) ≈ 2.269")
p
