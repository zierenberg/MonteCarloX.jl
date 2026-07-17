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
# The nonreciprocal coupling and the ``\kappa``-driven ``T_c`` shift follow
# [Garcés & Levis 2025, Akritidis et al. 2026]; the update is standard Glauber dynamics [Glauber 1963]. Full
# references are listed at the bottom of this page.
#
# The coupling ``\kappa`` is temperature-independent, so the local energy is β-free and
# the nonreciprocal system — composed as `PairInteraction(J)` plus the cone extra
# `VisionConeInteraction(κ)` — plugs into the same `spin_flip!`/`accept!` interface as
# the equilibrium models, with no special update. (Per-flip cost is within a few percent
# of the equilibrium Ising system; see the benchmarks page.)

import Pkg; Pkg.activate(joinpath(@__DIR__, "..")); Pkg.instantiate()  #src

using Random, Statistics, Plots, DelimitedFiles
using MonteCarloX, MCXSpins

datadir   = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "data")))  # hide
scan_file = joinpath(datadir, "glauber_nonreciprocal_scan.tsv")  # hide
nothing #hide

# ## Parameters

L    = 32
seed = 42
Ts   = collect(2.0:0.5:10.0)
κs   = (0.0, 0.5, 1.0)
nothing # hide

# ## Simulation helper
#
# A sweep is `N` single-spin `spin_flip!` calls, exactly as for the equilibrium
# models. We measure the mean absolute magnetization per spin.

sweep!(sys, alg, n_sweeps) =
    (for _ in 1:n_sweeps, _ in 1:length(sys.spins); spin_flip!(sys, alg); end)

function mean_abs_m(κ, T; L=L, warmup=100_000, samples=100_000, seed=seed)
    sys = VisionConeIsingSystem([L, L]; κ=κ*T)
    init!(sys, :up)
    alg = GlauberAlgorithm(Xoshiro(seed); β=1/T)
    sweep!(sys, alg, warmup)
    acc = 0.0
    for _ in 1:samples
        sweep!(sys, alg, 1)
        acc += abs(magnetization(sys)) / length(sys.spins)
    end
    return acc / samples
end
nothing #hide

# ## Nonreciprocity raises the critical temperature
#
# Scanning temperature for several ``\kappa`` shows the ordering crossover moving to
# higher ``T`` as nonreciprocity increases, while ``T_c(0)\approx2.269`` recovers the
# equilibrium 2D Ising value.

if !isfile(scan_file)                                           # hide
m = [mean_abs_m(κ, T) for T in Ts, κ in κs]
header = permutedims(["T"; ["kappa$(κ)" for κ in κs]])          # hide
writedlm(scan_file, [header; hcat(Ts, m)], '\t')                # hide
end                                                             # hide
m = readdlm(scan_file, '\t'; header = true)[1][:, 2:end]        # hide

p = plot(xlabel="Temperature T", ylabel="⟨|m|⟩",
         title="Nonreciprocity raises Tc", legend=:topright)
for (k, κ) in enumerate(κs)
    plot!(p, Ts, m[:, k]; marker=:circle, lw=2, label="κ = $κ")
end
vline!(p, [2.269]; ls=:dash, color=:gray, label="Tc(κ=0) ≈ 2.269")
p

# ## References
#
# - R. J. Glauber, *Time-dependent statistics of the Ising model*, J. Math. Phys. **4**, 294 (1963).
#   [doi:10.1063/1.1703954](https://doi.org/10.1063/1.1703954)
# - A. Garcés, D. Levis, *Nonreciprocal Ising model*, J. Stat. Mech. **2025**, 043205 (2025).
# - M. Akritidis et al., *Fate of the Ising universality class under nonreciprocal interactions*, [arXiv:2606.06981](https://arxiv.org/abs/2606.06981)