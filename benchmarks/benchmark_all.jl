# # Spin systems
#
# The MCXSpins spin systems — Ising and Heisenberg — measured against the **prime examples** of other Julia Monte Carlo packages: MonteCarlo.jl and
# Carlo.jl on 2D Ising, SpinMC.jl on the cubic Heisenberg magnet.  Each package
# runs a model it showcases as example online; MCX runs the identical physics
# with its plain generic loop — an `IsingSystem` or `HeisenbergSystem` driven by
# `spin_flip!` under a `MetropolisAlgorithm` (the MCX code is shown for each). 
# Both run on the same machine in the same session.
# 
# Physics is checked against an exact referee wherever one exists (Beale's
# 2D-Ising log-DOS via `reweight`).  Speed is the **MCX speedup**
# ``t_\text{reference}/t_\text{MCX}`` per attempted flip — above 1 means MCX is
# faster; the ratio cancels the machine. The [Benchmarks](@ref) overview
# collects every speedup.
# 
# A dedicated **compiled C program** (`ising_cpu_modes.c`) running the same 2D
# Ising sets the bare-metal speed ceiling, where MCX is inevitably slower
# (speedup below 1); the fuller hand-optimized-Julia dissection lives in the
# MCXSpins top-performance benchmark.
# 
# The heavy runs are cached to `docs/src/data/` — delete `bench_*.tsv` and
# `benchmarks.tsv` there and rerun `julia --project=benchmarks
# benchmarks/benchmark_all.jl` to regenerate.
#
# External framework comparisons are isolated by package:
# `benchmarks/MonteCarlo`, `benchmarks/Carlo`, `benchmarks/SpinMC`, and
# `benchmarks/Sunny` each contain their own `Project.toml` + `benchmark.jl`.
# This file is the orchestrator and only spawns those scripts when cache files
# are missing.

import Pkg; Pkg.activate(joinpath(@__DIR__)); Pkg.instantiate()  #src

using Random, Printf, Plots, DelimitedFiles, Markdown
using MonteCarloX, MCXSpins

datadir = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "docs", "src", "data")))  # hide
factors_file = joinpath(datadir, "benchmarks.tsv")                                # hide
bench_file(name) = joinpath(datadir, "bench_$(name).tsv")                         # hide
status(msg) = println("[benchmark_all] ", msg)                                     # hide
function recfactor(comparison, case, ref_ns, mcx_ns)                              # hide
    new = !isfile(factors_file)                                                   # hide
    open(factors_file, "a") do io                                                 # hide
        new && println(io, "comparison\tcase\tcpu\tref_ns_per_flip\tmcx_ns_per_flip\tspeedup_mcx")  # hide
        @printf(io, "%s\t%s\t%s\t%.3f\t%.3f\t%.3f\n",                             # hide
                comparison, case, Sys.CPU_NAME, ref_ns, mcx_ns, ref_ns / mcx_ns)  # hide
    end                                                                           # hide
end                                                                               # hide

# Two conventions used throughout. Timing is the minimum over interleaved repetitions of a pure update loop, per attempted flip:

ns_per_flip(run, nflips; reps=3) = 1e9 * minimum(@elapsed(run()) for _ in 1:reps) / nflips
nothing # hide

# and the MCX contender is always the plain generic loop — `IsingSystem` (or `HeisenbergSystem`) driven by `spin_flip!` under a `MetropolisAlgorithm`. The hot loop always runs through the named `mcx_sweep!` below, so `spin_flip!` executes under a function barrier with concretely-typed `sys`/`alg`:

mcx_sweep!(sys, alg, n) = (for _ in 1:n; spin_flip!(sys, alg); end)
nothing # hide

# Each reference is matched on site-selection: Carlo.jl and SpinMC.jl pick a random site per step (two rng draws), so they meet MCX's default random-site `spin_flip!`; MonteCarlo.jl sweeps sites in typewriter order with a single draw per step, so it is met by the site-taking primitive `spin_flip!(sys, alg, i)` run over `eachindex` — same access pattern, same one draw:

mcx_sweep_seq!(sys, alg, nsweeps) = (for _ in 1:nsweeps, i in eachindex(sys.spins); spin_flip!(sys, alg, i); end)
nothing # hide

# Both sides draw from the same generator: MCX seeds a `Xoshiro`, which is Julia's default RNG (`Random.default_rng()`, a Xoshiro256++ since Julia 1.7). The reference frameworks use that same default — SpinMC.jl copies `Random.GLOBAL_RNG`, MonteCarlo.jl uses `Random.default_rng()`, Carlo.jl takes an `Xoshiro` context — so the comparison is not skewed by RNG throughput.

function mcx_ising_time_seq(L, β; sweeps=5_000)
    sys = IsingSystem([L, L]); N = L * L
    init!(sys, :random, rng=MersenneTwister(2))
    alg = MetropolisAlgorithm(Xoshiro(1); β=β)
    mcx_sweep_seq!(sys, alg, 200)
    return ns_per_flip(() -> mcx_sweep_seq!(sys, alg, sweeps), sweeps * N)
end
nothing # hide

# ## Speed ceiling: compiled C
#
# Every framework below pays for generality; the floor is a dedicated **compiled C program**
# (`MCXSpins/references/ising_cpu_modes.c`) running the identical 2D Ising physics — continuous
# acceptance, xoshiro RNG, typewriter sweep at β = 0.44 — reporting its own ns/flip. MCX is inevitably slower; the point is to measure how much the generic `spin_flip!`
# loop costs against bare metal. The C sweeps in typewriter order with one draw per step, so the
# MCX contender is the sequential sweep (`mcx_ising_time_seq`) — same access pattern, same one draw.
# The physics is the same Ising already refereed below, so this row is a pure speed ceiling; the C
# is compiled with the system `cc` only when regenerating:

function c_ceiling(; L=64)                                                        # hide
    src = joinpath(@__DIR__, "..", "MCXSpins", "references", "ising_cpu_modes.c") # hide
    bin = joinpath(mktempdir(), "ising_cpu_modes")                                # hide
    run(`cc -O3 -march=native -std=c11 -o $bin $src -lm`)                         # hide
    row = split(last(split(readchomp(`$bin cont-std xoshiro 2026`), '\n')))       # hide
    return parse(Float64, row[6]), -parse(Int, row[8]) / (L * L)   # ns/flip, e/site  # hide
end                                                                               # hide

if !isfile(bench_file("cceiling"))                                                # hide
status("regenerating cceiling")                                                   # hide
c_ns, c_e = c_ceiling()                                                           # hide
mcx_ns = mcx_ising_time_seq(64, 0.44)                                             # hide
recfactor("optimized C kernel (2D Ising 64×64)", "cont/xoshiro", c_ns, mcx_ns)   # hide
writedlm(bench_file("cceiling"), [["c_ns" "mcx_ns" "c_e_site"]; [c_ns mcx_ns c_e]], '\t')  # hide
else                                                                              # hide
status("using cached cceiling")                                                  # hide
end                                                                               # hide

d = readdlm(bench_file("cceiling"), '\t'; header=true)[1]                         # hide
Markdown.parse(@sprintf(                                                          # hide
    "Compiled C **%.2f ns/flip** (e/site = %.3f, the refereed Ising) vs MCX sequential **%.2f ns/flip** → MCX speedup **%.2f×** — the constant cost of the generic `spin_flip!` loop against bare-metal C.",  # hide
    d[1, 1], d[1, 3], d[1, 2], d[1, 1] / d[1, 2]))                                # hide

# ## MonteCarlo.jl
#
# Reference: [MonteCarlo.jl](https://github.com/carstenbauer/MonteCarlo.jl) on its
# [`example/ising2d`](https://github.com/carstenbauer/MonteCarlo.jl/tree/master/example/ising2d)
# prime example — 2D Ising, `IsingModel` + `MC` + `run!` with 10⁴
# thermalization and 10³ measurement sweeps. The exact Beale density of states referees
# the energy. Physics runs the full `run!` pipeline; speed times the bare per-sweep kernel
# `MonteCarlo.sweep` (as for Carlo below), so construction, thermalization, measurement and
# checkpoint scheduling stay outside the number. MonteCarlo.jl sweeps sites in typewriter
# order with a single rng draw per step, so MCX is matched with the same access pattern —
# the site-taking `spin_flip!(sys, alg, i)` over `eachindex` (`mcx_ising_time_seq`):

if !isfile(bench_file("montecarlo"))                                              # hide
status("running MonteCarlo benchmark env")                                        # hide
run(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "MonteCarlo")) $(joinpath(@__DIR__, "MonteCarlo", "benchmark.jl"))`)  # hide
else                                                                              # hide
status("using cached MonteCarlo benchmark")                                      # hide
end                                                                               # hide

d = readdlm(bench_file("montecarlo"), '\t'; header=true)[1]                       # hide
plot(d[:, 1], d[:, 2]; marker=:circle, lw=2, label="MonteCarlo.jl",               # hide
     ylabel="e per site", title="2D Ising 8×8, exact referee",                    # hide
     layout=(2, 1), legend=:bottomright, subplot=1)                               # hide
plot!(d[:, 1], d[:, 3]; marker=:diamond, ls=:dash, lw=2, label="MCX", subplot=1)  # hide
plot!(d[:, 1], d[:, 4]; ls=:dot, color=:gray, lw=2, label="exact", subplot=1)     # hide
plot!(d[:, 1], d[:, 2] .- d[:, 4]; marker=:circle, lw=2,                          # hide
      label="MonteCarlo.jl − exact", xlabel="T", ylabel="Δe vs exact", subplot=2) # hide
plot!(d[:, 1], d[:, 3] .- d[:, 4]; marker=:diamond, lw=2, label="MCX − exact", subplot=2)  # hide

# ## Carlo.jl
#
# Reference: [Carlo.jl](https://github.com/lukas-weber/Carlo.jl) is a *framework*
# (scheduler, MPI, HDF5 checkpointing, binning) — the sweep kernel comes from its
# reference model [Ising.jl](https://github.com/lukas-weber/Ising.jl), whose
# [example job](https://github.com/lukas-weber/Ising.jl/tree/main/example)
# is 2D Ising at 20 000 sweeps + 2 000 thermalization, binsize 100. Physics runs the FULL
# pipeline (job → `results.json`); speed times the bare `Carlo.sweep!` kernel:

if !isfile(bench_file("carlo"))                                                   # hide
status("running Carlo benchmark env")                                             # hide
run(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "Carlo")) $(joinpath(@__DIR__, "Carlo", "benchmark.jl"))`)  # hide
else                                                                              # hide
status("using cached Carlo benchmark")                                           # hide
end                                                                               # hide

d = readdlm(bench_file("carlo"), '\t'; header=true)[1]                            # hide
plot(d[:, 1], d[:, 2]; marker=:circle, lw=2, label="Carlo.jl pipeline",           # hide
     ylabel="e per site", title="2D Ising 8×8, exact referee",                    # hide
     layout=(2, 1), legend=:bottomright, subplot=1)                               # hide
plot!(d[:, 1], d[:, 3]; marker=:diamond, ls=:dash, lw=2, label="MCX", subplot=1)  # hide
plot!(d[:, 1], d[:, 4]; ls=:dot, color=:gray, lw=2, label="exact", subplot=1)     # hide
plot!(d[:, 1], d[:, 2] .- d[:, 4]; marker=:circle, lw=2, label="Carlo.jl − exact",  # hide
      xlabel="T", ylabel="Δe vs exact", subplot=2)                                # hide
plot!(d[:, 1], d[:, 3] .- d[:, 4]; marker=:diamond, lw=2, label="MCX − exact", subplot=2)  # hide

# ## SpinMC.jl
#
# Reference: [SpinMC.jl](https://github.com/fbuessen/SpinMC.jl) on its
# [cubic-magnetization prime example](https://github.com/fbuessen/SpinMC.jl#specific-heat-and-magnetization-of-a-cubic-lattice-heisenberg-ferromagnet)
# — the 8×8×8 ferromagnetic Heisenberg model (interaction matrix −𝟙 on the
# three nearest-neighbor bonds), single-site Metropolis with uniform-on-sphere proposals,
# the same proposal MCX uses. There is no exact finite-size referee, so ``|m|(T)`` is
# compared code-vs-code with statistical errors:

if !isfile(bench_file("spinmc"))                                                  # hide
status("running SpinMC benchmark env")                                            # hide
run(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "SpinMC")) $(joinpath(@__DIR__, "SpinMC", "benchmark.jl"))`)  # hide
else                                                                              # hide
status("using cached SpinMC benchmark")                                          # hide
end                                                                               # hide

d = readdlm(bench_file("spinmc"), '\t'; header=true)[1]                           # hide
plot(d[:, 1], d[:, 2]; yerror=d[:, 3], marker=:circle, lw=2, label="SpinMC.jl",   # hide
     xscale=:log10, ylabel="⟨|m|⟩", title="Heisenberg 8×8×8 (Tc ≈ 1.443)",        # hide
     layout=(2, 1), legend=:bottomleft, subplot=1)                                # hide
plot!(d[:, 1], d[:, 4]; yerror=d[:, 5], marker=:diamond, ls=:dash, lw=2,          # hide
      label="MCX", subplot=1)                                                     # hide
plot!(d[:, 1], d[:, 4] .- d[:, 2]; yerror=sqrt.(d[:, 3].^2 .+ d[:, 5].^2),        # hide
      marker=:circle, lw=2, color=:gray, label="MCX − SpinMC.jl",                 # hide
      xscale=:log10, xlabel="T", ylabel="Δ⟨|m|⟩", subplot=2)                      # hide

# ## Sunny.jl
#
# Reference: [Sunny.jl](https://github.com/SunnySuite/Sunny.jl) on its
# [Monte Carlo Ising prime example](https://github.com/SunnySuite/Sunny.jl/blob/main/examples/05_MC_Ising.jl)
# — 2D Ising with a `System` and `LocalSampler(..., propose=propose_flip)`.
# Physics is checked against the exact Beale referee on a small lattice; speed is measured
# for the local-update kernel in ns/attempted flip, matched to MCX's random-site `spin_flip!`.
#
# Sunny depends on a newer JLD2 than MonteCarlo.jl, so this benchmark runs in
# a dedicated environment (`benchmarks/Sunny/Project.toml`). If the cached data
# file is missing, we regenerate it by spawning `benchmarks/Sunny/benchmark.jl`.

if !isfile(bench_file("sunny_ising"))                                             # hide
status("running Sunny benchmark env")                                             # hide
run(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "Sunny")) $(joinpath(@__DIR__, "Sunny", "benchmark.jl"))`)  # hide
else                                                                              # hide
status("using cached Sunny benchmark")                                           # hide
end                                                                               # hide

d = readdlm(bench_file("sunny_ising"), '\t'; header=true)[1]                     # hide
plot(d[:, 1], d[:, 2]; marker=:circle, lw=2, label="Sunny.jl",                    # hide
     ylabel="e per site", title="2D Ising 8×8, exact referee",                    # hide
     layout=(2, 1), legend=:bottomright, subplot=1)                               # hide
plot!(d[:, 1], d[:, 3]; marker=:diamond, ls=:dash, lw=2, label="MCX", subplot=1)  # hide
plot!(d[:, 1], d[:, 4]; ls=:dot, color=:gray, lw=2, label="exact", subplot=1)     # hide
plot!(d[:, 1], d[:, 2] .- d[:, 4]; marker=:circle, lw=2,                          # hide
      label="Sunny.jl − exact", xlabel="T", ylabel="Δe vs exact", subplot=2)      # hide
plot!(d[:, 1], d[:, 3] .- d[:, 4]; marker=:diamond, lw=2, label="MCX − exact", subplot=2)  # hide
