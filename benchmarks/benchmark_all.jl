# # Spin systems
#
# The MCXSpins spin systems — Ising and Heisenberg — measured against the **prime examples** of other Julia Monte Carlo packages: MonteCarlo.jl and Carlo.jl on 2D Ising, SpinMC.jl on the cubic Heisenberg magnet. 
# Each package runs a model it showcases as example online; MCX runs the identical physics with its plain generic loop — an `IsingSystem` or `HeisenbergSystem` driven by `spin_flip!` under a `MetropolisAlgorithm` (the MCX code is shown for each). 
# Both run on the same machine in the same session.
#
# Physics is checked against an exact referee wherever one exists (Beale's 2D-Ising log-DOS via `reweight`). 
# Speed is the **MCX speedup** ``t_\text{reference}/t_\text{MCX}`` per attempted flip — above 1 means MCX is faster; the ratio cancels the machine. The [Benchmarks](@ref) overview collects every speedup. 
#
# The heavy runs are cached to `docs/src/data/` — delete `bench_*.tsv` and
# `benchmarks.tsv` there and rerun
# `julia --project=benchmarks benchmarks/benchmark_all.jl` to regenerate.

import Pkg; Pkg.activate(joinpath(@__DIR__)); Pkg.instantiate()  #src

using Random, Statistics, Printf, Plots, DelimitedFiles
using StatsBase: weights
using MonteCarloX, MCXSpins

datadir = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "docs", "src", "data")))  # hide
factors_file = joinpath(datadir, "benchmarks.tsv")                                # hide
bench_file(name) = joinpath(datadir, "bench_$(name).tsv")                         # hide
function recfactor(comparison, case, ref_ns, mcx_ns)                              # hide
    new = !isfile(factors_file)                                                   # hide
    open(factors_file, "a") do io                                                 # hide
        new && println(io, "comparison\tcase\tcpu\tref_ns_per_flip\tmcx_ns_per_flip\tfactor_mcx")  # hide
        @printf(io, "%s\t%s\t%s\t%.3f\t%.3f\t%.3f\n",                             # hide
                comparison, case, Sys.CPU_NAME, ref_ns, mcx_ns, mcx_ns / ref_ns)  # hide
    end                                                                           # hide
end                                                                               # hide
quiet(f) = redirect_stdout(() -> redirect_stderr(f, devnull), devnull)            # hide
float_val(x) = parse(Float64, strip(first(split(string(x), '±'))))                # hide
nothing                                                                           # hide

# Two conventions used throughout. Timing is the minimum over interleaved repetitions of a pure update loop, per attempted flip:

ns_per_flip(run, nflips; reps=3) = 1e9 * minimum(@elapsed(run()) for _ in 1:reps) / nflips

# and the MCX contender is always the plain generic loop — `IsingSystem` (or `HeisenbergSystem`) driven by `spin_flip!` under a `MetropolisAlgorithm`. The hot loop always runs through the named `mcx_sweep!` below, so `spin_flip!` executes under a function barrier with concretely-typed `sys`/`alg`:

mcx_sweep!(sys, alg, n) = (for _ in 1:n; spin_flip!(sys, alg); end)

# Each reference is matched on site-selection: Carlo.jl and SpinMC.jl pick a random site per step (two rng draws), so they meet MCX's default random-site `spin_flip!`; MonteCarlo.jl sweeps sites in typewriter order with a single draw per step, so it is met by the site-taking primitive `spin_flip!(sys, alg, i)` run over `eachindex` — same access pattern, same one draw:

mcx_sweep_seq!(sys, alg, nsweeps) = (for _ in 1:nsweeps, i in eachindex(sys.spins); spin_flip!(sys, alg, i); end)

# Both sides draw from the same generator: MCX seeds a `Xoshiro`, which is Julia's default RNG (`Random.default_rng()`, a Xoshiro256++ since Julia 1.7). The reference frameworks use that same default — SpinMC.jl copies `Random.GLOBAL_RNG`, MonteCarlo.jl uses `Random.default_rng()`, Carlo.jl takes an `Xoshiro` context — so the comparison is not skewed by RNG throughput.

function mcx_ising_em(L, β; therm=2_000, sweeps=5_000)
    sys = IsingSystem([L, L]); N = L * L
    init!(sys, :random, rng=MersenneTwister(3))
    alg = MetropolisAlgorithm(Xoshiro(4); β=β)
    mcx_sweep!(sys, alg, therm * N)
    e = m = 0.0
    for _ in 1:sweeps
        mcx_sweep!(sys, alg, N)
        e += energy(sys) / N; m += abs(magnetization(sys)) / N
    end
    return e / sweeps, m / sweeps
end

function mcx_ising_time(L, β; sweeps=5_000)
    sys = IsingSystem([L, L]); N = L * L
    init!(sys, :random, rng=MersenneTwister(2))
    alg = MetropolisAlgorithm(Xoshiro(1); β=β)
    mcx_sweep!(sys, alg, 200 * N)
    return ns_per_flip(() -> mcx_sweep!(sys, alg, sweeps * N), sweeps * N)
end

function mcx_ising_time_seq(L, β; sweeps=5_000)
    sys = IsingSystem([L, L]); N = L * L
    init!(sys, :random, rng=MersenneTwister(2))
    alg = MetropolisAlgorithm(Xoshiro(1); β=β)
    mcx_sweep_seq!(sys, alg, 200)
    return ns_per_flip(() -> mcx_sweep_seq!(sys, alg, sweeps), sweeps * N)
end
nothing # hide

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
import MonteCarlo                                                                 # hide
function mc_em(L, β; therm=10^4, sweeps=10^3)
    mc = MonteCarlo.MC(MonteCarlo.IsingModel(dims=2, L=L), beta=β)
    MonteCarlo.run!(mc; sweeps=sweeps, thermalization=therm, verbose=false)
    meas = MonteCarlo.measurements(mc)
    return mean(meas[:Energy].e), mean(meas[:Magn].m)
end

function mc_kernel_time(L, T; sweeps=5_000)
    mc = MonteCarlo.MC(MonteCarlo.IsingModel(dims=2, L=L), beta=1 / T)
    for _ in 1:200; MonteCarlo.sweep(mc); end
    return ns_per_flip(() -> (for _ in 1:sweeps; MonteCarlo.sweep(mc); end), sweeps * L * L)
end

L, Ts = 8, [1.4, 1.8, 2.269, 3.0, 3.8]
logdos = logdos_exact_ising2D(L); E = get_centers(logdos)
e_exact = [mean(E, weights(reweight(logdos, -E ./ T))) / L^2 for T in Ts]
e_ref = [mc_em(L, 1 / T)[1] for T in Ts]
e_mcx = [mcx_ising_em(L, 1 / T; therm=10^4, sweeps=10^3)[1] for T in Ts]
writedlm(bench_file("montecarlo"),                                                # hide
         [["T" "e_ref" "e_mcx" "e_exact"]; hcat(Ts, e_ref, e_mcx, e_exact)], '\t')  # hide
for T in (1.4, 2.269, 3.8)                                                        # hide
    s, Lb = 5_000, 64                                                             # hide
    recfactor("MonteCarlo.jl (2D Ising 64×64)", "T=$T",                           # hide
              mc_kernel_time(Lb, T; sweeps=s), mcx_ising_time_seq(Lb, 1 / T; sweeps=s))  # hide
end                                                                               # hide
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
using Carlo, Carlo.JobTools                                                       # hide
import Ising                                                                      # hide
function carlo_e(Ts; L=8, sweeps=20_000, therm=2_000)
    tm = TaskMaker()
    tm.sweeps = sweeps; tm.thermalization = therm; tm.binsize = 100; tm.Lx = L
    foreach(T -> (tm.T = T; task(tm)), Ts)
    dir = joinpath(mktempdir(), "job")
    job = JobInfo(dir, Ising.MC; tasks=make_tasks(tm),
                  checkpoint_time="30:00", run_time="24:00:00")
    quiet(() -> start(job, ["run", "--single"]))
    return [float_val(r["Energy"]) for r in Carlo.ResultTools.dataframe(dir * ".results.json")]
end

function carlo_kernel_time(L, T; sweeps=5_000)
    params = Dict(:Lx => L, :T => T)
    mc = Ising.MC(params)
    ctx = Carlo.MCContext(sweeps, 0, Xoshiro(1), Carlo.Measurements(100))
    Carlo.init!(mc, ctx, params)
    for _ in 1:200; Carlo.sweep!(mc, ctx); end
    return ns_per_flip(() -> (for _ in 1:sweeps; Carlo.sweep!(mc, ctx); end), sweeps * L * L)
end

L, Ts = 8, [1.0, 1.429, 2.0, 2.429, 3.0]
logdos = logdos_exact_ising2D(L); E = get_centers(logdos)
e_exact = [mean(E, weights(reweight(logdos, -E ./ T))) / L^2 for T in Ts]
e_ref = carlo_e(Ts; L=L)
e_mcx = [mcx_ising_em(L, 1 / T; therm=2_000, sweeps=20_000)[1] for T in Ts]
writedlm(bench_file("carlo"),                                                     # hide
         [["T" "e_ref" "e_mcx" "e_exact"]; hcat(Ts, e_ref, e_mcx, e_exact)], '\t')  # hide
for T in (1.0, 2.286, 3.0)                                                        # hide
    recfactor("Carlo.jl kernel (2D Ising 16×16)", "T=$T",                         # hide
              carlo_kernel_time(16, T), mcx_ising_time(16, 1 / T))                # hide
end                                                                               # hide
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
using SpinMC                                                                      # hide
function spinmc_lattice(L)
    uc = UnitCell((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    b = addBasisSite!(uc, (0.0, 0.0, 0.0))
    M = [-1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 -1.0]
    for dir in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        addInteraction!(uc, b, b, M, dir)
    end
    return Lattice(uc, (L, L, L))
end

function spinmc_m(T; L=8, therm=2_000, sweeps=5_000)
    mc = SpinMC.MonteCarlo(spinmc_lattice(L), 1 / T, therm, sweeps; reportInterval=10^9)
    quiet(() -> SpinMC.run!(mc))
    return mean(mc.observables.magnetization), SpinMC.std_error(mc.observables.magnetization)
end

function mcx_heisenberg_m(T; L=8, therm=2_000, sweeps=5_000)
    sys = HeisenbergSystem([L, L, L]); N = L^3
    init!(sys, :random, rng=MersenneTwister(3))
    alg = MetropolisAlgorithm(Xoshiro(4); β=1 / T)
    mcx_sweep!(sys, alg, therm * N)
    ms = [(mcx_sweep!(sys, alg, N);
           sqrt(sum(abs2, magnetization(sys))) / N) for _ in 1:sweeps]
    return mean(ms), std(ms) / sqrt(length(ms) / 100)        ## crude blocking (100 sweeps)
end

Ts = [10.0 * (0.1 / 10.0)^((i - 1) / 7) for i in 1:8]        ## the example's log grid
ref = spinmc_m.(Ts)
mcx = mcx_heisenberg_m.(Ts)
writedlm(bench_file("spinmc"),                                                    # hide
         [["T" "m_ref" "dm_ref" "m_mcx" "dm_mcx"];                                # hide
          hcat(Ts, first.(ref), last.(ref), first.(mcx), last.(mcx))], '\t')      # hide
for T in (0.5, 1.443, 10.0)                                                       # hide
    s, N = 5_000, 512                                                             # hide
    lat = spinmc_lattice(8)                                                       # hide
    t_ref = ns_per_flip(s * N) do                                                 # hide
        quiet(() -> SpinMC.run!(SpinMC.MonteCarlo(lat, 1 / T, s, 0; reportInterval=10^9)))  # hide
    end                                                                           # hide
    sys = HeisenbergSystem([8, 8, 8])                                             # hide
    init!(sys, :random, rng=MersenneTwister(2))                                   # hide
    alg = MetropolisAlgorithm(Xoshiro(1); β=1 / T)                                # hide
    mcx_sweep!(sys, alg, 200 * N)                                                 # hide
    t_mcx = ns_per_flip(() -> mcx_sweep!(sys, alg, s * N), s * N)                 # hide
    recfactor("SpinMC.jl (Heisenberg 8×8×8)", "T=$T", t_ref, t_mcx)               # hide
end                                                                               # hide
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
