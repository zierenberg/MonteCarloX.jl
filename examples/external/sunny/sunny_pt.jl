# # Parallel tempering on the 2D Ising model
#
# Four stages, each a single `run_pt_*` function that times system construction, sampling, and
# WHAM analysis independently:
#
# 1. Sunny native
# 2. Sunny model + MCX parallel tempering
# 3. full MCX
# 4. compare reconstructed density of states and runtime
#
# Reference: [Sunny's PT and WHAM example](https://github.com/SunnySuite/Sunny.jl/blob/main/examples/extra/Advanced_MC/PT_WHAM_ising2d.jl)
#
# Run: `julia --project=examples/external/sunny -t auto examples/external/sunny/sunny_pt.jl`

using Random, Statistics, Plots, Printf, DelimitedFiles, Markdown
using Sunny, MonteCarloX, MCXSpins
import MonteCarloX: histogram
import MCXSpins: energy, logdos_exact_ising2D

SEED = 42
L = 20
kT_min, kT_max = 0.5, 10.0
n_replicas = 40
n_therm = 1_000
n_measure = 2_000
measure_interval = 10
exch_interval = 5
nothing # hide

# ## 1. Sunny native
#
# We start from the reference workflow, where Sunny manages the system, the proposal, and the
# entire parallel-tempering loop.

function run_pt_sunny(; n_therm=n_therm, n_measure=n_measure)
    t_create = @elapsed begin
        latvecs = Sunny.lattice_vectors(1, 1, 10, 90, 90, 90)
        sys = Sunny.System(Sunny.Crystal(latvecs, [[0, 0, 0]]),
                           [1 => Sunny.Moment(s=1, g=-1)], :dipole; dims=(L, L, 1), seed=0)
        Sunny.polarize_spins!(sys, [0, 0, 1])
        Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, (1, 0, 0)))
        kT_sched = collect(range(kT_min, kT_max, length=n_replicas)) 
        PT = Sunny.ParallelTempering(sys, Sunny.LocalSampler(; kT=0, propose=Sunny.propose_flip), kT_sched)
        E_hists = [Sunny.Histogram(bin_size=1.0) for _ in 1:PT.n_replicas]
    end
    t_run = @elapsed begin
        Sunny.step_ensemble!(PT, n_therm, exch_interval)
        for _ in 1:n_measure
            Sunny.step_ensemble!(PT, measure_interval, exch_interval)
            for (j, sampler) in enumerate(PT.samplers)
                E_hists[j][sampler.ΔE] += 1
            end
        end
    end
    t_wham = @elapsed dos = Sunny.WHAM(E_hists, kT_sched; n_iters=10_000) 
    dos, PT.n_accept ./ PT.n_exch, (t_create, t_run, t_wham)
end

# ## 2. Sunny model + MCX algorithm
#
# Here, we want to show how to use Sunny models with algorithms from MonteCarloX. Since Sunny does not save the energy inside the model, we need a small wrapper and take the opportunity to set up a constructor.

mutable struct SunnyIsing{S}
    sys::S
    energy::Float64
end
function SunnyIsing(L::Int)
    latvecs = Sunny.lattice_vectors(1, 1, 10, 90, 90, 90)
    sys = Sunny.System(Sunny.Crystal(latvecs, [[0, 0, 0]]),
                       [1 => Sunny.Moment(s=1, g=-1)], :dipole; dims=(L, L, 1))
    Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, (1, 0, 0)))
    Sunny.polarize_spins!(sys, (0, 0, 1))
    SunnyIsing(sys, Sunny.energy_per_site(sys) * L^2)
end
MCXSpins.energy(s::SunnyIsing) = s.energy

# To match the interface of a native MonteCarloX system, we overload the `spin_flip!` function. The local `sweep!` then mirrors Sunny's built-in; the only difference is the wrapped single-site step (`accept!` + `setspin!` instead of Sunny's inlined Metropolis test).

function MCXSpins.spin_flip!(s::SunnyIsing, alg)
    site = rand(alg.rng, Sunny.eachsite(s.sys))
    prop = Sunny.propose_flip(s.sys, site)
    ΔE = Sunny.local_energy_change(s.sys, site, prop)
    if accept!(alg, ΔE)
        Sunny.setspin!(s.sys, prop, site)
        s.energy += ΔE
    end
end

sweep!(sys, alg, n) = (for _ in 1:n, _ in 1:L^2; spin_flip!(sys, alg); end)

# Now we can run the same parallel tempering loop as in the Sunny-native example, but with a MonteCarloX algorithm. The `with_parallel` block handles the parallelization and the replica-exchange logic. Here, we explicitly control the loops which gives us access into the systems internal state at each point in time.

function run_pt_bridge(; n_therm=n_therm, n_measure=n_measure)
    t_create = @elapsed begin
        kT_sched = collect(range(kT_min, kT_max, length=n_replicas))
        systems = [SunnyIsing(L) for _ in 1:n_replicas]
        pt = ParallelTempering(1 ./ kT_sched; seed=SEED, rng=Xoshiro)
        E_hists = [histogram(-2L^2:4:2L^2) for _ in 1:n_replicas]
    end
    t_run = @elapsed begin
        for _ in 1:(n_therm ÷ exch_interval)
            with_parallel(pt) do r, alg
                sweep!(systems[r], alg, exch_interval)
            end
            MonteCarloX.update!(pt, energy.(systems))
        end
        for _ in 1:n_measure
            for _ in 1:(measure_interval ÷ exch_interval)
                with_parallel(pt) do r, alg
                    sweep!(systems[r], alg, exch_interval)
                end
                MonteCarloX.update!(pt, energy.(systems))
            end
            for (r, s) in enumerate(systems)
                E_hists[index(pt, r)][energy(s)] += 1
            end
        end
    end
    t_wham = @elapsed dos = wham(E_hists, kT_sched, n_iters=10_000)
    dos, acceptance_rates(pt), (t_create, t_run, t_wham)
end

# ## 3. Full MCX
#
# Finally, we drop the Sunny glue entirely and build the systems from a native `IsingSystem`. The
# sampling loop is identical to the bridge — which is exactly the point: once a model exposes
# `spin_flip!` and `energy`, the same MonteCarloX algorithm drives it unchanged.

function run_pt_mcx(; n_therm=n_therm, n_measure=n_measure)
    t_create = @elapsed begin
        kT_sched = collect(range(kT_min, kT_max, length=n_replicas)) 
        systems = [IsingSystem([L, L]) for _ in 1:n_replicas]
        pt = ParallelTempering(1 ./ kT_sched; seed=SEED, rng=Xoshiro)
        E_hists = [histogram(-2L^2:4:2L^2) for _ in 1:n_replicas]
        for s in systems; init!(s, :up) end
    end
    t_run = @elapsed begin
        for _ in 1:(n_therm ÷ exch_interval)
            with_parallel(pt) do r, alg
                sweep!(systems[r], alg, exch_interval)
            end
            MonteCarloX.update!(pt, energy.(systems))
        end
        for _ in 1:n_measure
            for _ in 1:(measure_interval ÷ exch_interval)
                with_parallel(pt) do r, alg
                    sweep!(systems[r], alg, exch_interval)
                end
                MonteCarloX.update!(pt, energy.(systems))
            end
            for (r, s) in enumerate(systems)
                E_hists[index(pt, r)][energy(s)] += 1
            end
        end
    end
    t_wham = @elapsed dos = wham(E_hists, kT_sched, n_iters=10_000)
    dos, acceptance_rates(pt), (t_create, t_run, t_wham)
end

# ## 4. Comparison
#
# We time `create`, `run`, and `wham` independently for each implementation, reconstruct the
# density of states, and — as in the mcmc examples — cache both the timings and the reconstructed
# curves to TSV so the table and figure regenerate without rerunning the simulation. `<p_exch>` is
# the mean replica-exchange acceptance over the finite ladder edges.

datadir     = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "data")))  # hide
dos_file    = joinpath(datadir, "sunny_pt_L$(L)_dos.tsv")     # hide
timing_file = joinpath(datadir, "sunny_pt_L$(L)_timing.tsv")  # hide

## Restrict each ln-DoS to the common energy window and anchor it to zero at the lowest      # hide
## sampled energy (the Beale convention); `Emax` optionally caps the upper energy.            # hide
function common_logdos(curves...; Emax=nothing)                                               # hide
    lo = maximum(minimum(E) for (E, _) in curves)                                             # hide
    hi = Emax === nothing ? minimum(maximum(E) for (E, _) in curves) : Emax                   # hide
    map(curves) do (E, log_g)                                                                 # hide
        m = lo .<= E .<= hi                                                                   # hide
        (E[m], log_g[m] .- log_g[m][argmin(E[m])])                                            # hide
    end                                                                                       # hide
end                                                                                           # hide

if !isfile(dos_file)                                                        # hide
    run_pt_sunny(n_therm=exch_interval, n_measure=1)                        # hide
    run_pt_bridge(n_therm=exch_interval, n_measure=1)                       # hide
    run_pt_mcx(n_therm=exch_interval, n_measure=1)                          # hide
    sunny_dos,  sunny_A,  sunny_t  = run_pt_sunny()                         # hide
    bridge_dos, bridge_A, bridge_t = run_pt_bridge()                        # hide
    mcx_dos,    mcx_A,    mcx_t    = run_pt_mcx()                           # hide

    # overlay the exact Beale density of states (MCXSpins) when it is tabulated for this L
    labeled = [("Sunny", sunny_dos), ("MCX bridge", bridge_dos), ("MCX native", mcx_dos)]
    if isfile(joinpath(pkgdir(MCXSpins), "data", "exact_solutions", "ising2D_$(L)x$(L).csv"))
        vex = logdos_exact_ising2D(L; format=:vector)
        push!(labeled, ("exact", (first.(vex), last.(vex))))
    end

    # reconstructed curves on a shared energy grid (NaN where a curve has no data)
    names  = first.(labeled)
    normed = common_logdos((c[2] for c in labeled)...)
    Egrid  = sort(unique(reduce(vcat, [E for (E, _) in normed])))
    pos    = Dict(e => i for (i, e) in enumerate(Egrid))
    dosmat = fill(NaN, length(Egrid), length(names))
    for (j, (E, p)) in enumerate(normed), k in eachindex(E)
        dosmat[pos[E[k]], j] = p[k]
    end

    # mean/min/max replica-exchange acceptance over ladder edges (min flags a tunneling bottleneck)
    pstats(A) = (a = filter(isfinite, A); (mean(a), minimum(a), maximum(a)))
    runs    = (("Sunny native", sunny_t, sunny_A), ("MCX bridge", bridge_t, bridge_A), ("MCX native", mcx_t, mcx_A))
    timemat = reduce(vcat, permutedims([n, t[1], t[2], t[3], pstats(A)...]) for (n, t, A) in runs)

    mkpath(datadir)                                                                        # hide
    writedlm(dos_file,    [permutedims(["E"; names]); hcat(Egrid, dosmat)], '\t')          # hide
    writedlm(timing_file, [["implementation" "create" "run" "wham" "p_mean" "p_min" "p_max"]; timemat], '\t')  # hide
end                                                                         # hide

# The three implementations, timed side by side. `create`/`run`/`wham` are the three timed
# stages; `⟨p_exch⟩` is the mean replica-exchange acceptance over the ladder edges (min–max in
# parentheses flag a tunneling bottleneck). Identical physics, so the numbers are directly
# comparable.

timedata = readdlm(timing_file, '\t'; header=true)[1]                                          # hide
io = IOBuffer()                                                                                 # hide
println(io, "| implementation | create [s] | run [s] | wham [s] | ⟨p_exch⟩ (min–max) |")       # hide
println(io, "|---|---:|---:|---:|---:|")                                                        # hide
for r in axes(timedata, 1)                                                                      # hide
    println(io, @sprintf("| %s | %.3f | %.3f | %.3f | %.3f (%.3f–%.3f) |",                      # hide
        timedata[r, 1], timedata[r, 2], timedata[r, 3], timedata[r, 4],                         # hide
        timedata[r, 5], timedata[r, 6], timedata[r, 7]))                                        # hide
end                                                                                             # hide
Markdown.parse(String(take!(io)))                                                               # hide

# The reconstructed log-density of states, all three implementations against the exact Beale
# reference — they overlap, which is the correctness check that makes the timing table meaningful.

dosdata, doshdr = readdlm(dos_file, '\t'; header=true)                                          # hide
E = dosdata[:, 1]                                                                               # hide
plt = plot(; xlabel="E", ylabel="ln p(E)", title="2D Ising PT, L=$L")                           # hide
for j in 2:size(dosdata, 2)                                                                     # hide
    plot!(plt, E, dosdata[:, j]; label=doshdr[j], lw=2,                                         # hide
          ls=(doshdr[j] == "exact" ? :dash : :solid), lc=(doshdr[j] == "exact" ? :black : :auto))  # hide
end                                                                                             # hide
savefig(plt, joinpath(@__DIR__, "sunny_pt_dos.png"))                                            # hide
plt                                                                                             # hide
