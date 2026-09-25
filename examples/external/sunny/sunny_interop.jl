# # Driving Sunny models with MonteCarloX
#
# [Sunny.jl](https://github.com/SunnySuite/Sunny.jl) builds and measures spin models; MonteCarloX
# provides sampling algorithms. Connecting them needs no wrapper type, no subtyping, and no method
# defined on any MonteCarloX type — only two things, both of which Sunny already has:
#
# 1. a **local move**, written once, that asks `accept!` for the decision, and
# 2. the **reaction coordinate** the algorithm's ensemble scores — here the energy.
#
# Write those once and every MCX algorithm is available. This page runs four of them on the same
# Sunny model: parallel tempering, Wang-Landau, replica-exchange Wang-Landau, and — to show that
# the replica dynamics need not be a Markov chain at all — parallel tempering over Sunny's own
# Langevin integrator.
#
# Run: `julia -t auto examples/external/sunny/sunny_interop.jl`
# (append `--rerun` to overwrite the cached results instead of reloading them)

import Pkg; Pkg.activate(@__DIR__)  #src

using Random, Statistics, Plots, Printf, DelimitedFiles, Markdown
using Sunny, MonteCarloX
import MonteCarloX: update!
import MCXSpins: logdos_exact_ising2D

SEED = 42
L = 20                                                    # 2D Ising, L×L sites
nothing #hide

# ## The model and the glue
#
# A plain Sunny system — the 2D Ising ferromagnet of Sunny's own Monte Carlo examples.

function sunny_ising(L::Int)
    latvecs = Sunny.lattice_vectors(1, 1, 10, 90, 90, 90)
    sys = Sunny.System(Sunny.Crystal(latvecs, [[0, 0, 0]]),
                       [1 => Sunny.Moment(s=1, g=-1)], :dipole; dims=(L, L, 1))
    Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, (1, 0, 0)))
    Sunny.polarize_spins!(sys, (0, 0, 1))
    sys
end

# The move. All the domain knowledge is Sunny's (which site, which proposal, what the energy
# change is); MonteCarloX contributes only `accept!`, so `alg` may be *any* MCX Markov-chain
# algorithm carrying *any* ensemble — that is what makes the same function serve all four
# algorithms below.
#
# It is written in the two-argument `accept!(alg, E_new, E_old)` form, which is valid for every
# ensemble. Flat-histogram ensembles (Wang-Landau, multicanonical) need those absolute energies —
# they bin on them — so the energy is carried alongside the system and updated by `ΔE` rather than
# recomputed. A replica is therefore the pair `(spins, E)`; nothing is wrapped, nothing subtyped.

function update!(sys::Sunny.System, alg, E::Base.RefValue{Float64})
    site = rand(alg.rng, Sunny.eachsite(sys))
    prop = Sunny.propose_flip(sys, site)
    ΔE   = Sunny.local_energy_change(sys, site, prop)
    if accept!(alg, E[] + ΔE, E[])
        Sunny.setspin!(sys, prop, site)
        E[] += ΔE
    end
end

function ising_replica(L::Int)
    sys = sunny_ising(L)
    (spins = sys, E = Ref(Sunny.energy(sys)))
end

sweep!(r, alg) = (for _ in 1:length(Sunny.eachsite(r.spins)); update!(r.spins, alg, r.E); end)
energies(replicas) = [r.E[] for r in replicas]     # the exchange coordinate, read off the replicas

# For a canonical ensemble alone the accumulator is optional — `accept!(alg, ΔE)` never needs the
# absolute energy, and `Sunny.energy(sys)` recomputes it cheaply enough at exchange time. Keeping
# it makes one move function cover the flat-histogram algorithms too.

# ## Parallel tempering
#
# The loop is the two alternating halves written out: `advance!` runs `sweep!` on every replica in parallel up
# to the next exchange, `attempt_exchange!` attempts the swaps. Measurements are the caller's own
# business — here each replica's energy is binned into the histogram of the rung it currently sits
# on, which `index` identifies — and `wham` reweights those into the density of states.
#
# Reference for the physics: [Sunny's PT + WHAM example](https://github.com/SunnySuite/Sunny.jl/blob/main/examples/extra/Advanced_MC/PT_WHAM_ising2d.jl).

# The ladder is geometric in temperature. That matters: spaced linearly over the same range the
# exchange acceptance collapses to 0.07 at the cold end and the reconstructed low-energy tail is
# off by ~3 in `ln g`, against ~0.8 here — a ladder bottleneck, not a sampling bug.

n_replicas, kT_min, kT_max = 40, 0.5, 10.0
pt_therm, pt_measure, measure_interval, exch_interval = 1_000, 2_000, 10, 5
kT_sched = exp.(range(log(kT_min), log(kT_max), length=n_replicas))

function run_pt()
    replicas = [ising_replica(L) for _ in 1:n_replicas]
    pt = ParallelTempering(1 ./ kT_sched; seed=SEED, rng=Xoshiro)
    hists = [BinnedObject(-2L^2:4:2L^2, 0.0; boundary=ZeroBoundary()) for _ in 1:n_replicas]

    for _ in 1:(pt_therm ÷ exch_interval)
        advance!(sweep!, pt, replicas, exch_interval)
        attempt_exchange!(pt, energies(replicas))
    end
    for _ in 1:pt_measure
        for _ in 1:(measure_interval ÷ exch_interval)
            advance!(sweep!, pt, replicas, exch_interval)
            attempt_exchange!(pt, energies(replicas))
        end
        ## bin each replica's energy into the histogram of the rung it currently sits on
        for r in eachindex(replicas)
            push!(hists[ensemble_index(pt, r)], replicas[r].E[])
        end
    end
    wham(hists, kT_sched, n_iters=10_000), acceptance_rates(pt)
end

# ## Wang-Landau
#
# The same move, a different ensemble: `WangLandauAlgorithm` adapts its own log-weights until the
# energy histogram is flat, then halves the modification factor and starts over. One chain, so
# `advance!` is the plain repeat — the same verb the ladder sections use.
#
# Reference: [Sunny's Wang-Landau example](https://github.com/SunnySuite/Sunny.jl/blob/main/examples/extra/Advanced_MC/WL_ising2d.jl).

wl_iters, sweeps_per_check, max_hchecks, flatness_p = 20, 1_000, 100, 0.8

isflat(alg, lo, hi) = (h = ensemble(alg).histogram;
                       maximum(h.values) > 0 &&
                       flatness(h, lo, hi; criterion=:mean_over_min) <= 1/flatness_p)

logdos(ens) = (get_centers(ens.logweight)[ens.visited], -ens.logweight.values[ens.visited])

function run_wl()
    r = ising_replica(L)
    alg = WangLandauAlgorithm(Xoshiro(SEED), -2L^2:4:2L^2)
    for it in 1:wl_iters
        for _ in 1:max_hchecks
            advance!(sweep!, alg, r, sweeps_per_check)
            isflat(alg, -2L^2, 2L^2) && break
        end
        update_logweight!(ensemble(alg))
        it < wl_iters && reset!(alg)
    end
    logdos(ensemble(alg))
end

# ## Replica-exchange Wang-Landau
#
# Wang-Landau restricted to overlapping energy windows, one per replica, coupled by replica
# exchange. The ladder parameter is the *window*, not a temperature — each rung carries its own
# `WangLandauEnsemble` over its own energy range, which the parameter-schedule constructor builds
# directly. Because a swap moves the ensemble rather than the configuration, a window migrates to
# whichever replica slot mixes it best, so the windows must be re-sorted by their own bounds before
# the per-window curves are stitched together.
#
# Reference: [Sunny's REWL example](https://github.com/SunnySuite/Sunny.jl/blob/main/examples/extra/Advanced_MC/REWL_ising2d.jl).

n_wins, win_overlap, rewl_exch_interval = 4, 0.8, 100

function get_windows(bounds, n_wins, overlap)
    Δ = abs(bounds[2] - bounds[1]); n = 1 / (1 - overlap)
    width = Δ * n / (n_wins + n - 1)
    pos = bounds[1]
    map(1:n_wins) do _
        w = (pos, pos + width); pos += width / n; w
    end
end

## snap a per-site window to the integer energy spectrum of the L×L Ising model
window_ensemble(win) = WangLandauEnsemble(
    BinnedObject(round(Int, win[1] * L^2 / 4) * 4 : 4 : round(Int, win[2] * L^2 / 4) * 4, 0.0;
                 boundary=NegInfBoundary()))

## Metropolis quench/heat a replica into its window before Wang-Landau starts sampling it.
function seed_window!(r, alg)
    E_min, E_max = extrema(get_centers(ensemble(alg).logweight))
    if !(E_min <= r.E[] <= E_max)
        drive = MetropolisAlgorithm(alg.rng; β = r.E[] < E_min ? 0.0 : 100.0)
        for _ in 1:200
            sweep!(r, drive)
            E_min <= r.E[] <= E_max && break
        end
    end
end

function run_rewl()
    windows = get_windows((-2.0, 2.0), n_wins, win_overlap)
    rewl = ReplicaExchange([window_ensemble(w) for w in windows]; seed=SEED, rng=Xoshiro)
    replicas = [ising_replica(L) for _ in 1:n_wins]
    with_parallel(rewl) do w, alg
        seed_window!(replicas[w], alg)
    end

    for it in 1:wl_iters
        for _ in 1:max_hchecks
            for _ in 1:(sweeps_per_check ÷ rewl_exch_interval)
                advance!(sweep!, rewl, replicas, rewl_exch_interval)
                attempt_exchange!(rewl, energies(replicas))
            end
            all(w -> isflat(algorithm(rewl, w), extrema(get_centers(ensemble(algorithm(rewl, w)).logweight))...),
                1:n_wins) && break
        end
        with_parallel(rewl) do _, alg
            update_logweight!(ensemble(alg))
            it < wl_iters && reset!(alg)
        end
    end
    merge_logdos([algorithm(rewl, w) for w in 1:n_wins]), acceptance_rates(rewl)
end

## Stitch the per-window curves (sorted by their own energy bounds, then matched on the overlap)  #hide
## into one density of states. Generic post-processing, not interop — a candidate for MCX core.   #hide
function merge_logdos(algs)                                                                       #hide
    sorted = sort(algs; by = alg -> minimum(get_centers(ensemble(alg).logweight)))                #hide
    E_wins, ln_g_wins = Vector{Float64}[], Vector{Float64}[]                                      #hide
    for alg in sorted                                                                             #hide
        E, ln_g = logdos(ensemble(alg))                                                           #hide
        push!(E_wins, reverse(E ./ L^2)); push!(ln_g_wins, reverse(ln_g))                         #hide
    end                                                                                           #hide
    E, ln_g = merge_windows(E_wins, ln_g_wins)                                                    #hide
    round.(Int, E .* L^2), ln_g                                                                   #hide
end                                                                                               #hide
function merge_windows(E_wins::Vector{Vector{Float64}}, ln_g_wins::Vector{Vector{Float64}})       #hide
    E = Float64[]; ln_g = Float64[]                                                               #hide
    nearest_index(val, arr) = argmin(abs.(arr .- val))                                            #hide
    n_wins = length(E_wins)                                                                       #hide
    Em = E_wins[2][end]                                                                           #hide
    m_prev = length(E_wins[1])                                                                    #hide
    shift = 0.0                                                                                   #hide
    for w in 1:n_wins-1                                                                           #hide
        i1_hi = nearest_index(E_wins[w][1], E_wins[w+1])                                           #hide
        i2_lo = nearest_index(Em, E_wins[w])                                                       #hide
        i2_hi = nearest_index(Em, E_wins[w+1])                                                     #hide
        lo_index_of = Dict(E_wins[w][i] => i for i in 1:i2_lo)                                     #hide
        common = [(lo_index_of[E_wins[w+1][j]], j) for j in i1_hi:i2_hi if haskey(lo_index_of, E_wins[w+1][j])]  #hide
        isempty(common) && error("windows $w and $(w+1) share no visited energy in their overlap") #hide
        ln_g_lo = [ln_g_wins[w][i] for (i, _) in common]                                           #hide
        ln_g_hi = [ln_g_wins[w+1][j] for (_, j) in common]                                         #hide
        m = length(common) > 1 ? argmin(abs.(diff(ln_g_hi .- ln_g_lo))) : 1                        #hide
        mp_lo, mp_hi = common[m]                                                                   #hide
        Em = E_wins[w][mp_lo]                                                                      #hide
        pushfirst!(ln_g, (ln_g_wins[w][1+mp_lo:m_prev] .+ shift)...)                               #hide
        pushfirst!(E, E_wins[w][1+mp_lo:m_prev]...)                                                #hide
        m_prev = mp_hi                                                                             #hide
        shift += ln_g_lo[m] - ln_g_hi[m]                                                           #hide
        if w < n_wins-1                                                                            #hide
            Em = max(Em, E_wins[w+2][end])                                                         #hide
        else                                                                                       #hide
            pushfirst!(ln_g, (ln_g_wins[w+1][1:mp_hi] .+ shift)...)                                #hide
            pushfirst!(E, E_wins[w+1][1:mp_hi]...)                                                 #hide
        end                                                                                        #hide
    end                                                                                            #hide
    E, ln_g .- minimum(ln_g)                                                                       #hide
end                                                                                                #hide

# ## Beyond accept/reject: tempering Sunny's Langevin dynamics
#
# Nothing in the ladder assumes its replicas are Markov chains. Here they are advanced by Sunny's
# `Langevin` integrator — a stochastic Landau-Lifshitz equation — and **no `accept!` appears at
# all**; the MCX algorithm object is purely the carrier of the rung's ensemble. A swap moves that
# ensemble, so the integrator's temperature is re-read from it at the top of every block.
#
# The model is a cubic Heisenberg ferromagnet, where the spins are continuous and the flip proposal
# above does not apply. The unit `advance!` repeats is one integrator step.

L_heis, n_langevin_replicas = 6, 20
kT_langevin = exp.(range(log(0.5), log(3.0), length=n_langevin_replicas))
dt, damping, n_langevin, lv_therm, lv_measure = 0.025, 0.1, 50, 200, 1_000

function heisenberg_replica(L, seed, kT)
    latvecs = Sunny.lattice_vectors(1, 1, 1, 90, 90, 90)
    sys = Sunny.System(Sunny.Crystal(latvecs, [[0, 0, 0]]),
                       [1 => Sunny.Moment(s=1, g=2)], :dipole; dims=(L, L, L), seed=seed)
    Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, (1, 0, 0)))
    Sunny.randomize_spins!(sys)
    (spins = sys, dynamics = Sunny.Langevin(dt; damping, kT))
end

magnetization(sys) = sqrt(sum(abs2, sum(sys.dipoles))) / length(sys.dipoles)

## one integrator step, at the temperature of whichever rung this replica currently holds
function langevin_step!(r, alg)
    r.dynamics.kT = 1 / ensemble(alg).beta
    Sunny.step!(r.spins, r.dynamics)
end

function run_langevin_pt()
    replicas = [heisenberg_replica(L_heis, SEED + i, kT_langevin[i]) for i in 1:n_langevin_replicas]
    pt = ParallelTempering(1 ./ kT_langevin; seed=SEED, rng=Xoshiro)

    for _ in 1:lv_therm
        advance!(langevin_step!, pt, replicas, n_langevin)
        attempt_exchange!(pt, [Sunny.energy(r.spins) for r in replicas])
    end
    E, M = zeros(n_langevin_replicas), zeros(n_langevin_replicas)
    for _ in 1:lv_measure
        advance!(langevin_step!, pt, replicas, n_langevin)
        attempt_exchange!(pt, [Sunny.energy(r.spins) for r in replicas])
        for r in eachindex(replicas)
            i = ensemble_index(pt, r)
            E[i] += Sunny.energy_per_site(replicas[r].spins)
            M[i] += magnetization(replicas[r].spins)
        end
    end
    E ./ lv_measure, M ./ lv_measure, acceptance_rates(pt)
end

# ## Results
#
# All three Ising algorithms reconstruct the same density of states, checked against the exact
# Beale reference. Results are cached to TSV so the table and figures regenerate without rerunning.

datadir    = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "data")))  #hide
dos_file   = joinpath(datadir, "sunny_interop_L$(L)_dos.tsv")        #hide
lv_file    = joinpath(datadir, "sunny_interop_langevin.tsv")         #hide
meta_file  = joinpath(datadir, "sunny_interop_summary.tsv")          #hide
rerun      = "--rerun" in ARGS || "--reset" in ARGS                  #hide

## Restrict every ln-DoS to the commonly sampled energy window and anchor it at its own peak — #hide
## the bin with the best statistics in every method, so the comparison is not dominated by the  #hide
## choice of reference point.                                                                   #hide
function common_logdos(curves...)                                                              #hide
    lo = maximum(minimum(E) for (E, _) in curves)                                              #hide
    hi = minimum(maximum(E) for (E, _) in curves)                                              #hide
    map(curves) do (E, log_g)                                                                  #hide
        m = lo .<= E .<= hi                                                                    #hide
        (E[m], log_g[m] .- maximum(log_g[m]))                                                  #hide
    end                                                                                        #hide
end                                                                                            #hide
runstage(label, f) = (print(stderr, label, " ... "); flush(stderr);                            #hide
                      t = @elapsed (r = f()); println(stderr, @sprintf("%.1f s", t)); (r, t))  #hide
if rerun || !isfile(dos_file)                                                                  #hide
    ((pt_dos,  pt_p),  t_pt)   = runstage("parallel tempering        ", run_pt)                #hide
    ((wl_dos),         t_wl)   = runstage("Wang-Landau               ", run_wl)                #hide
    ((rewl_dos, re_p), t_rewl) = runstage("replica-exchange WL       ", run_rewl)              #hide
    ((E_lv, M_lv, lv_p), t_lv) = runstage("Langevin parallel tempering", run_langevin_pt)      #hide
                                                                                               #hide
    labeled = [("parallel tempering", pt_dos), ("Wang-Landau", wl_dos), ("REWL", rewl_dos)]    #hide
    vex = logdos_exact_ising2D(L; format=:vector)                                              #hide
    push!(labeled, ("exact", (first.(vex), last.(vex))))                                       #hide
    names  = first.(labeled)                                                                   #hide
    normed = common_logdos((c[2] for c in labeled)...)                                         #hide
    Egrid  = sort(unique(reduce(vcat, [E for (E, _) in normed])))                              #hide
    pos    = Dict(e => i for (i, e) in enumerate(Egrid))                                       #hide
    dosmat = fill(NaN, length(Egrid), length(names))                                           #hide
    for (j, (E, p)) in enumerate(normed), k in eachindex(E)                                    #hide
        dosmat[pos[E[k]], j] = p[k]                                                            #hide
    end                                                                                        #hide
    ex = findfirst(==("exact"), names)                                                         #hide
    devs = [maximum(abs(dosmat[i,j] - dosmat[i,ex])                                            #hide
                    for i in axes(dosmat,1) if isfinite(dosmat[i,j]) && isfinite(dosmat[i,ex]))#hide
            for j in eachindex(names)]                                                          #hide
    pstat(p) = isempty(p) ? (NaN, NaN) : (mean(filter(isfinite, p)), minimum(filter(isfinite, p)))  #hide
                                                                                               #hide
    mkpath(datadir)                                                                            #hide
    writedlm(dos_file, [permutedims(["E"; names]); hcat(Egrid, dosmat)], '\t')                 #hide
    writedlm(lv_file, [["kT" "E_per_site" "m"]; hcat(kT_langevin, E_lv, M_lv)], '\t')          #hide
    writedlm(meta_file,                                                                        #hide
        [["algorithm" "run_s" "max_dev_exact" "p_mean" "p_min"];                               #hide
         ["parallel tempering" t_pt devs[1] pstat(pt_p)...];                                   #hide
         ["Wang-Landau" t_wl devs[2] NaN NaN];                                                 #hide
         ["REWL" t_rewl devs[3] pstat(re_p)...];                                               #hide
         ["Langevin PT (Heisenberg)" t_lv NaN pstat(lv_p)...]], '\t')                          #hide
else                                                                                           #hide
    println(stderr, "loaded precomputed results from $(relpath(dos_file)) (pass --rerun to recompute)")  #src
end                                                                                            #hide

# `max |Δln g|` is the largest deviation from the exact Beale density of states over the commonly
# sampled energy range; `⟨p_exch⟩` is the mean (min) replica-exchange acceptance over the ladder
# edges.

meta = readdlm(meta_file, '\t'; header=true)[1]                                                   #hide
println("\nSunny + MonteCarloX: 2D Ising L=$L, cubic Heisenberg L=$L_heis  (threads=$(Threads.nthreads()))")  #hide
io = IOBuffer()                                                                                   #hide
println(io, "| algorithm | run [s] | max \\|Δln g\\| vs exact | ⟨p_exch⟩ (min) |")               #hide
println(io, "|---|---:|---:|---:|")                                                               #hide
for r in axes(meta, 1)                                                                            #hide
    dev = isnan(meta[r,3]) ? "—" : @sprintf("%.2f", meta[r,3])                                    #hide
    p   = isnan(meta[r,4]) ? "—" : @sprintf("%.2f (%.2f)", meta[r,4], meta[r,5])                  #hide
    println(io, @sprintf("| %s | %.1f | %s | %s |", meta[r,1], meta[r,2], dev, p))                #hide
    println(@sprintf("%-26s %7.1f s   maxdev=%-6s  p=%s", meta[r,1], meta[r,2], dev, p))          #hide
end                                                                                               #hide
Markdown.parse(String(take!(io)))                                                                 #hide

# The reconstructed log-density of states, all three Ising algorithms against the exact reference.

dosdata, doshdr = readdlm(dos_file, '\t'; header=true)                                            #hide
plt = plot(; xlabel="E", ylabel="ln g(E)", title="2D Ising, L=$L", legend=:bottom)                #hide
for j in 2:size(dosdata, 2)                                                                       #hide
    isexact = doshdr[j] == "exact"                                                                #hide
    plot!(plt, dosdata[:, 1], dosdata[:, j]; label=doshdr[j], lw=2,                               #hide
          ls=(isexact ? :dash : :solid), lc=(isexact ? :black : :auto))                           #hide
end                                                                                               #hide
savefig(plt, joinpath(@__DIR__, "sunny_interop_dos.png"))                                         #hide
plt                                                                                               #hide

# And the Langevin ladder: energy per site and magnetization across the Curie point
# (``k_BT_c \approx 1.443`` for this model).

lv, _ = readdlm(lv_file, '\t'; header=true)                                                       #hide
pE = plot(lv[:,1], lv[:,2]; xlabel="kT", ylabel="E / site", lw=2, m=:circle, ms=3, legend=false)  #hide
pM = plot(lv[:,1], lv[:,3]; xlabel="kT", ylabel="|m|", lw=2, m=:circle, ms=3, legend=false)       #hide
## combined subplots do not reserve room for the outer axis labels — set the margins explicitly  #hide
plt2 = plot(pE, pM; layout=(1,2), size=(900, 360), plot_title="Langevin PT, cubic Heisenberg",    #hide
            left_margin=6Plots.mm, bottom_margin=6Plots.mm)                                       #hide
savefig(plt2, joinpath(@__DIR__, "sunny_interop_langevin.png"))                                   #hide
plt2                                                                                              #hide
