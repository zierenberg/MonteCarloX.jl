# # Replica-exchange Wang-Landau on the 2D Ising model
#
# Two stages, each a single `run_rewl_*` function timed independently:
#
# 1. Sunny native
# 2. full MCX
# 3. compare reconstructed density of states and runtime
#
# Reference: [Sunny's REWL example](https://github.com/SunnySuite/Sunny.jl/blob/main/examples/extra/Advanced_MC/REWL_ising2d.jl)
#
# Run: `julia --project=examples/external/sunny -t auto examples/external/sunny/sunny_rewl.jl`
# (append `--rerun` to overwrite the cached results instead of reloading them)

using Random, Statistics, Plots, Printf, DelimitedFiles, Markdown
using Sunny, MonteCarloX, MCXSpins
import MCXSpins: logdos_exact_ising2D

SEED = 42
L = 20
n_wins = 4                 # energy windows / replicas
win_overlap = 0.8
n_iters = 20               # Wang-Landau iterations (matches sunny_wl.jl's refinement)
max_hchecks_per_iter = 100 # max flatness checks per iteration
hcheck_interval = 1000     # sweeps between flatness checks
exch_interval = 100        # sweeps between exchange attempts
flatness_p = 0.8
nothing # hide

# ## 1. Sunny native
#
# The reference workflow: Sunny owns the windowed systems, the proposals, the exchanges, and the
# Wang-Landau loop, then merges the per-window densities of state.

function run_rewl_sunny()
    latvecs = Sunny.lattice_vectors(1, 1, 10, 90, 90, 90)
    sys = Sunny.System(Sunny.Crystal(latvecs, [[0, 0, 0]]),
                       [1 => Sunny.Moment(s=1, g=-1)], :dipole; dims=(L, L, 1))
    Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, (1, 0, 0)))
    Sunny.polarize_spins!(sys, (0, 0, 1))
    windows = Sunny.get_windows((-2.0, 2.0), n_wins, win_overlap)
    REWL = Sunny.ParallelWangLandau(; sys, bin_size=1/L^2, propose=Sunny.propose_flip, windows)
    for _ in 1:n_iters
        for _ in 1:max_hchecks_per_iter
            Sunny.step_ensemble!(REWL, hcheck_interval, exch_interval)
            flat = fill(false, length(REWL.samplers))
            Threads.@threads for j in eachindex(REWL.samplers)
                flat[j] = Sunny.check_flat(REWL.samplers[j].hist; p=flatness_p)
            end
            all(flat) && break
        end
        Threads.@threads for sampler in REWL.samplers
            Sunny.reset!(sampler.hist)
            sampler.ln_f /= 2
        end
    end
    E_wins = [Sunny.get_keys(wl.ln_g) for wl in REWL.samplers]
    ln_g_wins = [Sunny.get_vals(wl.ln_g) for wl in REWL.samplers]
    E, ln_g = Sunny.merge(E_wins, ln_g_wins)
    round.(Int, E .* L^2), ln_g          # back to the integer total-energy spectrum
end

# ## 2. Full MCX
#
# Each replica owns one energy window; `get_windows` lays them out with the requested overlap,
# `window_ensemble` snaps a window to the integer energy grid, `seed_window!` drives a fresh
# system into its window, and `rewl!`/`merge_logdos` run the loop and stitch the per-window
# curves. The native `IsingSystem` exposes the `spin_flip!`/`energy` interface that all of this
# is built on.
#
#Note: this should serve as a basis for MCX core addons that enable a more simple realization of REWL with the MCX API (combining RE and WL)

function get_windows(bounds::Tuple{Float64,Float64}, n_wins::Int, overlap::Float64)
    Δ = abs(bounds[2] - bounds[1]); n = 1 / (1 - overlap)
    width = Δ * n / (n_wins + n - 1)
    wins = Tuple{Float64,Float64}[]
    pos = bounds[1]
    for _ in 1:n_wins
        push!(wins, (pos, pos + width))
        pos += width / n
    end
    wins
end

function window_ensemble(win)
    lo = round(Int, win[1] * L^2 / 4) * 4        # snap window bounds to the integer energy spectrum
    hi = round(Int, win[2] * L^2 / 4) * 4
    WangLandauEnsemble(BinnedObject(lo:4:hi, 0.0; boundary=NegInfBoundary()))  # discrete integer energies
end

# Metropolis quench/heat a system into [E_min, E_max] before Wang-Landau starts sampling it.
# (Note: this is okay for the example but will likely fail for more complex systems with rugged energy landscapes)
function seed_window!(sys, alg, energies, r)
    E_min, E_max = extrema(get_centers(ensemble(alg).logweight))
    E0 = Float64(MCXSpins.energy(sys))
    if !(E_min <= E0 <= E_max)
        drive = MetropolisAlgorithm(alg.rng; β = E0 < E_min ? 0.0 : 100.0)
        for _ in 1:200
            for _ in 1:L^2; spin_flip!(sys, drive); end
            E_min <= Float64(MCXSpins.energy(sys)) <= E_max && break
        end
    end
    energies[r] = Float64(MCXSpins.energy(sys))
end

# Merge the visited per-window curves (descending energy) into one density of states.
#
# Replica exchange swaps *ensembles* between `algs[i]`/`algs[j]` on acceptance — that's how a
# window migrates to a better-mixing replica slot. So after `rewl!`, `algs[r]` no longer
# necessarily holds window `r`; the windows must be re-sorted by their own energy bounds before
# stitching, or adjacent-window merging silently operates on the wrong pairs and truncates the
# reconstructed curve.
function merge_logdos(algs)
    sorted = sort(algs; by = alg -> minimum(get_centers(ensemble(alg).logweight)))
    E_wins, ln_g_wins = Vector{Float64}[], Vector{Float64}[]
    for alg in sorted
        ens = ensemble(alg)
        push!(E_wins, reverse(get_centers(ens.logweight)[ens.visited] ./ L^2))
        push!(ln_g_wins, reverse(-ens.logweight.values[ens.visited]))
    end
    E, ln_g = merge_windows(E_wins, ln_g_wins)
    round.(Int, E .* L^2), ln_g          # back to the integer total-energy spectrum
end

function merge_windows(E_wins::Vector{Vector{Float64}}, ln_g_wins::Vector{Vector{Float64}})
    E = Float64[]; ln_g = Float64[]
    nearest_index(val, arr) = argmin(abs.(arr .- val))
    n_wins = length(E_wins)
    Em = E_wins[2][end]
    m_prev = length(E_wins[1])
    shift = 0.0
    for w in 1:n_wins-1
        i1_lo = 1
        i1_hi = nearest_index(E_wins[w][1], E_wins[w+1])
        i2_lo = nearest_index(Em, E_wins[w])
        i2_hi = nearest_index(Em, E_wins[w+1])
        # Match by energy value, not raw index: unvisited bins leave gaps, so the two windows
        # can have a different number of sampled points over the same overlap range.
        lo_index_of = Dict(E_wins[w][i] => i for i in i1_lo:i2_lo)
        common = [(lo_index_of[E_wins[w+1][j]], j) for j in i1_hi:i2_hi if haskey(lo_index_of, E_wins[w+1][j])]
        isempty(common) && error("windows $w and $(w+1) share no visited energy in their overlap")
        ln_g_lo = [ln_g_wins[w][i] for (i, _) in common]
        ln_g_hi = [ln_g_wins[w+1][j] for (_, j) in common]
        m = length(common) > 1 ? argmin(abs.(diff(ln_g_hi .- ln_g_lo))) : 1
        mp_lo, mp_hi = common[m]
        Em = E_wins[w][mp_lo]
        pushfirst!(ln_g, (ln_g_wins[w][1+mp_lo:m_prev] .+ shift)...)
        pushfirst!(E, E_wins[w][1+mp_lo:m_prev]...)
        m_prev = mp_hi
        shift += ln_g_lo[m] - ln_g_hi[m]
        if w < n_wins-1
            Em = max(Em, E_wins[w+2][end])
        else
            pushfirst!(ln_g, (ln_g_wins[w+1][1:mp_hi] .+ shift)...)
            pushfirst!(E, E_wins[w+1][1:mp_hi]...)
        end
    end
    E, ln_g .- minimum(ln_g)
end

function run_rewl_mcx()
    windows = get_windows((-2.0, 2.0), n_wins, win_overlap)
    algs = [MetropolisHastingsAlgorithm(Xoshiro(SEED + r), window_ensemble(windows[r])) for r in 1:n_wins]
    rewl = ReplicaExchange(ThreadsBackend(n_wins), algs)
    systems = [IsingSystem([L, L]) for _ in 1:n_wins]
    
    energies = zeros(Float64, n_wins)
    Threads.@threads for r in 1:n_wins
        init!(systems[r], :random; rng=algs[r].rng)
        seed_window!(systems[r], algs[r], energies, r)
    end
        n = length(systems)
    for i in 1:n_iters
        for _ in 1:max_hchecks_per_iter
            for step in 1:hcheck_interval
                Threads.@threads for r in 1:n
                    for _ in 1:L^2; spin_flip!(systems[r], algs[r]); end
                    energies[r] = Float64(MCXSpins.energy(systems[r]))
                end
                step % exch_interval == 0 && MonteCarloX.update!(rewl, energies)
            end
            flat = fill(false, n)
            Threads.@threads for r in 1:n
                h = ensemble(algs[r]).histogram
                flat[r] = maximum(h.values) > 0 &&
                          flatness(h, extrema(get_centers(h))...; criterion=:mean_over_min) <= 1/flatness_p
            end
            all(flat) && break
        end
        Threads.@threads for r in 1:n
            update_logweight!(ensemble(algs[r]))
            i < n_iters && reset!(algs[r])
        end
    end

    merge_logdos(algs)
end

# ## 3. Comparison
#
# We time each implementation, merge the per-window density of states, and cache both the timings
# and the reconstructed curves to TSV so the table and figure regenerate without rerunning the
# simulation. The exact Beale density of states is overlaid as a referee when tabulated for this `L`.

datadir     = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "data")))  # hide
dos_file    = joinpath(datadir, "sunny_rewl_L$(L)_dos.tsv")     # hide
timing_file = joinpath(datadir, "sunny_rewl_L$(L)_timing.tsv")  # hide
rerun       = "--rerun" in ARGS || "--reset" in ARGS  # pass --rerun to overwrite the cached results  # hide

## Restrict each ln-DoS to the common energy window and anchor it to zero at the lowest         # hide
## sampled energy (the Beale convention); `Emax` optionally caps the upper energy.              # hide
function common_logdos(curves...; Emax=nothing)                                                 # hide
    lo = maximum(minimum(E) for (E, _) in curves)                                               # hide
    hi = Emax === nothing ? minimum(maximum(E) for (E, _) in curves) : Emax                     # hide
    map(curves) do (E, log_g)                                                                   # hide
        m = lo .<= E .<= hi                                                                     # hide
        (E[m], log_g[m] .- log_g[m][argmin(E[m])])                                              # hide
    end                                                                                         # hide
end                                                                                             # hide

runstage(label, f) = (print(stderr, label, " ... "); flush(stderr); r = f(); println(stderr, "done"); r)  # hide
if rerun || !isfile(dos_file)                                                       # hide
    run_rewl_sunny(); run_rewl_mcx()                                               # warmup (compile) # hide
    t_sunny  = @elapsed sunny  = runstage("Sunny native", run_rewl_sunny)         # hide
    t_mcx    = @elapsed mcx    = runstage("MCX native  ", run_rewl_mcx)           # hide

    labeled = [("Sunny", sunny), ("MCX native", mcx)]                             # hide
    if isfile(joinpath(pkgdir(MCXSpins), "data", "exact_solutions", "ising2D_$(L)x$(L).csv"))  # hide
        vex = logdos_exact_ising2D(L; format=:vector)                                           # hide
        push!(labeled, ("exact", (first.(vex), last.(vex))))                                    # hide
    end                                                                                         # hide

    names  = first.(labeled)                                                       # hide
    normed = common_logdos((c[2] for c in labeled)...)                             # hide
    Egrid  = sort(unique(reduce(vcat, [E for (E, _) in normed])))                  # hide
    pos    = Dict(e => i for (i, e) in enumerate(Egrid))                           # hide
    dosmat = fill(NaN, length(Egrid), length(names))                              # hide
    for (j, (E, p)) in enumerate(normed), k in eachindex(E)                        # hide
        dosmat[pos[E[k]], j] = p[k]                                                # hide
    end                                                                           # hide

    timemat = ["Sunny native" t_sunny  1.0                                        # hide
               "MCX native"   t_mcx    t_sunny / t_mcx]                           # hide

    mkpath(datadir)                                                                          # hide
    writedlm(dos_file,    [permutedims(["E"; names]); hcat(Egrid, dosmat)], '\t')            # hide
    writedlm(timing_file, [["implementation" "time" "speedup"]; timemat], '\t')             # hide
else                                                                                # hide
    println(stderr, "loaded precomputed results from $(relpath(dos_file)) (pass --rerun to recompute)")  #src
end                                                                                 # hide

# The two implementations, timed side by side; `speedup` is relative to the Sunny-native run.

timedata = readdlm(timing_file, '\t'; header=true)[1]                                             # hide
## echo setup + timings to the terminal (the Markdown table below is for the rendered docs)       # hide
println("\nReplica-Exchange Wang-Landau: 2D Ising, L=$L, n_wins=$n_wins  (threads=$(Threads.nthreads()))")  # hide
println(@sprintf("overlap=%.2f  n_iters=%d  hcheck_interval=%d  exch_interval=%d  flatness=%.2f", # hide
                 win_overlap, n_iters, hcheck_interval, exch_interval, flatness_p))               # hide
println(@sprintf("%-14s %10s %9s", "implementation", "time [s]", "speedup"))                      # hide
for r in axes(timedata, 1)                                                                        # hide
    println(@sprintf("%-14s %10.3f %8.2f×", timedata[r, 1], timedata[r, 2], timedata[r, 3]))      # hide
end                                                                                               # hide
io = IOBuffer()                                                                                    # hide
println(io, "| implementation | time [s] | speedup |")                                            # hide
println(io, "|---|---:|---:|")                                                                     # hide
for r in axes(timedata, 1)                                                                         # hide
    println(io, @sprintf("| %s | %.3f | %.2f× |", timedata[r, 1], timedata[r, 2], timedata[r, 3]))  # hide
end                                                                                                # hide
Markdown.parse(String(take!(io)))                                                                 # hide

# The merged log-density of states against the exact Beale reference — anchored to zero at the
# ground-state energy, they overlap where each curve has support.

dosdata, doshdr = readdlm(dos_file, '\t'; header=true)                                          # hide
E = dosdata[:, 1]                                                                               # hide
plt = plot(; xlabel="E", ylabel="ln g(E)", title="2D Ising REWL, L=$L")                         # hide
for j in 2:size(dosdata, 2)                                                                     # hide
    plot!(plt, E, dosdata[:, j]; label=doshdr[j], lw=2,                                         # hide
          ls=(doshdr[j] == "exact" ? :dash : :solid), lc=(doshdr[j] == "exact" ? :black : :auto))  # hide
end                                                                                             # hide
savefig(plt, joinpath(@__DIR__, "sunny_rewl_dos.png"))                                          # hide
plt                                                                                             # hide
