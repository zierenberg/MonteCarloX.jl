# # Wang-Landau on the 2D Ising model
#
# Four stages, each a single `run_wl_*` function timed independently:
#
# 1. Sunny native
# 2. Sunny model + MCX Wang-Landau
# 3. full MCX
# 4. compare reconstructed density of states and runtime
#
# Reference: [Sunny's Wang-Landau example](https://github.com/SunnySuite/Sunny.jl/blob/main/examples/extra/Advanced_MC/WL_ising2d.jl)
#
# Run: `julia --project=examples/external/sunny examples/external/sunny/sunny_wl.jl`
# (append `--rerun` to overwrite the cached results instead of reloading them)

using Random, Statistics, Plots, Printf, DelimitedFiles, Markdown
using Sunny, MonteCarloX, MCXSpins
import MCXSpins: logdos_exact_ising2D

SEED = 42
L = 20
n_iters = 20
nsweeps_per_check = 1_000
max_hchecks_per_iter = 100
flatness_threshold = 0.8
nothing # hide

# ## 1. Sunny native
#
# The reference workflow: Sunny owns the system, the proposal, and the Wang-Landau loop — sweep
# until the energy histogram is flat, then halve `ln f` and repeat.

function run_wl_sunny()
    latvecs = Sunny.lattice_vectors(1, 1, 10, 90, 90, 90)
    sys = Sunny.System(Sunny.Crystal(latvecs, [[0, 0, 0]]),
                       [1 => Sunny.Moment(s=1, g=-1)], :dipole; dims=(L, L, 1))
    Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, (1, 0, 0)))
    Sunny.polarize_spins!(sys, (0, 0, 1))
    wl = Sunny.WangLandau(; sys, bin_size=1/L^2, bounds=(-2.0, 2.0), propose=Sunny.propose_flip)
    for _ in 1:n_iters
        for _ in 1:max_hchecks_per_iter
            Sunny.step_ensemble!(wl, nsweeps_per_check)
            Sunny.check_flat(wl.hist; p=flatness_threshold) && break
        end
        Sunny.reset!(wl.hist)
        wl.ln_f /= 2
    end
    round.(Int, Sunny.get_keys(wl.ln_g) .* L^2), Sunny.get_vals(wl.ln_g)  # snap Sunny's per-site keys to the integer spectrum
end

# ## 2. Sunny model + MCX algorithm
#
# We wrap a Sunny system so it satisfies the MCX system interface (`energy` + `spin_flip!`), then
# drive it with MCX's Wang-Landau. Because Wang-Landau bins on energy, the single-site step passes
# the absolute pair `accept!(alg, E_new, E_old)` rather than just `ΔE`.
# Notice this is identical to the parallel-tempering example so only has to be implemented once to give access to all algorithms in MCX.

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

function MCXSpins.spin_flip!(s::SunnyIsing, alg)
    site = rand(alg.rng, Sunny.eachsite(s.sys))
    prop = Sunny.propose_flip(s.sys, site)
    ΔE = Sunny.local_energy_change(s.sys, site, prop)
    if accept!(alg, s.energy + ΔE, s.energy)
        Sunny.setspin!(s.sys, prop, site)
        s.energy += ΔE
    end
end

# Shared MCX pieces: a local sweep, the flatness test, and the ln-g readout. Notice here the energy is not normalized per site, so the histogram is binned on the integer spectrum of the 2D Ising model.

sweep!(sys, alg, n) = (for _ in 1:n, _ in 1:L^2; spin_flip!(sys, alg); end)
isflat(alg) = flatness(ensemble(alg).histogram, -2L^2, 2L^2; criterion=:mean_over_min) <= 1/flatness_threshold

function run_wl_bridge()
    sys = SunnyIsing(L)
    alg = WangLandauAlgorithm(Xoshiro(SEED), -2L^2:4:2L^2)
    for it in 1:n_iters
        for _ in 1:max_hchecks_per_iter
            sweep!(sys, alg, nsweeps_per_check)
            isflat(alg) && break
        end
        update_logweight!(ensemble(alg))
        it < n_iters && reset!(alg)
    end
    ens = ensemble(alg)
    sampled = ens.histogram.values .> 0
    get_centers(ens.logweight)[sampled], -ens.logweight.values[sampled]
end

# ## 3. Full MCX
#
# Dropping the Sunny glue: a native `IsingSystem` exposes the same `spin_flip!`/`energy` interface,
# so the identical loop drives it unchanged.

function run_wl_mcx()
    sys = IsingSystem([L, L])
    init!(sys, :up)
    alg = WangLandauAlgorithm(Xoshiro(SEED), -2L^2:4:2L^2)
    for it in 1:n_iters
        for _ in 1:max_hchecks_per_iter
            sweep!(sys, alg, nsweeps_per_check)
            isflat(alg) && break
        end
        update_logweight!(ensemble(alg))
        it < n_iters && reset!(alg)
    end
    ens = ensemble(alg)
    sampled = ens.histogram.values .> 0
    get_centers(ens.logweight)[sampled], -ens.logweight.values[sampled]
end

# ## 4. Comparison
#
# We time each implementation, reconstruct the density of states, and cache both the timings and
# the reconstructed curves to TSV so the table and figure regenerate without rerunning the
# simulation. The exact Beale density of states is overlaid as a referee when tabulated for this `L`.

datadir     = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "data")))  # hide
dos_file    = joinpath(datadir, "sunny_wl_L$(L)_dos.tsv")     # hide
timing_file = joinpath(datadir, "sunny_wl_L$(L)_timing.tsv")  # hide
rerun       = "--rerun" in ARGS || "--reset" in ARGS  # pass --rerun to overwrite the cached results  # hide

## Restrict each ln-DoS to the common energy window and anchor it to zero at the E=0 bin (the   # hide
## peak of the density), so every curve runs negative; `Emax` optionally caps the upper energy. # hide
function common_logdos(curves...; Emax=nothing)                                                 # hide
    lo = maximum(minimum(E) for (E, _) in curves)                                               # hide
    hi = Emax === nothing ? minimum(maximum(E) for (E, _) in curves) : Emax                     # hide
    map(curves) do (E, log_g)                                                                   # hide
        m = lo .<= E .<= hi                                                                     # hide
        (E[m], log_g[m] .- log_g[m][argmin(abs.(E[m]))])                                        # hide
    end                                                                                         # hide
end                                                                                             # hide

runstage(label, f) = (print(stderr, label, " ... "); flush(stderr); r = f(); println(stderr, "done"); r)  # hide
if rerun || !isfile(dos_file)                                                       # hide
    run_wl_sunny(); run_wl_bridge(); run_wl_mcx()                                   # warmup (compile) # hide
    t_sunny  = @elapsed sunny  = runstage("Sunny native", run_wl_sunny)            # hide
    t_bridge = @elapsed bridge = runstage("MCX bridge  ", run_wl_bridge)           # hide
    t_mcx    = @elapsed mcx    = runstage("MCX native  ", run_wl_mcx)              # hide

    labeled = [("Sunny", sunny), ("MCX bridge", bridge), ("MCX native", mcx)]      # hide
    if isfile(joinpath(pkgdir(MCXSpins), "data", "exact_solutions", "ising2D_$(L)x$(L).csv"))  # hide
        vex = logdos_exact_ising2D(L; format=:vector)                                           # hide
        push!(labeled, ("exact", (first.(vex), last.(vex))))                                    # hide
    end                                                                                         # hide

    names  = first.(labeled)                                                        # hide
    normed = common_logdos((c[2] for c in labeled)...)                              # hide
    Egrid  = sort(unique(reduce(vcat, [E for (E, _) in normed])))                   # hide
    pos    = Dict(e => i for (i, e) in enumerate(Egrid))                            # hide
    dosmat = fill(NaN, length(Egrid), length(names))                               # hide
    for (j, (E, p)) in enumerate(normed), k in eachindex(E)                         # hide
        dosmat[pos[E[k]], j] = p[k]                                                 # hide
    end                                                                            # hide

    timemat = ["Sunny native" t_sunny  1.0                                         # hide
               "MCX bridge"   t_bridge t_sunny / t_bridge                          # hide
               "MCX native"   t_mcx    t_sunny / t_mcx]                            # hide

    mkpath(datadir)                                                                          # hide
    writedlm(dos_file,    [permutedims(["E"; names]); hcat(Egrid, dosmat)], '\t')            # hide
    writedlm(timing_file, [["implementation" "time" "speedup"]; timemat], '\t')             # hide
else                                                                                # hide
    println(stderr, "loaded precomputed results from $(relpath(dos_file)) (pass --rerun to recompute)")  #src
end                                                                                 # hide

# The three implementations, timed side by side; `speedup` is relative to the Sunny-native run.

timedata = readdlm(timing_file, '\t'; header=true)[1]                                             # hide
## echo setup + timings to the terminal (the Markdown table below is for the rendered docs)       # hide
println("\nWang-Landau: 2D Ising, L=$L")                                                          # hide
println(@sprintf("n_iters=%d  nsweeps_per_check=%d  max_hchecks_per_iter=%d  flatness=%.2f",      # hide
                 n_iters, nsweeps_per_check, max_hchecks_per_iter, flatness_threshold))           # hide
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

# The reconstructed log-density of states against the exact Beale reference — anchored to zero at
# the E=0 bin (the peak), so the curves run negative and any deviation is easy to spot.

dosdata, doshdr = readdlm(dos_file, '\t'; header=true)                                          # hide
E = dosdata[:, 1]                                                                               # hide
plt = plot(; xlabel="E", ylabel="ln g(E)", title="2D Ising WL, L=$L")                           # hide
for j in 2:size(dosdata, 2)                                                                     # hide
    plot!(plt, E, dosdata[:, j]; label=doshdr[j], lw=2,                                         # hide
          ls=(doshdr[j] == "exact" ? :dash : :solid), lc=(doshdr[j] == "exact" ? :black : :auto))  # hide
end                                                                                             # hide
savefig(plt, joinpath(@__DIR__, "sunny_wl_dos.png"))                                            # hide
plt                                                                                             # hide
