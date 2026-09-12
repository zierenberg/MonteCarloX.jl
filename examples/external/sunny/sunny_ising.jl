# # Sunny interop tutorial: 2D Ising in four stages
#
# Three stages, each a single `run_ising_*` function that times thermalization and production
# independently per temperature, plus a comparison stage:
#
# 1. Sunny native
# 2. Sunny model + MCX algorithm
# 3. full MCX
# 4. compare physics and timing
#
# The point is not to hide the required glue code; it is to document the exact steps needed to
# connect an external model to MonteCarloX algorithms.
#
# Reference: [Sunny's Ising model tutorial](https://github.com/SunnySuite/Sunny.jl/blob/main/examples/05_MC_Ising.jl)
#
# Run: `julia --project=examples/external/sunny examples/external/sunny/sunny_ising.jl`
# (append `--rerun` to overwrite the cached results instead of reloading them)

using Random, Statistics, Plots, Printf, DelimitedFiles, Markdown
using StatsBase: weights, mean
using Sunny, MonteCarloX, MCXSpins
import MCXSpins: logdos_exact_ising2D, IsingSystem, energy, spin_flip!
import MonteCarloX: reweight, get_centers

SEED = 42
L, therm, prod = 20, 20_000, 500_000
Tc = 2 / log(1 + sqrt(2.0))
Ts = [1.8, Tc, 3.0]
nothing # hide

logdos = logdos_exact_ising2D(L)
Egrid = get_centers(logdos)
e_exact = [mean(Egrid, weights(reweight(logdos, -Egrid ./ T))) / L^2 for T in Ts]

# ## 1. Sunny native
#
# The reference workflow: the system and the update rule live entirely inside Sunny.

function run_ising_sunny(Ts, therm, prod)
    latvecs = Sunny.lattice_vectors(1, 1, 10, 90, 90, 90)
    sys = Sunny.System(Sunny.Crystal(latvecs, [[0, 0, 0]]), [1 => Sunny.Moment(s=1, g=-1)], :dipole; dims=(L, L, 1))
    Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, (1, 0, 0)))
    es, ts = Float64[], Float64[]
    for (i, T) in enumerate(Ts)
        Sunny.polarize_spins!(sys, (0, 0, 1))
        rng_init = Xoshiro(SEED + i)
        for site in Sunny.eachsite(sys)
            rand(rng_init, Bool) && Sunny.setspin!(sys, Sunny.propose_flip(sys, site), site)
        end
        copy!(sys.rng, Xoshiro(SEED + 100 + i))
        sampler = Sunny.LocalSampler(kT=T, nsweeps=1.0, propose=Sunny.propose_flip)
        for _ in 1:therm; Sunny.step!(sys, sampler); end
        t = @elapsed begin
            e = 0.0
            for _ in 1:prod
                Sunny.step!(sys, sampler)
                e += Sunny.energy_per_site(sys)
            end
            e /= prod
        end
        push!(es, e); push!(ts, t)
    end
    es, ts
end

# ## 2. Sunny model + MCX algorithm
#
# Since Sunny does not save the energy inside the model, we wrap it in a small struct and take
# the opportunity to overload `MCXSpins.energy`/`MCXSpins.spin_flip!` so a Sunny system satisfies
# the MCX model interface. Sections 2 and 3 then share the same `sweep!`.

mutable struct SunnyIsing{S}
    sys::S
    energy::Float64
end
function SunnyIsing(L::Int)
    latvecs = Sunny.lattice_vectors(1, 1, 10, 90, 90, 90)
    sys = Sunny.System(Sunny.Crystal(latvecs, [[0, 0, 0]]), [1 => Sunny.Moment(s=1, g=-1)], :dipole; dims=(L, L, 1))
    Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, (1, 0, 0)))
    SunnyIsing(sys, 0.0)
end
MCXSpins.energy(s::SunnyIsing) = s.energy

function MCXSpins.spin_flip!(s::SunnyIsing, alg)
    site = rand(alg.rng, Sunny.eachsite(s.sys))
    prop = Sunny.propose_flip(s.sys, site)
    ΔE = Sunny.local_energy_change(s.sys, site, prop)
    if accept!(alg, ΔE)
        Sunny.setspin!(s.sys, prop, site)
        s.energy += ΔE
    end
end

sweep!(sys, alg, n) = (for _ in 1:n; spin_flip!(sys, alg); end)

function run_ising_bridge(Ts, therm, prod)
    es, ts = Float64[], Float64[]
    for (i, T) in enumerate(Ts)
        s = SunnyIsing(L)
        Sunny.polarize_spins!(s.sys, (0, 0, 1))
        rng_init = Xoshiro(SEED + i)
        for site in Sunny.eachsite(s.sys)
            rand(rng_init, Bool) && Sunny.setspin!(s.sys, Sunny.propose_flip(s.sys, site), site)
        end
        s.energy = Sunny.energy_per_site(s.sys) * L^2
        alg = MetropolisAlgorithm(Xoshiro(SEED + 200 + i); β=1/T)
        for _ in 1:therm; sweep!(s, alg, L^2); end
        reset!(alg)
        t = @elapsed begin
            e = 0.0
            for _ in 1:prod
                sweep!(s, alg, L^2)
                e += energy(s) / L^2
            end
            e /= prod
        end
        push!(es, e); push!(ts, t)
    end
    es, ts
end

# ## 3. Full MCX
#
# Dropping the Sunny glue: the native `IsingSystem` exposes the same `spin_flip!`/`energy`
# interface, so the identical `sweep!` drives it unchanged.

function run_ising_mcx(Ts, therm, prod)
    es, ts = Float64[], Float64[]
    for (i, T) in enumerate(Ts)
        sys = IsingSystem([L, L])
        init!(sys, :random, rng=MersenneTwister(SEED + i))
        alg = MetropolisAlgorithm(Xoshiro(SEED + 100 + i); β=1/T)
        for _ in 1:therm; sweep!(sys, alg, L^2); end
        reset!(alg)
        t = @elapsed begin
            e = 0.0
            for _ in 1:prod
                sweep!(sys, alg, L^2)
                e += energy(sys) / L^2
            end
            e /= prod
        end
        push!(es, e); push!(ts, t)
    end
    es, ts
end

# ## 4. Comparison
#
# We time each implementation per temperature, then cache both the timings and the measured
# energies to TSV so the table and figure regenerate without rerunning the simulation.

datadir     = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "data")))  # hide
energy_file = joinpath(datadir, "sunny_ising_L$(L)_energy.tsv")   # hide
timing_file = joinpath(datadir, "sunny_ising_L$(L)_timing.tsv")   # hide
rerun       = "--rerun" in ARGS || "--reset" in ARGS  # pass --rerun to overwrite the cached results  # hide

runstage(label, f) = (print(stderr, label, " ... "); flush(stderr); r = f(); println(stderr, "done"); r)  # hide
if rerun || !isfile(energy_file)                                            # hide
    run_ising_sunny([Tc], 2, 2); run_ising_bridge([Tc], 2, 2); run_ising_mcx([Tc], 2, 2)  # warmup (compile) # hide
    e_s,  t_s  = runstage("Sunny native", () -> run_ising_sunny(Ts, therm, prod))   # hide
    e_bs, t_bs = runstage("MCX bridge  ", () -> run_ising_bridge(Ts, therm, prod))  # hide
    e_n,  t_n  = runstage("MCX native  ", () -> run_ising_mcx(Ts, therm, prod))     # hide

    mkpath(datadir)                                                                                # hide
    energymat = hcat(Ts, e_exact, e_s, e_bs, e_n)                                                  # hide
    timingmat = hcat(Ts, t_s, t_bs, t_n)                                                           # hide
    writedlm(energy_file, [["T" "e_exact" "e_sunny" "e_bridge" "e_native"]; energymat], '\t')      # hide
    writedlm(timing_file, [["T" "t_sunny" "t_bridge" "t_native"]; timingmat], '\t')                # hide
else                                                                         # hide
    println(stderr, "loaded precomputed results from $(relpath(energy_file)) (pass --rerun to recompute)")  #src
end                                                                          # hide

# The three implementations, timed side by side per temperature.

energydata = readdlm(energy_file, '\t'; header=true)[1]                                          # hide
timingdata = readdlm(timing_file, '\t'; header=true)[1]                                          # hide
## echo setup + results to the terminal (the Markdown table below is for the rendered docs)      # hide
println("\nSunny interop: 2D Ising, L=$L")                                                       # hide
println(@sprintf("therm=%d  prod=%d", therm, prod))                                              # hide
println(@sprintf("%-7s %12s %12s %12s %12s  %10s %11s %11s", "T", "e_exact", "e_sunny", "e_bridge", "e_native", "t_sunny(s)", "t_bridge(s)", "t_native(s)"))  # hide
for r in axes(energydata, 1)                                                                     # hide
    println(@sprintf("%-7.3f %12.6f %12.6f %12.6f %12.6f  %10.4f %11.4f %11.4f",                 # hide
        energydata[r, 1], energydata[r, 2], energydata[r, 3], energydata[r, 4], energydata[r, 5], # hide
        timingdata[r, 2], timingdata[r, 3], timingdata[r, 4]))                                   # hide
end                                                                                               # hide
io = IOBuffer()                                                                                   # hide
println(io, "| T | e_exact | e_sunny | e_bridge | e_native | t_sunny [s] | t_bridge [s] | t_native [s] |")  # hide
println(io, "|---:|---:|---:|---:|---:|---:|---:|---:|")                                          # hide
for r in axes(energydata, 1)                                                                      # hide
    println(io, @sprintf("| %.3f | %.6f | %.6f | %.6f | %.6f | %.4f | %.4f | %.4f |",             # hide
        energydata[r, 1], energydata[r, 2], energydata[r, 3], energydata[r, 4], energydata[r, 5], # hide
        timingdata[r, 2], timingdata[r, 3], timingdata[r, 4]))                                   # hide
end                                                                                                # hide
Markdown.parse(String(take!(io)))                                                                 # hide

# The energy per site against the exact (Beale) reference, all three implementations
# overlapping; the reference is drawn as a continuous curve reweighted from the exact
# density of states on a fine temperature grid, not just at the three sampled points.

Tgrid = range(1.2, 3.8, length=300)                                                          # hide
e_exact_curve = [mean(Egrid, weights(reweight(logdos, -Egrid ./ T))) / L^2 for T in Tgrid]    # hide

plt = plot(Tgrid, e_exact_curve; lw=2, ls=:dot, color=:black, label="exact",                 # hide
           xlabel="T", ylabel="e per site", title="2D Ising, L=$L")                          # hide
scatter!(plt, energydata[:, 1], energydata[:, 3]; marker=:o, label="Sunny")                  # hide
scatter!(plt, energydata[:, 1], energydata[:, 4]; marker=:d, label="MCX bridge")             # hide
scatter!(plt, energydata[:, 1], energydata[:, 5]; marker=:s, label="MCX native")             # hide
savefig(plt, joinpath(@__DIR__, "sunny_ising_energy.png"))                                   # hide
plt                                                                                           # hide
