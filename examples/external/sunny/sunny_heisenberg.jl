# # Sunny interop tutorial: Heisenberg model in four stages
#
# Three stages, each a single `run_heisenberg_*` function that times thermalization and
# production independently per temperature, plus a comparison stage:
#
# 1. Sunny native
# 2. Sunny model + MCX algorithm
# 3. full MCX
# 4. compare physics and timing
#
# Run: `julia --project=examples/external/sunny examples/external/sunny/sunny_heisenberg.jl`
# (append `--rerun` to overwrite the cached results instead of reloading them)

using Random, Statistics, Plots, Printf, DelimitedFiles, Markdown
using Sunny, MonteCarloX, MCXSpins
import MCXSpins: HeisenbergSystem, energy, magnetization, spin_flip!

SEED = 42
L, therm, prod = 20, 5000, 20_000
Ts = [0.5, 1.0, 1.443, 3.0]
nothing # hide

# ## 1. Sunny native
#
# The reference workflow: the system and the update rule live entirely inside Sunny.

function run_heisenberg_sunny(Ts, therm, prod)
    latvecs = Sunny.lattice_vectors(1, 1, 1, 90, 90, 90)
    sys = Sunny.System(Sunny.Crystal(latvecs, [[0, 0, 0]]), [1 => Sunny.Moment(s=1, g=2)], :dipole; dims=(L, L, L))
    Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, (1, 0, 0)))
    es, ms, ts = Float64[], Float64[], Float64[]
    for (i, T) in enumerate(Ts)
        Sunny.randomize_spins!(sys)
        copy!(sys.rng, Xoshiro(SEED + 100 + i))
        sampler = Sunny.LocalSampler(kT=T, nsweeps=1.0, propose=Sunny.propose_uniform)
        for _ in 1:therm; Sunny.step!(sys, sampler); end
        t = @elapsed begin
            e = m = 0.0
            for _ in 1:prod
                Sunny.step!(sys, sampler)
                e += Sunny.energy_per_site(sys)
                mv = sum(sys.dipoles)
                m += sqrt(sum(abs2, mv)) / length(sys.dipoles)
            end
            e /= prod; m /= prod
        end
        push!(es, e); push!(ms, m); push!(ts, t)
    end
    es, ms, ts
end

# ## 2. Sunny model + MCX algorithm
#
# Since Sunny does not save the energy inside the model, we wrap it in a small struct and take
# the opportunity to overload `MCXSpins.energy`/`MCXSpins.spin_flip!` so a Sunny system satisfies
# the MCX model interface. Sections 2 and 3 then share the same `sweep!`.

mutable struct SunnyHeisenberg{S}
    sys::S
    energy::Float64
end
function SunnyHeisenberg(L::Int)
    latvecs = Sunny.lattice_vectors(1, 1, 1, 90, 90, 90)
    sys = Sunny.System(Sunny.Crystal(latvecs, [[0, 0, 0]]), [1 => Sunny.Moment(s=1, g=2)], :dipole; dims=(L, L, L))
    Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, (1, 0, 0)))
    SunnyHeisenberg(sys, 0.0)
end
MCXSpins.energy(s::SunnyHeisenberg) = s.energy

function MCXSpins.spin_flip!(s::SunnyHeisenberg, alg)
    site = rand(alg.rng, Sunny.eachsite(s.sys))
    prop = Sunny.propose_uniform(s.sys, site)
    ΔE = Sunny.local_energy_change(s.sys, site, prop)
    if accept!(alg, ΔE)
        Sunny.setspin!(s.sys, prop, site)
        s.energy += ΔE
    end
end

sweep!(sys, alg, n) = (for _ in 1:n; spin_flip!(sys, alg); end)

function run_heisenberg_bridge(Ts, therm, prod)
    N = L^3
    es, ms, ts = Float64[], Float64[], Float64[]
    for (i, T) in enumerate(Ts)
        s = SunnyHeisenberg(L)
        Sunny.randomize_spins!(s.sys)
        s.energy = Sunny.energy_per_site(s.sys) * N
        copy!(s.sys.rng, Xoshiro(SEED + 100 + i))
        alg = MetropolisAlgorithm(Xoshiro(SEED + 200 + i); β=1/T)
        for _ in 1:therm; sweep!(s, alg, N); end
        reset!(alg)
        t = @elapsed begin
            e = m = 0.0
            for _ in 1:prod
                sweep!(s, alg, N)
                e += energy(s) / N
                mv = sum(s.sys.dipoles)
                m += sqrt(sum(abs2, mv)) / N
            end
            e /= prod; m /= prod
        end
        push!(es, e); push!(ms, m); push!(ts, t)
    end
    es, ms, ts
end

# ## 3. Full MCX
#
# Dropping the Sunny glue: the native `HeisenbergSystem` exposes the same `spin_flip!`/`energy`
# interface, so the identical `sweep!` drives it unchanged.

function run_heisenberg_mcx(Ts, therm, prod)
    N = L^3
    es, ms, ts = Float64[], Float64[], Float64[]
    for (i, T) in enumerate(Ts)
        sys = HeisenbergSystem([L, L, L])
        init!(sys, :random, rng=MersenneTwister(SEED + i))
        alg = MetropolisAlgorithm(Xoshiro(SEED + 100 + i); β=1/T)
        for _ in 1:therm; sweep!(sys, alg, N); end
        reset!(alg)
        t = @elapsed begin
            e = m = 0.0
            for _ in 1:prod
                sweep!(sys, alg, N)
                e += energy(sys) / N
                m += sqrt(sum(abs2, magnetization(sys))) / N
            end
            e /= prod; m /= prod
        end
        push!(es, e); push!(ms, m); push!(ts, t)
    end
    es, ms, ts
end

# ## 4. Comparison
#
# We time each implementation per temperature, then cache both the timings and the measured
# energy/magnetization to TSV so the tables and figure regenerate without rerunning the simulation.

datadir     = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "data")))  # hide
obs_file    = joinpath(datadir, "sunny_heisenberg_L$(L)_observables.tsv")  # hide
timing_file = joinpath(datadir, "sunny_heisenberg_L$(L)_timing.tsv")      # hide
rerun       = "--rerun" in ARGS || "--reset" in ARGS  # pass --rerun to overwrite the cached results  # hide

runstage(label, f) = (print(stderr, label, " ... "); flush(stderr); r = f(); println(stderr, "done"); r)  # hide
if rerun || !isfile(obs_file)                                                    # hide
    run_heisenberg_sunny([Ts[1]], 2, 2); run_heisenberg_bridge([Ts[1]], 2, 2); run_heisenberg_mcx([Ts[1]], 2, 2)  # warmup (compile) # hide
    e_s,  m_s,  t_s  = runstage("Sunny native", () -> run_heisenberg_sunny(Ts, therm, prod))   # hide
    e_bs, m_bs, t_bs = runstage("MCX bridge  ", () -> run_heisenberg_bridge(Ts, therm, prod))  # hide
    e_n,  m_n,  t_n  = runstage("MCX native  ", () -> run_heisenberg_mcx(Ts, therm, prod))     # hide

    mkpath(datadir)                                                                                      # hide
    obsmat = hcat(Ts, e_s, e_bs, e_n, m_s, m_bs, m_n)                                                    # hide
    timingmat = hcat(Ts, t_s, t_bs, t_n)                                                                 # hide
    writedlm(obs_file,    [["T" "e_sunny" "e_bridge" "e_native" "m_sunny" "m_bridge" "m_native"]; obsmat], '\t')  # hide
    writedlm(timing_file, [["T" "t_sunny" "t_bridge" "t_native"]; timingmat], '\t')                      # hide
else                                                                         # hide
    println(stderr, "loaded precomputed results from $(relpath(obs_file)) (pass --rerun to recompute)")  #src
end                                                                          # hide

# The three implementations, timed side by side per temperature.

obsdata    = readdlm(obs_file, '\t'; header=true)[1]                                             # hide
timingdata = readdlm(timing_file, '\t'; header=true)[1]                                          # hide
## echo setup + results to the terminal (the Markdown table below is for the rendered docs)      # hide
println("\nSunny interop: Heisenberg, L=$L")                                                     # hide
println(@sprintf("therm=%d  prod=%d", therm, prod))                                              # hide
println(@sprintf("%-7s %12s %12s %12s  %12s %12s %12s  %10s %11s %11s", "T",                     # hide
    "e_sunny", "e_bridge", "e_native", "m_sunny", "m_bridge", "m_native",                        # hide
    "t_sunny(s)", "t_bridge(s)", "t_native(s)"))                                                 # hide
for r in axes(obsdata, 1)                                                                        # hide
    println(@sprintf("%-7.3f %12.6f %12.6f %12.6f  %12.6f %12.6f %12.6f  %10.4f %11.4f %11.4f",  # hide
        obsdata[r, 1], obsdata[r, 2], obsdata[r, 3], obsdata[r, 4],                              # hide
        obsdata[r, 5], obsdata[r, 6], obsdata[r, 7],                                             # hide
        timingdata[r, 2], timingdata[r, 3], timingdata[r, 4]))                                   # hide
end                                                                                               # hide
io = IOBuffer()                                                                                   # hide
println(io, "| T | e_sunny | e_bridge | e_native | m_sunny | m_bridge | m_native | t_sunny [s] | t_bridge [s] | t_native [s] |")  # hide
println(io, "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")                               # hide
for r in axes(obsdata, 1)                                                                        # hide
    println(io, @sprintf("| %.3f | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f | %.4f | %.4f | %.4f |",  # hide
        obsdata[r, 1], obsdata[r, 2], obsdata[r, 3], obsdata[r, 4],                              # hide
        obsdata[r, 5], obsdata[r, 6], obsdata[r, 7],                                             # hide
        timingdata[r, 2], timingdata[r, 3], timingdata[r, 4]))                                   # hide
end                                                                                                # hide
Markdown.parse(String(take!(io)))                                                                 # hide

# Energy and magnetization per site vs. temperature, all three implementations overlapping.

p1 = plot(obsdata[:, 1], obsdata[:, 2]; marker=:o, xlabel="T", ylabel="e per site", label="Sunny", title="Heisenberg, L=$L")  # hide
plot!(p1, obsdata[:, 1], obsdata[:, 3]; marker=:d, label="MCX bridge")   # hide
plot!(p1, obsdata[:, 1], obsdata[:, 4]; marker=:s, label="MCX native")  # hide

p2 = plot(obsdata[:, 1], obsdata[:, 5]; marker=:o, xlabel="T", ylabel="|m| per site", label="Sunny")  # hide
plot!(p2, obsdata[:, 1], obsdata[:, 6]; marker=:d, label="MCX bridge")   # hide
plot!(p2, obsdata[:, 1], obsdata[:, 7]; marker=:s, label="MCX native")  # hide

plt = plot(p1, p2; layout=(1, 2), size=(900, 380))               # hide
savefig(plt, joinpath(@__DIR__, "sunny_heisenberg_observables.png"))  # hide
plt                                                               # hide
