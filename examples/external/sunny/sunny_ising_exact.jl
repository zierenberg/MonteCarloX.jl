
# # Sunny Ising with MCX Algorithm Control
#
# Bridge example: model state/proposals/energy deltas come from Sunny, while
# acceptance/counters come from MonteCarloX (`MetropolisAlgorithm`).
#
# We compare three chains:
# 1) Sunny native `LocalSampler`
# 2) MCX algorithm on Sunny state updates (bridge)
# 3) MCX native (`MCXSpins.IsingSystem` + `MetropolisAlgorithm`)
#
# and include exact finite-size 2D Ising energies and measured runtimes.
import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()  #src

using Random, Statistics
using StatsBase: weights, mean
using Plots
using Sunny
using MCXSpins: logdos_exact_ising2D, IsingSystem, init!, energy, spin_flip!
using MonteCarloX: reweight, get_centers, MetropolisAlgorithm, accept!, acceptance_rate, reset!

smoke = get(ENV, "MCX_SMOKE", get(ENV, "MCX_CI", "false")) == "true"

L = 8
Tc = 2 / log(1 + sqrt(2.0))
therm_sweeps = smoke ? 300 : 2_000
prod_sweeps = smoke ? 2_000 : 12_000
Ts = [1.8, Tc, 3.0]

function make_sunny_ising(L)
    latvecs = lattice_vectors(1, 1, 10, 90, 90, 90)
    crystal = Crystal(latvecs, [[0, 0, 0]])
    sys = System(crystal, [1 => Moment(s=1, g=-1)], :dipole; dims=(L, L, 1))
    polarize_spins!(sys, (0, 0, 1))
    set_exchange!(sys, -1.0, Bond(1, 1, (1, 0, 0)))
    set_field!(sys, (0.0, 0.0, 0.0))
    return sys
end

function init_sunny_ising_pm!(sys, rng::AbstractRNG)
    # Keep Sunny in the strict Ising subspace: each spin is either +z or -z.
    polarize_spins!(sys, (0, 0, 1))
    for site in eachsite(sys)
        if rand(rng, Bool)
            Sunny.setspin!(sys, Sunny.propose_flip(sys, site), site)
        end
    end
    return nothing
end

function step_mcx!(sys, alg)
    site = rand(sys.rng, eachsite(sys))
    proposal = Sunny.propose_flip(sys, site)
    dE = Sunny.local_energy_change(sys, site, proposal)
    if accept!(alg, dE)
        Sunny.setspin!(sys, proposal, site)
    end
    return nothing
end

function sweep_mcx!(sys, alg)
    for _ in 1:Sunny.nsites(sys)
        step_mcx!(sys, alg)
    end
    return nothing
end

function sample_with_sunny(sys, T; therm_sweeps, prod_sweeps)
    sampler = LocalSampler(kT=T, nsweeps=1.0, propose=propose_flip)
    for _ in 1:therm_sweeps
        step!(sys, sampler)
    end
    es = Float64[]
    for _ in 1:prod_sweeps
        step!(sys, sampler)
        push!(es, energy_per_site(sys))
    end
    return mean(es), sampler
end

function sample_with_mcx(sys, T; therm_sweeps, prod_sweeps, seed=1234)
    alg = MetropolisAlgorithm(Xoshiro(seed); β=1 / T)
    for _ in 1:therm_sweeps
        sweep_mcx!(sys, alg)
    end
    reset!(alg)
    es = Float64[]
    for _ in 1:prod_sweeps
        sweep_mcx!(sys, alg)
        push!(es, energy_per_site(sys))
    end
    return mean(es), alg
end

function sample_with_mcx_native!(sys, alg, N; therm_sweeps, prod_sweeps)
    for _ in 1:(therm_sweeps * N)
        spin_flip!(sys, alg)
    end
    reset!(alg)
    es = Float64[]
    for _ in 1:prod_sweeps
        for _ in 1:N
            spin_flip!(sys, alg)
        end
        push!(es, energy(sys) / N)
    end
    return mean(es), alg
end

function time_seconds(run!; reps=3)
    tmin = typemax(Int)
    for _ in 1:reps
        t0 = time_ns()
        run!()
        dt = time_ns() - t0
        tmin = min(tmin, dt)
    end
    return tmin / 1e9
end

function build_sunny_systems(L, Ts; seed_offset)
    systems = Vector{Any}(undef, length(Ts))
    for (i, T) in pairs(Ts)
        sys = make_sunny_ising(L)
        init_sunny_ising_pm!(sys, Xoshiro(seed_offset + round(Int, 1_000 * T)))
        systems[i] = sys
    end
    return systems
end

function build_native_states(L, Ts; seed_offset)
    states = Vector{Any}(undef, length(Ts))
    for (i, T) in pairs(Ts)
        sys = IsingSystem([L, L])
        init!(sys, :random, rng=MersenneTwister(seed_offset + round(Int, 1_000 * T)))
        alg = MetropolisAlgorithm(Xoshiro(seed_offset + 1_000 + round(Int, 1_000 * T)); β=1 / T)
        states[i] = (sys, alg)
    end
    return states
end

function update_sunny!(es, systems, Ts; therm_sweeps, prod_sweeps)
    empty!(es)
    for (i, T) in pairs(Ts)
        e, _ = sample_with_sunny(systems[i], T; therm_sweeps=therm_sweeps, prod_sweeps=prod_sweeps)
        push!(es, e)
    end
    return nothing
end

function update_bridge!(es, accs, systems, Ts; therm_sweeps, prod_sweeps)
    empty!(es)
    empty!(accs)
    for (i, T) in pairs(Ts)
        e, alg = sample_with_mcx(systems[i], T; therm_sweeps=therm_sweeps, prod_sweeps=prod_sweeps)
        push!(es, e)
        push!(accs, acceptance_rate(alg))
    end
    return nothing
end

function update_native!(es, accs, states, Ts, N; therm_sweeps, prod_sweeps)
    empty!(es)
    empty!(accs)
    for i in eachindex(Ts)
        sys, alg = states[i]
        e, _ = sample_with_mcx_native!(sys, alg, N; therm_sweeps=therm_sweeps, prod_sweeps=prod_sweeps)
        push!(es, e)
        push!(accs, acceptance_rate(alg))
    end
    return nothing
end

logdos = logdos_exact_ising2D(L)
E = get_centers(logdos)
e_exact = [mean(E, weights(reweight(logdos, -E ./ T))) / L^2 for T in Ts]

e_sunny = Float64[]
e_bridge = Float64[]
e_native = Float64[]
acc_bridge = Float64[]
acc_native = Float64[]
N = L^2

# Warm compilation for construction and update paths before measuring.
build_sunny_systems(L, Ts; seed_offset=10_000)
build_sunny_systems(L, Ts; seed_offset=20_000)
build_native_states(L, Ts; seed_offset=30_000)

update_sunny!(Float64[], build_sunny_systems(L, Ts; seed_offset=40_000), Ts; therm_sweeps=1, prod_sweeps=1)
update_bridge!(Float64[], Float64[], build_sunny_systems(L, Ts; seed_offset=50_000), Ts; therm_sweeps=1, prod_sweeps=1)
update_native!(Float64[], Float64[], build_native_states(L, Ts; seed_offset=60_000), Ts, N; therm_sweeps=1, prod_sweeps=1)

time_construct_sunny = time_seconds(() -> build_sunny_systems(L, Ts; seed_offset=10_000))
time_construct_bridge = time_seconds(() -> build_sunny_systems(L, Ts; seed_offset=20_000))
time_construct_native = time_seconds(() -> build_native_states(L, Ts; seed_offset=30_000))

sunny_systems = build_sunny_systems(L, Ts; seed_offset=10_000)
bridge_systems = build_sunny_systems(L, Ts; seed_offset=20_000)
native_states = build_native_states(L, Ts; seed_offset=30_000)

time_sunny = time_seconds(() -> update_sunny!(e_sunny, sunny_systems, Ts; therm_sweeps=therm_sweeps, prod_sweeps=prod_sweeps), reps=1)
time_bridge = time_seconds(() -> update_bridge!(e_bridge, acc_bridge, bridge_systems, Ts; therm_sweeps=therm_sweeps, prod_sweeps=prod_sweeps), reps=1)
time_native = time_seconds(() -> update_native!(e_native, acc_native, native_states, Ts, N; therm_sweeps=therm_sweeps, prod_sweeps=prod_sweeps), reps=1)

println("Sunny Ising bridge vs native MCX (L=$(L), therm=$(therm_sweeps), prod=$(prod_sweeps), nT=$(length(Ts)))")
for i in eachindex(Ts)
    de_sunny = abs(e_sunny[i] - e_exact[i])
    de_bridge = abs(e_bridge[i] - e_exact[i])
    de_native = abs(e_native[i] - e_exact[i])
    println("  T=$(round(Ts[i], digits=6))  e_exact=$(round(e_exact[i], digits=6))  e_sunny=$(round(e_sunny[i], digits=6))  e_bridge=$(round(e_bridge[i], digits=6))  e_native=$(round(e_native[i], digits=6))")
    println("      |Δe|: Sunny=$(round(de_sunny, digits=6))  Bridge=$(round(de_bridge, digits=6))  MCX-native=$(round(de_native, digits=6))")
end
println("  acceptance bridge (mean over T) = $(round(mean(acc_bridge), digits=6))")
println("  acceptance native (mean over T) = $(round(mean(acc_native), digits=6))")
println("  construction runtime Sunny [s]        = $(round(time_construct_sunny, digits=3))")
println("  construction runtime MCX-on-Sunny [s] = $(round(time_construct_bridge, digits=3))")
println("  construction runtime MCX-native [s]   = $(round(time_construct_native, digits=3))")
println("  update runtime Sunny LocalSampler [s] = $(round(time_sunny, digits=3))")
println("  update runtime MCX-on-Sunny [s]       = $(round(time_bridge, digits=3))")
println("  update runtime MCX-native [s]         = $(round(time_native, digits=3))")

construct_times = [time_construct_sunny, time_construct_bridge, time_construct_native]
update_times = [time_sunny, time_bridge, time_native]
println("  comparison (update-only):")
println("    MCX-on-Sunny / Sunny      = $(round(time_bridge / time_sunny, digits=3))")
println("    MCX-native / Sunny        = $(round(time_native / time_sunny, digits=3))")
println("    MCX-on-Sunny / MCX-native = $(round(time_bridge / time_native, digits=3))")

p1 = plot(Ts, e_exact; lw=2, ls=:dot, color=:black, label="exact", xlabel="T", ylabel="e per site", title="2D Ising (L=$(L))")
plot!(p1, Ts, e_sunny; marker=:circle, lw=2, label="Sunny LocalSampler")
plot!(p1, Ts, e_bridge; marker=:diamond, lw=2, label="MCX on Sunny")
plot!(p1, Ts, e_native; marker=:square, lw=2, label="MCX native")

labels = ["Sunny", "MCX-on-Sunny", "MCX-native"]
runtime_matrix = hcat(construct_times, update_times)
p2 = bar(labels, runtime_matrix; label=["construction" "update"], bar_position=:dodge,
         ylabel="runtime [s]", title="Separated timing: construction vs updates")

plot(p1, p2; layout=(1, 2), size=(1060, 380))
