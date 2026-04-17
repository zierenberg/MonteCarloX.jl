# %% #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src
include(joinpath(@__DIR__, "..", "defaults.jl"))    #src

using Random, Statistics, StatsBase, Plots
using MonteCarloX, SpinSystems

# %%
# ## Background
#
# Parallel tempering (replica exchange) runs multiple replicas at different
# inverse temperatures $\beta$. Each replica performs local Metropolis sweeps,
# then neighboring replicas attempt an exchange.
#
# The main practical benefit is robust sampling across continuous phase
# transitions and better exploration of rugged energy landscapes via repeated
# annealing across the temperature ladder.
#
# In this example we keep the loop structure intentionally simple:
# 1) thermalize,
# 2) run measurement sweeps,
# 3) attempt exchanges,
# 4) compare sampled energy distributions to exact references.

# %%
# ## Constants

const CI_MODE = get(ENV, "MCX_SMOKE", get(ENV, "MCX_CI", "false")) == "true"

L = 8
nreplicas = CI_MODE ? 2 : 4
nmeasurements = CI_MODE ? 200 : 20_000
ntherm_init = CI_MODE ? 20 : 2_000
ntherm_after_exchange = CI_MODE ? 1 : 20
sweeps_between_exchange = CI_MODE ? 4 : 200
measurement_interval = 1

Tmin = 1.5
Tmax = 3.0
seed = 42

nreplicas >= 2 || throw(ArgumentError("nreplicas must be >= 2"))
nmeasurements >= 1 || throw(ArgumentError("nmeasurements must be >= 1"))
sweeps_between_exchange >= 1 || throw(ArgumentError("sweeps_between_exchange must be >= 1"))
measurement_interval >= 1 || throw(ArgumentError("measurement_interval must be >= 1"))

# %%
# ## Sweep helper

function sweep_replica!(sys, alg, L)
    for _ in 1:(L * L)
        spin_flip!(sys, alg)
    end
    return nothing
end

# %%
# ## Serial PT run

betas = set_betas(nreplicas, inv(Tmax), inv(Tmin), :uniform)

systems = [Ising([L, L]) for _ in 1:nreplicas]
pt = ParallelTempering(betas; seed=seed, rng=MersenneTwister)

for r in 1:nreplicas
    init!(systems[r], :random; rng=algorithm(pt, r).rng)
    for _ in 1:ntherm_init
        sweep_replica!(systems[r], algorithm(pt, r), L)
    end
end

energies = zeros(Float64, nreplicas)
mags = zeros(Float64, nreplicas)
meas = Measurements([
    :energies => (_ -> copy(energies)) => Vector{Vector{Float64}}(),
    :mags => (_ -> copy(mags)) => Vector{Vector{Float64}}(),
], interval=measurement_interval)
index_trace = Vector{Vector{Int}}()

n_exchanges = nmeasurements ÷ sweeps_between_exchange
let sample_counter = 0
    for exch in 1:n_exchanges
        ## re-thermalize after exchange
        for r in 1:nreplicas
            for _ in 1:ntherm_after_exchange
                sweep_replica!(systems[r], algorithm(pt, r), L)
            end
        end

        ## measure
        for _ in 1:sweeps_between_exchange
            for r in 1:nreplicas
                sweep_replica!(systems[r], algorithm(pt, r), L)
                energies[r] = energy(systems[r])
                mags[r] = magnetization(systems[r])
            end
            sample_counter += 1
            n_before = length(data(meas, :energies))
            measure!(meas, nothing, sample_counter)
            if length(data(meas, :energies)) > n_before
                push!(index_trace, copy(index(pt)))
            end
        end
        ## replica exchange
        MonteCarloX.update!(pt, energies)
    end
end 

energy_samples = [Float64[] for _ in 1:nreplicas]
mag_samples = [Float64[] for _ in 1:nreplicas]
energy_trace = data(meas, :energies)
mag_trace = data(meas, :mags)

@inbounds for k in eachindex(energy_trace)
    idxs = index_trace[k]
    es = energy_trace[k]
    ms = mag_trace[k]
    for r in eachindex(idxs)
        push!(energy_samples[idxs[r]], es[r])
        push!(mag_samples[idxs[r]], ms[r])
    end
end

rates = acceptance_rates(pt)

println("PT Ising2D serial finished")
println("L = $(L), replicas = $(nreplicas)")
println("mean exchange acceptance = $(round(mean(rates), digits=4))")

# %%
# ## Plot vs exact

for (i, β) in enumerate(betas)
    println("beta[$i] = $(round(β, digits=4)), samples = $(length(energy_samples[i]))")
end

exact_logdos = logdos_exact_ising2D(L)
edges = get_edges(exact_logdos.bins[1])
plots = Any[]

for (i, β) in enumerate(betas)
    p_i = plot(xlabel="Energy", ylabel="Probability",
               title="beta=$(round(β, digits=3))", legend=:topright)

    samples_i = energy_samples[i]
    if !isempty(samples_i)
        hist_i = fit(Histogram, samples_i, edges, closed=:left)
        dist_i = StatsBase.normalize(hist_i; mode=:probability)
        plot!(p_i, dist_i; label="PT serial", lw=2, color=:steelblue)
    end

    dist_exact = distribution_exact_ising2D(L, β)
    plot!(p_i, get_centers(dist_exact), dist_exact.values; label="Exact", lw=2, color=:black, ls=:dash)
    push!(plots, p_i)
end

ncols = 2
nrows = 2
plot(plots...;
    layout=(nrows, ncols),
    size=(950, 720),
    margin=3Plots.mm,
    background_color=:white,
    background_color_outside=:white)
