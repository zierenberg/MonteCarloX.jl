# %% #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src
include(joinpath(@__DIR__, "..", "defaults.jl"))    #src

# # Parallel Tempering for 2D Ising
#
# This example demonstrates a compact replica-exchange setup with Boltzmann
# ensembles. Replicas keep their states local and only exchange inverse
# temperatures between neighboring replicas.

using Random, Statistics, StatsBase, Plots
using Base.Threads
using MonteCarloX, SpinSystems

const CI_MODE = get(ENV, "MCX_SMOKE", get(ENV, "MCX_CI", "false")) == "true"

L = 8
nreplicas = CI_MODE ? 2 : 4
nmeasurements = CI_MODE ? 200 : 20_000
ntherm_init = CI_MODE ? 20 : 2_000
ntherm_after_exchange = CI_MODE ? 1 : 20
sweeps_between_exchange = CI_MODE ? 4 : 200
# phase transition is at T_c ≈ 2.269, so we choose a range around it
Tmin = 1.5
Tmax = 3.0
seed = 42

# %%
# ## Construct β values, systems, and replica samplers

nreplicas >= 2 || throw(ArgumentError("nreplicas must be >= 2"))
nmeasurements >= 1 || throw(ArgumentError("nmeasurements must be >= 1"))
sweeps_between_exchange >= 1 || throw(ArgumentError("sweeps_between_exchange must be >= 1"))

βs = set_betas(nreplicas, inv(Tmax), inv(Tmin), :uniform)
stats = ExchangeStats(nreplicas - 1)
labels = collect(1:nreplicas)

systems = [IsingLatticeOptim(L, L) for _ in 1:nreplicas]
algs = [Metropolis(MersenneTwister(seed + r); β=βs[r]) for r in 1:nreplicas]

for r in 1:nreplicas
    init!(systems[r], :random; rng=algs[r].rng)
end

# %%
# ## Allocate measurement buffers

energies = zeros(Float64, nreplicas)
mags = zeros(Float64, nreplicas)
ex_rng = MersenneTwister(seed + 100_000)
energy_samples = [Float64[] for _ in 1:nreplicas]
exchange_attempts = 0

function sweep_replica!(sys, alg)
    for _ in 1:(L * L)
        spin_flip!(sys, alg)
    end
    return nothing
end

@inline function beta_index(beta, βs)
    idx = findfirst(isequal(beta), βs)
    idx === nothing && throw(ArgumentError("beta $beta not found in ladder"))
    return idx
end

# %%
# ## Initial thermalization

for _ in 1:ntherm_init
    @threads for r in 1:nreplicas
        sweep_replica!(systems[r], algs[r])
    end
end

# %%
# ## Run PT dynamics and record energies by beta

for meas in 1:nmeasurements
    @threads for r in 1:nreplicas
        sys = systems[r]
        alg = algs[r]
        sweep_replica!(sys, alg)
        energies[r] = energy(sys)
        mags[r] = magnetization(sys)
    end

    @inbounds for r in 1:nreplicas
        iβ = beta_index(inverse_temperature(algs[r]), βs)
        push!(energy_samples[iβ], energies[r])
    end

    if meas % sweeps_between_exchange == 0
        stage = exchange_attempts % 2
        attempt_exchange_pairs!(ex_rng, algs, energies, stage; stats=stats, labels=labels)
        exchange_attempts += 1

        for _ in 1:ntherm_after_exchange
            @threads for r in 1:nreplicas
                sweep_replica!(systems[r], algs[r])
                energies[r] = energy(systems[r])
                mags[r] = magnetization(systems[r])
            end
        end
    end
end

# %%
# ## Basic diagnostics

rates = acceptance_rates(stats)
println("PT Ising2D finished")
println("L = $(L), replicas = $(nreplicas), threads = $(nthreads())")
println("beta range = [$(round(maximum(βs), digits=4)), $(round(minimum(βs), digits=4))]")
println("mean |m| per spin = $(round(mean(abs.(mags)) / (L*L), digits=4))")
println("mean exchange acceptance = $(round(mean(rates), digits=4))")

# ## PT sampling and energy histograms
#
# We collected energies for each beta value in the ladder.

for (i, β) in enumerate(βs)
    println("beta[$i] = ", round(β, digits=4),
        " (T = ", round(inv(β), digits=4), "), samples = ", length(energy_samples[i]))
end

# ## Compare sampled distribution to exact canonical distribution
#
# For L=8, we compare the sampled energy histogram to the exact canonical
# distribution from the exact DOS.

exact_logdos = logdos_exact_ising2D(L)
edges = get_edges(exact_logdos.bins[1])
plots = Any[]

for (i, β) in enumerate(βs)
    samples = energy_samples[i]
    dist_exact = distribution_exact_ising2D(L, β)

    if isempty(samples)
        p_i = plot(xlabel="Energy", ylabel="Probability",
                   title="beta=$(round(β, digits=3)) (no samples)", legend=false)
        push!(plots, p_i)
        continue
    end

    hist = fit(Histogram, samples, edges, closed=:left)
    dist = StatsBase.normalize(hist; mode=:probability)

    p_i = plot(xlabel="Energy", ylabel="Probability",
               title="beta=$(round(β, digits=3))", legend=:topright)
    plot!(p_i, dist; label="PT", lw=2, color=:steelblue)
    plot!(p_i, get_centers(dist_exact), dist_exact.values; label="Exact", lw=2, color=:black, ls=:dash)
    push!(plots, p_i)
end

ncols = 2
nrows = 2
plot(plots...;
    layout=(nrows, ncols),
    size=(900, 700),
    margin=3Plots.mm,
    background_color=:white,
    background_color_outside=:white)
