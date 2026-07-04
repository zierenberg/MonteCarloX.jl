# # Parallel Tempering of the 2D Ising Model
#
# Parallel tempering (replica exchange) runs several replicas at different inverse
# temperatures ``\beta``. Each replica performs local Metropolis sweeps, then
# neighboring replicas attempt to swap configurations. The benefit is robust
# sampling across phase transitions and better exploration of rugged energy
# landscapes, via repeated annealing across the temperature ladder.
#
# The loop is intentionally simple: (1) thermalize, (2) run measurement sweeps,
# (3) attempt exchanges, and (4) compare the sampled energy distributions to the
# exact reference.

using Random, Statistics, StatsBase, Plots, DelimitedFiles
using MonteCarloX, MCXSpins

datadir      = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "data")))  # hide
samples_file = joinpath(datadir, "pt_Ising2D_energies.tsv")    # hide
betas_file   = joinpath(datadir, "pt_Ising2D_betas.tsv")       # hide

L                       = 8
nreplicas               = 4
nmeasurements           = 20_000
ntherm_init             = 2_000
ntherm_after_exchange   = 20
sweeps_between_exchange = 200
Tmin, Tmax              = 1.5, 3.0
seed                    = 42
nothing # hide

# ## Sweep helper
#
# One sweep is `L*L` attempted single-spin flips of one replica.

function sweep_replica!(sys, alg, L)
    for _ in 1:(L * L)
        spin_flip!(sys, alg)
    end
    return nothing
end

# ## Serial parallel-tempering run
#
# We build a temperature ladder, thermalize each replica, then alternate between
# measurement sweeps (recording each replica's energy, tagged by its current
# temperature index) and exchange attempts between neighbors.

if !isfile(samples_file)                                        # hide
betas   = set_betas(nreplicas, inv(Tmax), inv(Tmin), :uniform)
systems = [Ising([L, L]) for _ in 1:nreplicas]
pt      = ParallelTempering(betas; seed = seed, rng = MersenneTwister)

for r in 1:nreplicas
    init!(systems[r], :random; rng = algorithm(pt, r).rng)
    for _ in 1:ntherm_init
        sweep_replica!(systems[r], algorithm(pt, r), L)
    end
end

energies = zeros(Float64, nreplicas)
mags     = zeros(Float64, nreplicas)
meas = Measurements([
    :energies => (_ -> copy(energies)) => Vector{Vector{Float64}}(),
    :mags     => (_ -> copy(mags))     => Vector{Vector{Float64}}(),
], interval = 1)
index_trace = Vector{Vector{Int}}()

n_exchanges = nmeasurements ÷ sweeps_between_exchange
for exch in 1:n_exchanges
    for r in 1:nreplicas, _ in 1:ntherm_after_exchange       # re-thermalize after an exchange
        sweep_replica!(systems[r], algorithm(pt, r), L)
    end
    for s in 1:sweeps_between_exchange                        # measurement sweeps
        for r in 1:nreplicas
            sweep_replica!(systems[r], algorithm(pt, r), L)
            energies[r] = energy(systems[r])
            mags[r]     = magnetization(systems[r])
        end
        step     = (exch - 1) * sweeps_between_exchange + s
        n_before = length(data(meas, :energies))
        measure!(meas, nothing, step)
        length(data(meas, :energies)) > n_before && push!(index_trace, copy(index(pt)))
    end
    MonteCarloX.update!(pt, energies)                         # replica exchange
end

energy_samples = [Float64[] for _ in 1:nreplicas]            # regroup samples by temperature
energy_trace   = data(meas, :energies)
for k in eachindex(energy_trace)
    idxs = index_trace[k]; es = energy_trace[k]
    for r in eachindex(idxs)
        push!(energy_samples[idxs[r]], es[r])
    end
end

replica_col = reduce(vcat, [fill(r, length(energy_samples[r])) for r in 1:nreplicas])  # hide
energy_col  = reduce(vcat, energy_samples)                                              # hide
mkpath(datadir)                                                                         # hide
writedlm(samples_file, ["replica" "energy"; hcat(replica_col, energy_col)], '\t')       # hide
writedlm(betas_file,   ["beta"; betas], '\t')                                           # hide
end                                                             # hide
sm    = readdlm(samples_file, '\t'; header = true)[1]          # hide
betas = vec(readdlm(betas_file, '\t'; header = true)[1])       # hide
nreplicas = length(betas)                                      # hide
energy_samples = [Float64[] for _ in 1:nreplicas]             # hide
for row in 1:size(sm, 1); push!(energy_samples[Int(sm[row, 1])], sm[row, 2]); end  # hide
nothing # hide

# ## Sampled vs. exact energy distributions
#
# For each temperature the sampled energy histogram sits on top of the exact
# Boltzmann-weighted density of states ([Beale 1996](https://doi.org/10.1103/PhysRevLett.76.78)).

exact_logdos = logdos_exact_ising2D(L)
edges = get_edges(exact_logdos.bins[1])
plots = Any[]
for (i, β) in enumerate(betas)
    p_i = plot(xlabel = "energy", ylabel = "probability",
               title = "β = $(round(β, digits = 3))", legend = :topright)
    samples_i = energy_samples[i]
    if !isempty(samples_i)
        hist_i = fit(Histogram, samples_i, edges, closed = :left)
        dist_i = StatsBase.normalize(hist_i; mode = :probability)
        plot!(p_i, dist_i; label = "PT", lw = 2, color = :steelblue)
    end
    dist_exact = distribution_exact_ising2D(L, β)
    plot!(p_i, get_centers(dist_exact), dist_exact.values;
          label = "exact", lw = 2, color = :black, ls = :dash)
    push!(plots, p_i)
end
plot(plots...; layout = (2, 2), size = (950, 720), margin = 3Plots.mm)
