# # Parallel Tempering of the 2D Ising Model
#
# Parallel tempering (replica exchange) [Swendsen & Wang 1986; Hukushima & Nemoto 1996] runs several replicas at different inverse
# temperatures ``\beta``. Each replica performs local Metropolis sweeps, then
# neighboring replicas attempt to swap configurations. The benefit is robust
# sampling across phase transitions and better exploration of rugged energy
# landscapes, via repeated annealing across the temperature ladder.
#
# The loop is intentionally simple: (1) thermalize, (2) run measurement sweeps,
# (3) attempt exchanges, and (4) compare the sampled energy distributions to the
# exact reference.

using Random, Statistics, StatsBase, Plots, DelimitedFiles
using MonteCarloX, MCXSpins, MPI

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
# One sweep is one attempted single-spin flip per site of one replica (the function
# barrier keeps the hot loop specialized).

sweep!(sys, alg, n_sweeps) =
    (for _ in 1:n_sweeps, _ in 1:length(sys.spins); spin_flip!(sys, alg); end)
nothing # hide

# ## Serial parallel-tempering run
#
# We build a temperature ladder, thermalize each replica, then alternate between
# measurement sweeps (recording each replica's energy, tagged by its current
# temperature index) and exchange attempts between neighbors.

if !isfile(samples_file)                                        # hide
betas   = set_betas(nreplicas, inv(Tmax), inv(Tmin), :uniform)
systems = [IsingSystem([L, L]) for _ in 1:nreplicas]
pt      = ParallelTempering(betas; seed = seed, rng = Xoshiro)

for r in 1:nreplicas
    init!(systems[r], :random; rng = algorithm(pt, r).rng)
    sweep!(systems[r], algorithm(pt, r), ntherm_init)
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
    for r in 1:nreplicas                                      # re-thermalize after an exchange
        sweep!(systems[r], algorithm(pt, r), ntherm_after_exchange)
    end
    for s in 1:sweeps_between_exchange                        # measurement sweeps
        for r in 1:nreplicas
            sweep!(systems[r], algorithm(pt, r), 1)
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
# density of states ([Beale 1996](https://doi.org/10.1103/PhysRevLett.76.78))
# reweighted to ``\beta``: the canonical distribution is just importance weights
# ``\log w = \log g(E) - \beta E``, normalized by `weights`.

logdos  = logdos_exact_ising2D(L)          # BinnedObject: centers, edges, values
E_exact = get_centers(logdos)
edges   = get_edges(logdos.bins[1])
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
    P_exact = weights(reweight(logdos, -β .* E_exact))
    plot!(p_i, E_exact, P_exact;
          label = "exact", lw = 2, color = :black, ls = :dash)
    push!(plots, p_i)
end
plot(plots...; layout = (2, 2), size = (950, 720), margin = 3Plots.mm)

# ## Running in parallel
#
# In parallel tempering each replica already lives at its own temperature, so spreading the
# replicas across threads or MPI ranks is natural: every worker runs one replica's local
# sweeps, and only the scalar energies cross the exchange step. The canonical energy at each
# temperature is unchanged. Both variants reuse the parameters above and cache ⟨E⟩(β) so the
# single-process docs build can compare them.

# ### Threads (shared memory)
#
# One replica per thread over a `ThreadsBackend`; `with_parallel` runs the per-replica sweeps
# concurrently, then the exchange step swaps neighbours. Launch with `julia -t 4`.

pt_threads_file = joinpath(datadir, "pt_Ising2D_threads.tsv")      # hide
if Threads.nthreads() > 1 && !isfile(pt_threads_file)              # hide
backend = init(:threads)
betas_p = set_betas(size(backend), inv(Tmax), inv(Tmin), :uniform)
algs    = [MetropolisAlgorithm(Xoshiro(seed + i); β = betas_p[i]) for i in 1:size(backend)]
pt      = ParallelTempering(backend, algs)
systems = [IsingSystem([L, L]) for _ in 1:size(pt)]
for i in 1:size(pt); init!(systems[i], :random; rng = algs[i].rng); end

sumE = zeros(size(pt)); cntE = zeros(Int, size(pt))       # ⟨E⟩ accumulators per ladder rung
for i in 1:size(pt); sweep!(systems[i], algs[i], ntherm_init); end
for exch in 1:(nmeasurements ÷ sweeps_between_exchange)
    with_parallel(pt) do i, alg                           # per-replica sweeps run concurrently
        sys = systems[i]
        for _ in 1:sweeps_between_exchange
            sweep!(sys, alg, 1)
            sumE[index(pt, i)] += energy(sys); cntE[index(pt, i)] += 1
        end
    end
    MonteCarloX.update!(pt, [energy(systems[i]) for i in 1:size(pt)])   # replica exchange
end
writedlm(pt_threads_file, ["beta" "meanE"; hcat(betas_p, sumE ./ cntE)], '\t')  # hide
end                                                                # hide
nothing # hide

# ### MPI (distributed memory)
#
# One rank per replica; `init(:MPI)` selects the backend and the exchange coordinates swaps
# over MPI, moving only scalar energies. Launch with `mpiexec -n 4 julia`. A full standalone
# template is `examples/mcmc/pt_Ising2D_mpi.jl`.

pt_mpi_file = joinpath(datadir, "pt_Ising2D_mpi.tsv")             # hide
if get(ENV, "MCX_MPI", "0") == "1" && !isfile(pt_mpi_file)        # hide
backend = init(:MPI)                                      # one rank per replica
betas_p = set_betas(size(backend), inv(Tmax), inv(Tmin), :uniform)
pt  = ParallelTempering(betas_p; seed = seed, rng = Xoshiro, backend = backend)
alg = algorithm(pt)
sys = IsingSystem([L, L]); init!(sys, :random; rng = alg.rng)

sumE = zeros(size(backend)); cntE = zeros(Int, size(backend))
sweep!(sys, alg, ntherm_init)
for meas in 1:nmeasurements
    sweep!(sys, alg, 1)
    sumE[index(pt)] += energy(sys); cntE[index(pt)] += 1  # accumulate into the current rung
    meas % sweeps_between_exchange == 0 && MonteCarloX.update!(pt, energy(sys))
end
gsumE = MPI.Reduce(sumE, +, backend.comm; root = backend.root)   # combine rungs on the root
gcntE = MPI.Reduce(cntE, +, backend.comm; root = backend.root)
on_root(pt) do; writedlm(pt_mpi_file, ["beta" "meanE"; hcat(betas_p, gsumE ./ gcntE)], '\t'); end  # hide
finalize!(backend)
end                                                               # hide
nothing # hide

# ### Canonical energy: serial vs threads vs MPI
#
# All three reproduce the exact ``\langle E \rangle(\beta)`` across the temperature ladder:

if isfile(pt_threads_file) && isfile(pt_mpi_file)                 # hide
dt = readdlm(pt_threads_file, '\t'; header = true)[1]             # hide
dm = readdlm(pt_mpi_file,     '\t'; header = true)[1]             # hide
meanE_serial = [mean(energy_samples[i]) for i in 1:nreplicas]
meanE_exact  = [mean(E_exact, weights(reweight(logdos, -β .* E_exact))) for β in betas]
scatter(betas, meanE_exact; label = "exact", color = :black, ms = 8,
        xlabel = "β", ylabel = "⟨E⟩", legend = :topright)
scatter!(betas, meanE_serial; label = "serial", color = :gray, marker = :xcross, ms = 6)
scatter!(dt[:, 1], dt[:, 2]; label = "threads (×4)", color = :steelblue, marker = :diamond, ms = 5)
scatter!(dm[:, 1], dm[:, 2]; label = "MPI (×4)", color = :crimson, marker = :utriangle, ms = 4)
end                                                               # hide
# ## References
#
# - R. H. Swendsen, J.-S. Wang, *Replica Monte Carlo simulation of spin-glasses*,
#   Phys. Rev. Lett. **57**, 2607 (1986).
#   [doi:10.1103/PhysRevLett.57.2607](https://doi.org/10.1103/PhysRevLett.57.2607)
# - C. J. Geyer, *Markov chain Monte Carlo maximum likelihood*, in Computing Science and Statistics:
#   Proc. 23rd Symp. on the Interface, p. 156 (1991).
# - K. Hukushima, K. Nemoto, *Exchange Monte Carlo method and application to spin glass simulations*,
#   J. Phys. Soc. Jpn. **65**, 1604 (1996).
#   [doi:10.1143/JPSJ.65.1604](https://doi.org/10.1143/JPSJ.65.1604)
