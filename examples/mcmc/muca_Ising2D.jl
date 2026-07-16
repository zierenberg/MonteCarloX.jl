# # Multicanonical Sampling of the 2D Ising Model
#
# Multicanonical (muca) sampling [Berg & Neuhaus 1992] iteratively reshapes the sampling
# weights until the energy histogram is flat, so a single chain visits the whole energy range
# and the density of states ``g(E)`` can be read off from the converged weights. Each iteration
# records a histogram and updates the weights — the local moves are ordinary Metropolis
# under the reshaped weight. Over the iterations the histogram flattens
# and the estimate approaches the exact result of
# [Beale (1996)](https://doi.org/10.1103/PhysRevLett.76.78). Full references are listed at the
# bottom of this page.

using Random, StatsBase, Plots, DelimitedFiles
using MonteCarloX, MCXSpins, MPI

datadir   = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "data")))   # hide
lw_file   = joinpath(datadir, "muca_Ising2D_logweight.tsv")     # hide
hist_file = joinpath(datadir, "muca_Ising2D_histogram.tsv")     # hide
L      = 8
n_iter = 10
nothing # hide

# Each iteration thermalizes, records a histogram over the visited energies, and
# refines the log-weights toward a flat histogram — keeping every iteration's
# histogram and weights so we can watch the convergence.
# One sweep is one attempted flip per site; the function barrier keeps the hot loop
# specialized (14× faster than looping in global scope).
sweep!(sys, alg, n_sweeps) =
    (for _ in 1:n_sweeps, _ in 1:length(sys.spins); spin_flip!(sys, alg); end)

if !isfile(lw_file)                                             # hide
E   = get_centers(logdos_exact_ising2D(L))
sys = IsingSystem([L, L])
init!(sys, :random, rng = Xoshiro(1000))
alg = MulticanonicalAlgorithm(Xoshiro(1000), E)

W = zeros(length(E), n_iter)     # log-weights after each iteration
H = zeros(length(E), n_iter)     # recorded histogram of each iteration
for it in 1:n_iter
    sweep!(sys, alg, 1_000)
    reset!(alg)
    sweep!(sys, alg, 100_000)
    update_logweight!(ensemble(alg); mode = :simple)
    W[:, it] = ensemble(alg).logweight.values
    H[:, it] = ensemble(alg).histogram.values
end

header = permutedims(["E"; ["iter$(it)" for it in 1:n_iter]])   # hide
mkpath(datadir)                                                 # hide
writedlm(lw_file,   [header; hcat(E, W)], '\t')                 # hide
writedlm(hist_file, [header; hcat(E, H)], '\t')                 # hide
end                                                                 # hide
lw = readdlm(lw_file,   '\t'; header = true)[1]                     # hide
hh = readdlm(hist_file, '\t'; header = true)[1]                     # hide
E  = lw[:, 1]; W = lw[:, 2:end]; H = hh[:, 2:end]                   # hide
nothing # hide

# The estimated log density of states is ``-\log W(E)`` (anchored at ``E=0``),
# compared against the exact result; the error per iteration measures convergence.
exact  = logdos_exact_ising2D(L)
exact.values .-= exact[0]
finite = isfinite.(exact.values)
i0     = findfirst(==(0.0), E)
logg(w) = (g = -w; g .-= g[i0]; g[.!finite] .= NaN; g)
rmse(w) = sqrt(mean((logg(w)[finite] .- exact.values[finite]) .^ 2))
nothing # hide

# Over the iterations the histograms flatten (left), the estimated log-DOS
# converges onto the exact curve (middle), and the error decays (right).
cols = palette(:viridis, n_iter)

p1 = plot(xlabel = "E", ylabel = "counts", title = "histograms", legend = false)
for it in 1:n_iter
    plot!(p1, E, H[:, it]; lw = 2, color = cols[it])
end

p2 = plot(xlabel = "E", ylabel = "log g(E)", title = "estimated DOS", legend = false)
for it in 1:n_iter
    plot!(p2, E, logg(W[:, it]); lw = 2, color = cols[it])
end
plot!(p2, E, exact.values; lw = 2, color = :black, ls = :dash)

p3 = scatter(1:n_iter, [rmse(W[:, it]) for it in 1:n_iter];
             xlabel = "iteration", ylabel = "RMSE", title = "convergence",
             yscale = :log10, legend = false, color = :steelblue, ms = 5)

plot(p1, p2, p3; layout = (1, 3), size = (1100, 300), margin = 4Plots.mm)

# ## Running in parallel
#
# Multicanonical sampling parallelizes almost for free [Zierenberg, Marenz & Janke 2013]: run
# several **independent replicas** that share one weight function, merge their histograms after
# each iteration, refine the shared weights on the root, and broadcast them back. With `n`
# replicas each doing `1/n` of
# the sweeps, an iteration finishes in a fraction of the wall time — and converges to the
# *same* density of states. `ParallelMulticanonical` wraps this over a backend; only the
# backend (and, on a cluster, the communication) changes. Both variants below reuse the same
# `L`, `n_iter`, `E`, and `sweep!` as the serial run, and cache their per-iteration weights so
# the single-process docs build can reload and compare them.

# ### Threads (shared memory)
#
# One replica per thread over a `ThreadsBackend`, all sharing memory. Launch with
# `julia -t 4`.

threads_file = joinpath(datadir, "muca_Ising2D_threads.tsv")       # hide
tsv_header   = permutedims(["E"; ["iter$(it)" for it in 1:n_iter]]) # hide
E_bins       = get_centers(logdos_exact_ising2D(L))                 # hide
if Threads.nthreads() > 1 && !isfile(threads_file)                 # hide
backend = init(:threads)                                # one replica per thread
algs    = [MulticanonicalAlgorithm(Xoshiro(1000 + i), E_bins) for i in 1:size(backend)]
pmuca   = ParallelMulticanonical(backend, algs)
systems = [IsingSystem([L, L]) for _ in 1:size(backend)]
for i in 1:size(backend); init!(systems[i], :random; rng = algs[i].rng); end

W_par = zeros(length(E_bins), n_iter)
for it in 1:n_iter
    Threads.@threads for i in 1:size(pmuca)               # replicas run concurrently
        alg, sys = algorithm(pmuca, i), systems[i]
        sweep!(sys, alg, 1_000); reset!(alg)
        sweep!(sys, alg, 100_000 ÷ size(pmuca))           # each thread: 1/n of the sweeps
    end
    merge_histograms!(pmuca)                              # collect histograms on the root
    on_root(pmuca) do
        update_logweight!(ensemble(algorithm(pmuca, 1)); mode = :simple)
        W_par[:, it] = ensemble(algorithm(pmuca, 1)).logweight.values
    end
    distribute_logweight!(pmuca)                          # broadcast refined weights back
end
writedlm(threads_file, [tsv_header; hcat(E_bins, W_par)], '\t')    # hide
end                                                                # hide
nothing # hide

# ### MPI (distributed memory)
#
# On a cluster there is no shared array: **each rank owns one replica**, and the histogram
# merge and weight broadcast happen over MPI. The loop is otherwise identical — `init(:MPI)`
# selects the backend. Launch with `mpiexec -n 4 julia`. The same code lives as a full
# standalone template in `examples/mcmc/muca_Ising2D_mpi.jl`.

mpi_file = joinpath(datadir, "muca_Ising2D_mpi.tsv")               # hide
if get(ENV, "MCX_MPI", "0") == "1" && !isfile(mpi_file)            # hide
backend = init(:MPI)                                      # one rank per replica
alg     = MulticanonicalAlgorithm(Xoshiro(1000 + rank(backend)), E_bins)
pmuca   = ParallelMulticanonical(backend, alg)
sys     = IsingSystem([L, L]); init!(sys, :random; rng = alg.rng)

W_par = zeros(length(E_bins), n_iter)
for it in 1:n_iter
    sweep!(sys, alg, 1_000); reset!(alg)
    sweep!(sys, alg, 100_000 ÷ size(pmuca))               # each rank: 1/n of the sweeps
    merge_histograms!(pmuca)                              # MPI reduction to the root
    on_root(pmuca) do
        update_logweight!(ensemble(alg); mode = :simple)
        W_par[:, it] = ensemble(alg).logweight.values
    end
    distribute_logweight!(pmuca)                          # MPI broadcast
end
on_root(pmuca) do; writedlm(mpi_file, [tsv_header; hcat(E_bins, W_par)], '\t'); end   # hide
finalize!(backend)
end                                                                # hide
nothing # hide

# ### Convergence: serial vs threads vs MPI
#
# All three routes refine the same weights, so they land on the same density of states —
# parallelism buys wall-clock time, not accuracy. Their per-iteration error against Beale's
# exact log-DOS overlaps:

if isfile(threads_file) && isfile(mpi_file)                        # hide
Wt = readdlm(threads_file, '\t'; header = true)[1][:, 2:end]       # hide
Wm = readdlm(mpi_file,     '\t'; header = true)[1][:, 2:end]       # hide
scatter(1:n_iter, [rmse(W[:, it])  for it in 1:n_iter]; label = "serial",
        yscale = :log10, xlabel = "iteration", ylabel = "RMSE vs exact",
        color = :black, ms = 7, legend = :topright)
scatter!(1:n_iter, [rmse(Wt[:, it]) for it in 1:n_iter];
         label = "threads (×4)", color = :steelblue, marker = :diamond, ms = 5)
scatter!(1:n_iter, [rmse(Wm[:, it]) for it in 1:n_iter];
         label = "MPI (×4)", color = :crimson, marker = :utriangle, ms = 4)
end                                                                # hide

# ## References
#
# - B. A. Berg, T. Neuhaus, *Multicanonical ensemble: a new approach to simulate first-order
#   phase transitions*, Phys. Rev. Lett. **68**, 9 (1992).
#   [doi:10.1103/PhysRevLett.68.9](https://doi.org/10.1103/PhysRevLett.68.9)
# - W. Janke, *Multicanonical Monte Carlo simulations*, Physica A **254**, 164 (1998).
#   [doi:10.1016/S0378-4371(98)00014-4](https://doi.org/10.1016/S0378-4371(98)00014-4)
# - J. Zierenberg, M. Marenz, W. Janke, *Scaling properties of a parallel implementation of the
#   multicanonical algorithm*, Comput. Phys. Commun. **184**, 1155 (2013).
#   [doi:10.1016/j.cpc.2012.12.006](https://doi.org/10.1016/j.cpc.2012.12.006)
# - P. D. Beale, *Exact distribution of energies in the two-dimensional Ising model*,
#   Phys. Rev. Lett. **76**, 78 (1996).
#   [doi:10.1103/PhysRevLett.76.78](https://doi.org/10.1103/PhysRevLett.76.78)
