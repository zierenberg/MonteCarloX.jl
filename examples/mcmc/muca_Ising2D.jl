# # Multicanonical Sampling of the 2D Ising Model
#
# Multicanonical (muca) sampling iteratively reshapes the sampling weights until the
# energy histogram is flat, so a single chain visits the whole energy range and the
# density of states ``g(E)`` can be read off from the converged weights. Each
# iteration records a histogram and updates the weights; over the iterations the
# histogram flattens and the estimate approaches the exact result of
# [Beale (1996)](https://doi.org/10.1103/PhysRevLett.76.78).

using Random, StatsBase, Plots, DelimitedFiles
using MonteCarloX, MCXSpins

datadir   = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "data")))   # hide
lw_file   = joinpath(datadir, "muca_Ising2D_logweight.tsv")     # hide
hist_file = joinpath(datadir, "muca_Ising2D_histogram.tsv")     # hide
L      = 8
n_iter = 10
nothing # hide

# Each iteration thermalizes, records a histogram over the visited energies, and
# refines the log-weights toward a flat histogram — keeping every iteration's
# histogram and weights so we can watch the convergence.
if !isfile(lw_file)                                             # hide
E   = get_centers(logdos_exact_ising2D(L))
sys = Ising([L, L])
init!(sys, :random, rng = Xoshiro(1000))
alg = Multicanonical(Xoshiro(1000), E)

W = zeros(length(E), n_iter)     # log-weights after each iteration
H = zeros(length(E), n_iter)     # recorded histogram of each iteration
for it in 1:n_iter
    for _ in 1:(1_000   * length(sys.spins)); spin_flip!(sys, alg); end
    reset!(alg)
    for _ in 1:(100_000 * length(sys.spins)); spin_flip!(sys, alg); end
    update!(ensemble(alg); mode = :simple)
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
