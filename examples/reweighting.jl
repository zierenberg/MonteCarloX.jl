# %% Reweighting — full example
#
# Run cell-by-cell in VSCode to see the figures, or from the shell:
#     julia --project=examples examples/reweighting.jl                 # run + (interactive) plots
#     julia --project=examples examples/reweighting.jl docs/src/data   # also write TSV summaries
#
# Reweighting turns samples recorded under one ensemble into expectations under
# another (importance sampling after the fact). Three cases, each validated
# against the exact 2D-Ising density of states (Beale 1996):
#   1. single-histogram (Metropolis) reweighting to nearby β
#   2. multicanonical → any β from a single run
#   3. multicanonical → the density of states
#
# When an output directory is given as the first argument, each cell also writes a
# small TSV table there; the docs pages under docs/src/examples/ read those tables
# and never re-run this simulation.

using Random, StatsBase, Plots, DelimitedFiles
using MonteCarloX, MCXSpins

outdir = isempty(ARGS) ? nothing : ARGS[1]
outdir === nothing || mkpath(outdir)

function writetsv(name, header, data)
    outdir === nothing && return nothing
    open(joinpath(outdir, name), "w") do io
        writedlm(io, [reshape(collect(header), 1, :); data], '\t')
    end
    return nothing
end

# %% Parameters and exact reference

L    = 8
N    = L * L
seed = 1000
sweeps_therm = 1_000
sweeps_prod  = 50_000
muca_iter    = 10
muca_sweeps  = 100_000

exact_logdos          = logdos_exact_ising2D(L)
exact_logdos.values .-= exact_logdos[0]
Es    = get_centers(exact_logdos)
edges = get_edges(exact_logdos.bins[1])

finite = isfinite.(exact_logdos.values)
Es_f   = Es[finite]
logg_f = exact_logdos.values[finite]

# exact canonical energy per spin from the exact density of states
function exact_avgE(β)
    lw = logg_f .- β .* Es_f
    m  = maximum(lw)
    w  = exp.(lw .- m)
    return sum(Es_f .* w) / sum(w) / N
end

# reweight recorded energies from `source` to a Boltzmann target at β → ⟨E⟩/N
avgE(energies, source, β) =
    mean(energies, weights(reweight(energies, source, BoltzmannEnsemble(β = β)))) / N

βdense = collect(range(0.10, 0.60; length = 101))
writetsv("reweighting_exact_avgE.tsv", ("beta", "avgE_exact"),
         hcat(βdense, exact_avgE.(βdense)))

# %% 1. Single-histogram (Metropolis) reweighting
# A canonical chain at β0 records energies; reweighting reaches nearby β, but the
# effective sample size collapses away from β0.

β0    = 0.30
sys_c = Ising([L, L]); init!(sys_c, :random, rng = Xoshiro(seed))
alg_c = Metropolis(Xoshiro(seed); β = β0)
for _ in 1:(sweeps_therm * N); spin_flip!(sys_c, alg_c); end
meas_c = Measurements([:E => energy => Float64[]], interval = N)
for i in 1:(sweeps_prod * N)
    spin_flip!(sys_c, alg_c)
    measure!(meas_c, sys_c, i)
end
energies_c = meas_c[:E].data

# a thinned sample of the recorded energies, so the docs page can run the actual
# reweighting idiom live on real data without redoing the simulation
thin(v, n = 1000) = v[1:max(1, length(v) ÷ n):end]
writetsv("reweighting_energies_canonical.tsv", ("E",), reshape(thin(energies_c), :, 1))

source_c = BoltzmannEnsemble(β = β0)
βwin     = collect(range(β0 - 0.08, β0 + 0.08; length = 33))
E_sh     = [avgE(energies_c, source_c, β) for β in βwin]
ess_sh   = [ess(reweight(energies_c, source_c, BoltzmannEnsemble(β = β))) / length(energies_c) for β in βwin]
writetsv("reweighting_single_histogram.tsv", ("beta", "avgE_reweight", "ess_frac"),
         hcat(βwin, E_sh, ess_sh))

p1 = plot(βdense, exact_avgE.(βdense); lw = 3, color = :black, label = "exact",
          xlabel = "β", ylabel = "⟨E⟩/N", title = "single-histogram")
scatter!(p1, βwin, E_sh; ms = 3, color = 1, label = "reweighted")

# %% 2. Multicanonical → any β
# One multicanonical run flattens the energy histogram, so its samples span the
# whole spectrum and reweight to any β.

sys_m = Ising([L, L]); init!(sys_m, :random, rng = Xoshiro(seed))
alg_m = Multicanonical(Xoshiro(seed), Es)
for _ in 1:muca_iter
    for _ in 1:(sweeps_therm * N); spin_flip!(sys_m, alg_m); end
    reset!(alg_m)
    for _ in 1:(muca_sweeps * N);  spin_flip!(sys_m, alg_m); end
    update!(ensemble(alg_m); mode = :simple)
end
reset!(alg_m)
meas_m = Measurements([:E => energy => Float64[]], interval = N)
for i in 1:(sweeps_prod * N)
    spin_flip!(sys_m, alg_m)
    measure!(meas_m, sys_m, i)
end
energies_m = meas_m[:E].data

source_m = ensemble(alg_m)
# thinned muca energies + the converged multicanonical log-weights W(E), so the
# docs page can reconstruct the muca source and reweight live
writetsv("reweighting_energies_muca.tsv", ("E",), reshape(thin(energies_m), :, 1))
writetsv("reweighting_muca_logweight.tsv", ("E", "logW"), hcat(Es, source_m.logweight.values))
βscan    = collect(range(0.10, 0.60; length = 51))
E_muca   = [avgE(energies_m, source_m, β) for β in βscan]
writetsv("reweighting_muca_avgE.tsv", ("beta", "avgE_reweight"), hcat(βscan, E_muca))

p2 = plot(βdense, exact_avgE.(βdense); lw = 3, color = :black, label = "exact",
          xlabel = "β", ylabel = "⟨E⟩/N", title = "muca → any β")
scatter!(p2, βscan, E_muca; ms = 3, color = 3, label = "muca reweighted")

# %% 3. Multicanonical → density of states
# Reweighting the muca samples to a flat target (the default) makes a weighted
# energy histogram proportional to g(E) — the density of states.

iw_dos   = reweight(energies_m, source_m)
hist_dos = fit(Histogram, energies_m, weights(iw_dos), edges)
i0       = findfirst(==(0.0), Es)
nz       = hist_dos.weights .> 0
logg_est = fill(NaN, length(Es))
logg_est[nz]  = log.(hist_dos.weights[nz])
logg_est[nz] .-= logg_est[i0]
writetsv("reweighting_dos.tsv", ("E", "logg_est", "logg_exact"),
         hcat(Es[nz], logg_est[nz], exact_logdos.values[nz]))

p3 = plot(Es, exact_logdos.values; lw = 3, color = :black, label = "exact log g(E)",
          xlabel = "E", ylabel = "log g(E)", title = "muca → density of states")
scatter!(p3, Es[nz], logg_est[nz]; ms = 3, color = 4, label = "reweighted")

# %% (interactive) overview
plot(p1, p2, p3; layout = (1, 3), size = (1200, 320), margin = 5Plots.mm)
