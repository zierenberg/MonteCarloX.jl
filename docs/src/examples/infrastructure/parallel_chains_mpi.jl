# %% #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src
include(joinpath(@__DIR__, "..", "defaults.jl"))    #src

# # Parallel Chains (MPI)
#
# Run parallel Markov chains with an MPI backend (one rank per chain).
# Launch with:
# ```bash
# mpiexec -n 4 julia --project docs/src/examples/infrastructure/parallel_chains_mpi.jl
# ```
#
# This example is the MPI counterpart to `parallel_chains_threads.jl` and
# mirrors it section-by-section so you can see where the two backends
# share an API and where MPI's distributed memory forces asymmetries.
#
# Key differences from the threads version (kept visible on purpose):
#
# - Each rank owns exactly one `alg` and one `sys`; there is no array of
#   replicas on a single process. `algorithm(pc)` returns *this rank's*
#   algorithm, not a vector.
# - Data arrays are **rank-local**. To print global statistics or draw
#   plots we must gather/reduce explicitly (`merge!(...)`, `MPI.gather`,
#   `MPI.Reduce!`).
# - Anything that is "chain-1-only" in the threads example becomes a
#   `is_root(backend)` guard here (e.g. the `update!(ensemble; ...)` step
#   in the multicanonical refinement).
# - `update!(pt, ...)` takes a *scalar* local energy on MPI, but a vector
#   of energies (one per chain) on threads.

using Random, Statistics, StatsBase, Plots
using MPI
using MonteCarloX

# ## Setup
#
# Identical physics to the threads example: double-well potential with
# minima at ``x = \pm 1``.

const CI_MODE = get(ENV, "MCX_SMOKE", get(ENV, "MCX_CI", "false")) == "true"

mutable struct System
    x::Float64
end

E(x::Real) = 1/4*(x^2 - 2)^2
E(sys::System) = E(sys.x)

function update!(sys::System, alg::AbstractImportanceSampling; delta=0.1)
    x_new = sys.x + delta * randn(alg.rng)
    accept!(alg, E(x_new), E(sys)) && (sys.x = x_new)
end

xbins = -3:0.01:3
E_min = floor(minimum(E.(xbins)))
E_max = ceil(maximum(E.(xbins)))
Ebins = E_min:0.005:E_max

# ## Parameters

n_samples = CI_MODE ? 50 : 100_000
n_burn_in = CI_MODE ? 10 : 500
n_sweep   = CI_MODE ? 10 : 100
n_iter    = CI_MODE ? 2 : 10

backend  = init(:MPI)
beta_ref = 10.0
betas_pt = set_betas(size(backend), 0.1, beta_ref, :geometric)

# ## Independent parallel chains
#
# Each rank samples the same target independently. At high beta each
# chain gets trapped in whichever well it starts near — merging does NOT
# fix this, only diagnoses it.

pc_alg = Metropolis(Xoshiro(1000 + rank(backend) + 1); β=beta_ref)
pc     = ParallelChains(backend, pc_alg)
pc_sys = System(0.0)

# Rank-local storage: no cross-chain matrix exists in MPI memory.
pc_xs_local = zeros(Float64, n_samples)
pc_Es_local = zeros(Float64, n_samples)

let alg = algorithm(pc), sys = pc_sys
    for _ in 1:n_burn_in
        for _ in 1:n_sweep; update!(sys, alg); end
    end
    reset!(alg)
    for j in 1:n_samples
        for _ in 1:n_sweep; update!(sys, alg); end
        pc_xs_local[j] = sys.x
        pc_Es_local[j] = E(sys)
    end
end

# Per-chain statistics: compute local mean/std on each rank, gather to
# root for a per-chain printout. The threads example could read
# `pc_xs[i,:]` directly from any thread; we cannot.
pc_local_stats = [mean(pc_xs_local), std(pc_Es_local)]
pc_all_stats   = MPI.gather(pc_local_stats, backend.comm; root=backend.root)

# Global mean/std via an allreduce-style merge on (sum, sum², n).
pc_agg_local = [sum(pc_xs_local), sum(abs2, pc_xs_local), Float64(n_samples)]
pc_agg_total = MonteCarloX.merge!(copy(pc_agg_local), +, pc)
pc_mean_glob = pc_agg_total[1] / pc_agg_total[3]
pc_std_glob  = sqrt(pc_agg_total[2] / pc_agg_total[3] - pc_mean_glob^2)

if is_root(pc)
    println("## Independent chains (beta=$beta_ref)")
    for (i, s) in enumerate(pc_all_stats)
        println("  chain $i: mean=$(round(s[1]; digits=3)), std=$(round(s[2]; digits=3))")
    end
    println("  merged:  mean=$(round(pc_mean_glob; digits=3)), std=$(round(pc_std_glob; digits=3))")
end

# ## Parallel tempering
#
# One replica per rank at a different beta; replicas exchange
# configurations after local sweeps. Because the ladder permutation
# changes over time, each rank's samples cover many ladder indices — we
# store them in a ragged per-ladder vector and gather to root for plots.

pt_alg = Metropolis(Xoshiro(2000 + rank(backend) + 1); β=betas_pt[rank(backend) + 1])
pt     = ParallelTempering(backend, pt_alg)
pt_sys = System(0.0)

# Ragged per-ladder storage on each rank: samples contributed by this
# rank while it occupied each ladder position.
pt_xs_local = [Float64[] for _ in 1:size(pt)]
pt_Es_local = [Float64[] for _ in 1:size(pt)]

exchange_after_sample = 100
for s in 1:Int(n_samples / exchange_after_sample)
    alg = algorithm(pt)
    sys = pt_sys
    for _ in 1:n_burn_in
        for _ in 1:n_sweep; update!(sys, alg); end
    end
    reset!(alg)
    for j in 1:exchange_after_sample
        for _ in 1:n_sweep; update!(sys, alg); end
        push!(pt_xs_local[index(pt)], sys.x)
        push!(pt_Es_local[index(pt)], E(sys))
    end
    # NOTE API asymmetry: threads passes E.(pt_sys) (a vector, one entry
    # per replica on the shared-memory side); MPI passes the rank-local
    # scalar energy E(sys). Both are "the local energy(ies)", but the
    # signatures differ.
    MonteCarloX.update!(pt, E(sys))
end

# Gather ragged per-ladder samples to root. One `MPI.gather` per ladder
# position; could be compressed with a custom serialization, but kept
# explicit for clarity.
function gather_per_index(local_per_ladder, bk::MPIBackend)
    n = length(local_per_ladder)
    out = is_root(bk) ? [Float64[] for _ in 1:n] : Float64[]
    for i in 1:n
        chunks = MPI.gather(local_per_ladder[i], bk.comm; root=bk.root)
        if is_root(bk)
            append!(out[i], vcat(chunks...))
        end
    end
    return out
end

pt_xs_root = gather_per_index(pt_xs_local, backend)
pt_Es_root = gather_per_index(pt_Es_local, backend)

# `acceptance_rates(pt)` already does an MPI.Reduce to root internally.
pt_acc = acceptance_rates(pt)

if is_root(pt)
    println("\n## Parallel Tempering (betas = $(round.(betas_pt; digits=2)))")
    println("Exchange acceptance rates: ", round.(pt_acc; digits=3))
    for (i, b) in enumerate(betas_pt)
        s = pt_xs_root[i]
        println("  beta=$(round(b; digits=2)): $(length(s)) samples, mean=$(round(mean(s); digits=3)), std=$(round(std(s); digits=3))")
    end
end

# ## Multicanonical sampling with parallel chains
#
# Each rank runs a multicanonical chain; after each iteration we merge
# histograms to root, refine the logweight on root, and broadcast the
# new logweight back to all ranks.
#
# This is exactly the pattern the threads version wants (the "TODO: get
# root for threads!" comment in parallel_chains_threads.jl), with `is_root`
# making the asymmetric step explicit.

pmuca_alg = Multicanonical(Xoshiro(3000 + rank(backend) + 1), Ebins)
pmuca     = ParallelMulticanonical(backend, pmuca_alg)
pmuca_sys = System(0.0)

# Initialize logweights with low temperature so chains stay in-range and
# the first histogram merge has non-zero counts.
beta_bound = beta_ref * 2
if is_root(pmuca)
    set!(logweight(algorithm(pmuca)), E -> -beta_bound * E)
end
distribute_logweight!(pmuca)

E_left  = 0.0
E_right = 3.0

using ProgressMeter
prog = is_root(pmuca) ? Progress(n_iter; desc="pmuca iter") : nothing
for iter in 1:n_iter
    alg = algorithm(pmuca)
    sys = pmuca_sys
    for _ in 1:n_burn_in
        for _ in 1:n_sweep; update!(sys, alg, delta=0.05); end
    end
    reset!(alg)
    for j in 1:Int(round(n_samples / n_iter * iter))
        for _ in 1:n_sweep; update!(sys, alg, delta=0.05); end
    end

    # Reduce histograms to root; refine logweight on root; broadcast.
    merge_histograms!(pmuca)
    if is_root(pmuca)
        MonteCarloX.update!(ensemble(alg); mode=:simple)
        set!(logweight(alg), (E_right, E_max),
             E -> logweight(alg)(E_right) - beta_bound * (E - E_right))
    end
    distribute_logweight!(pmuca)

    is_root(pmuca) && next!(prog)
end
is_root(pmuca) && finish!(prog)

# Production run with final weights: rank-local samples.
pmuca_xs_local = zeros(Float64, n_samples)
pmuca_Es_local = zeros(Float64, n_samples)
let alg = algorithm(pmuca), sys = pmuca_sys
    for j in 1:n_samples
        for _ in 1:n_sweep; update!(sys, alg, delta=0.05); end
        pmuca_xs_local[j] = sys.x
        pmuca_Es_local[j] = E(sys)
    end
end

# Gather all samples to root for reweighting / plotting.
# NOTE these gathers MUST be called on every rank: MPI.gather is a
# collective operation. Guarding with `is_root` would deadlock.
pmuca_xs_chunks = MPI.gather(pmuca_xs_local, backend.comm; root=backend.root)
pmuca_Es_chunks = MPI.gather(pmuca_Es_local, backend.comm; root=backend.root)
pc_xs_chunks    = MPI.gather(pc_xs_local,    backend.comm; root=backend.root)
pc_Es_chunks    = MPI.gather(pc_Es_local,    backend.comm; root=backend.root)
pmuca_xs = is_root(backend) ? vcat(pmuca_xs_chunks...) : Float64[]
pmuca_Es = is_root(backend) ? vcat(pmuca_Es_chunks...) : Float64[]

# Acceptance rate per rank (collective, matches threads' per-chain printout).
pmuca_acc_local = acceptance_rate(algorithm(pmuca))
pmuca_acc_all   = MPI.gather(pmuca_acc_local, backend.comm; root=backend.root)

if is_root(pmuca)
    println("\n## Parallel Multicanonical")
    println("Acceptance rates (per rank): ", round.(pmuca_acc_all; digits=2))
    # TODO: print metrics like roundtrips that still need to be implemented.
end

# ## Comparison plot (root only)
#
# Same comparison as in the threads example, using the same
# `make_comparison_plot` function (defined identically in both files).
# Everything after this point runs on rank 0 only.

using LinearAlgebra

function make_comparison_plot(;
        pt_cold_xs, pt_cold_Es,              # single-chain vectors at the cold beta
        pc_chain_xs, pc_chain_Es,            # Vector-of-Vector, one per chain
        pmuca_xs, pmuca_Es,                  # flat vectors (all samples)
        lw_muca, beta_ref, xbins, Ebins)
    col_true, col_pt, col_pc, col_muca = :black, :steelblue, :crimson, :forestgreen
    centers(e) = e[1:end-1] .+ step(e)/2
    xs_exact   = collect(-3:0.0001:3)
    n_pc       = length(pc_chain_xs)

    function pdf_hist(samples, bins; logw=nothing)
        h = logw === nothing ? fit(Histogram, samples, bins) :
            fit(Histogram, samples, Weights(exp.(logw .- maximum(logw))), bins)
        return centers(bins), normalize(h; mode=:pdf).weights
    end

    # panel for P(x)
    px = plot(xlabel="x", ylabel="P(x)", legend=false)
    c, w = pdf_hist(pt_cold_xs, xbins); plot!(px, c, w; color=col_pt, lw=5)
    for i in 1:n_pc
        c, w = pdf_hist(pc_chain_xs[i], xbins); plot!(px, c, w; color=col_pc, alpha=i/n_pc, lw=2)
    end
    c, w = pdf_hist(pmuca_xs, xbins); plot!(px, c, w; color=col_muca, lw=2)
    c, w = pdf_hist(pmuca_xs, xbins; logw=-beta_ref .* pmuca_Es .- lw_muca.(pmuca_Es))
    plot!(px, c, w; color=col_muca, lw=4)
    c, w = pdf_hist(xs_exact, xbins; logw=-beta_ref .* E.(xs_exact))
    plot!(px, c, w; color=col_true, lw=2)

    # panel for P(E)
    pE = plot(xlabel="E", ylabel="P(E)", legend=false)
    c, w = pdf_hist(pt_cold_Es, Ebins); plot!(pE, c, w; color=col_pt, lw=5)
    for i in 1:n_pc
        c, w = pdf_hist(pc_chain_Es[i], Ebins); plot!(pE, c, w; color=col_pc, alpha=i/n_pc, lw=2)
    end
    c, w = pdf_hist(pmuca_Es, Ebins); plot!(pE, c, w; color=col_muca, lw=2)
    c, w = pdf_hist(pmuca_Es, Ebins; logw=-beta_ref .* pmuca_Es .- lw_muca.(pmuca_Es))
    plot!(pE, c, w; color=col_muca, lw=5)
    c, w = pdf_hist(E.(xs_exact), Ebins; logw=-beta_ref .* E.(xs_exact))
    plot!(pE, c, w; color=col_true, lw=2)

    yaxis!(px, :log10); ylims!(px, (1e-5, Inf)); xlims!(pE, (-3,3))
    yaxis!(pE, :log10); ylims!(pE, (1e-5, Inf)); xlims!(pE, (0,2))

    pl = plot(; framestyle=:none, legend=:inside,
              foreground_color_legend=nothing, background_color_legend=nothing)
    for (lab, col, lw, ls) in [
        ("True", col_true, 5, :solid),
        ("Parallel Tempering", col_pt, 5, :solid),
        ("Independent chains", col_pc, 2, :solid),
        ("Multicanonical", col_muca, 2, :solid),
        ("Muca (reweighted)", col_muca, 2, :dash)]
        plot!(pl, Float64[], Float64[]; label=lab, color=col, lw=lw, ls=ls)
    end

    return plot(px, pl, pE; layout=grid(1, 3, widths=[0.45, 0.1, 0.45]),
                size=(1100, 380), margins=5Plots.mm)
end

if is_root(backend)
    idx_cold = findmin(abs.(betas_pt .- beta_ref))[2]
    p = make_comparison_plot(;
        pt_cold_xs  = pt_xs_root[idx_cold],
        pt_cold_Es  = pt_Es_root[idx_cold],
        pc_chain_xs = pc_xs_chunks,
        pc_chain_Es = pc_Es_chunks,
        pmuca_xs    = pmuca_xs,
        pmuca_Es    = pmuca_Es,
        lw_muca     = logweight(algorithm(pmuca)),
        beta_ref    = beta_ref,
        xbins       = xbins,
        Ebins       = Ebins)

    savefig(p, "parallel_chains_mpi.png")
end

finalize!(backend)
