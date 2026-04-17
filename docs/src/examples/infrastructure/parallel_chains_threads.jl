# %% #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src
include(joinpath(@__DIR__, "..", "defaults.jl"))    #src

# # Parallel Chains (Threads)
#
# Run parallel Markov chains with a thread-based backend.
# Launch with:
# ```bash
# julia --threads=4 --project docs/src/examples/infrastructure/parallel_chains_threads.jl
# ```
#
# With only a single chain this fails due to replica-exchange requirements nthreads>=2;
#
# For the MPI version (HPC clusters), see `parallel_chains_mpi.jl`.
#
# Key differences from the MPI version (kept visible on purpose):
# - Operates explicitly on vector of `alg` and vector of `sys`;
# - Data is stored in shared memory and can be accessed by threads;
# - Parallelization needs to be actively enabled via `with_parallel` or `Threads.@threads`;
# - MonteCarloX API on parallel chain uses vectors of objects directly in shared memory

using Random, Statistics, StatsBase, Plots, ProgressMeter
using MonteCarloX

# ## Setup
#
# We sample a double-well potential ``E(x) = (x^2 - 1)^2`` with minima at
# ``x = \pm 1``.  At low temperature a single chain gets trapped in one
# well; parallel tempering enables barrier crossings.

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

n_samples = CI_MODE ? 50 : 50_000
n_burn_in = CI_MODE ? 10 : 500
n_sweep   = CI_MODE ? 10 : 100
n_iter    = CI_MODE ? 2 : 6

backend  = ThreadsBackend()
beta_ref = 10.0
betas_pt = set_betas(size(backend), 0.1, beta_ref, :geometric)

# ## Independent parallel chains
#
# Each chain samples the same target independently. At high beta (low
# temperature) each chain gets trapped in whichever well it starts near.
# Merging samples from all chains does NOT fix this — you just get a
# biased mixture of trapped chains.

pc_algs = [Metropolis(Xoshiro(1000 + i); β=beta_ref) for i in 1:size(backend)]
pc      = ParallelChains(backend, pc_algs)
pc_sys  = [System(0.0) for i in 1:size(backend)]

# shared memory storage: each thread writes to its own row, no synchronization needed.
pc_xs = zeros(Float64, size(backend), n_samples)
pc_Es = zeros(Float64, size(backend), n_samples)

time = @elapsed with_parallel(pc) do i, alg
    sys = pc_sys[i]
    for _ in 1:n_burn_in
        for _ in 1:n_sweep; update!(sys, alg); end
    end
    reset!(alg)
    for j in 1:n_samples
        for _ in 1:n_sweep; update!(sys, alg); end
        pc_xs[i, j] = sys.x
        pc_Es[i, j] = E(sys)
    end
end

# No need to merge samples, we have shared memory
# 
#

on_root(pc) do 
    println("## Independent chains (beta=$beta_ref) took $(round(time; digits=2)) seconds")
    for i in 1:size(pc)
        println("  chain $i: mean_x=$(round(mean(pc_xs[i,:]); digits=3)), mean_E=$(round(mean(pc_Es[i,:]); digits=3))")
    end
    println("  merged:  mean_x=$(round(mean(pc_xs); digits=3)), mean_E=$(round(mean(pc_Es); digits=3))")
end

# ## Parallel tempering
#
# Same potential, same number of samples, but now replicas at different
# temperatures exchange configurations.  High-temperature replicas cross
# the barrier freely; exchanges propagate this to the cold replica,
# producing correct bimodal sampling even at high beta.

pt_algs = [Metropolis(Xoshiro(2000 + i); β=betas_pt[i]) for i in 1:size(backend)]
pt      = ParallelTempering(backend, pt_algs)
pt_sys  = [System(0.0) for _ in 1:size(pt)]

# global storage
pt_xs = zeros(Float64, size(pt), n_samples)
pt_Es = zeros(Float64, size(pt), n_samples)

exchange_after_sample = 100
time = @elapsed for s in 1:Int(n_samples / exchange_after_sample)
    with_parallel(pt) do i, alg
        sys = pt_sys[i]
        for _ in 1:n_burn_in
            for _ in 1:n_sweep; update!(sys, alg); end
        end
        reset!(alg)
        for j in 1:exchange_after_sample
            for _ in 1:n_sweep; update!(sys, alg); end
            pt_xs[index(pt, i), (s-1)*exchange_after_sample + j] = sys.x
            pt_Es[index(pt, i), (s-1)*exchange_after_sample + j] = E(sys)
        end
    end
    # replica exchange gets full energy array, here we are no longer parallel
    MonteCarloX.update!(pt, E.(pt_sys))
end

# No need to gather samples, we have shared memory











on_root(pt) do _
    println("\n## Parallel Tempering (betas = $(round.(betas_pt; digits=2))) took $(round(time; digits=2)) seconds")
    println("Exchange acceptance rates: ", round.(acceptance_rates(pt); digits=3))
    for (i, b) in enumerate(betas_pt)
        s = pt_xs[i, :]
        println("  beta=$(round(b; digits=2)): $(length(s)) samples, mean=$(round(mean(s); digits=3)), std=$(round(std(s); digits=3))")
    end
end

# ## Multicanonical sampling with parallel chains
#
# Each chain runs a multicanonical sampler; after each iteration we merge
# histograms to root, refine the logweight on root, and distribute the
# new logweight back to all chains.
pmuca_algs = [Multicanonical(Xoshiro(3000 + i), Ebins) for i in 1:size(backend)]
pmuca      = ParallelMulticanonical(backend, pmuca_algs)
pmuca_sys  = [System(0.0) for i in 1:size(backend)]

# Initialize logweights with low temperature so chains stay in-range and
# the first histogram merge has non-zero counts.
beta_bound = beta_ref * 2
on_root(pmuca) do i
    alg = algorithm(pmuca, i)
    set!(logweight(alg), E -> -beta_bound * E)
end
distribute_logweight!(pmuca)

E_left  = 0.0
E_right = 3.0

prog = is_root(pmuca) ? Progress(n_iter; desc="pmuca iter") : nothing
time_iter = @elapsed for iter in 1:n_iter
    Threads.@threads for i in 1:size(pmuca)
        alg = algorithm(pmuca, i)
        sys = pmuca_sys[i]
        for _ in 1:n_burn_in
            for _ in 1:n_sweep; update!(sys, alg, delta=0.05); end
        end
        reset!(alg)
        for j in 1:Int(round(n_samples / n_iter * iter))
            for _ in 1:n_sweep; update!(sys, alg, delta=0.05); end
        end
    end
    merge_histograms!(pmuca)
    on_root(pmuca) do i
        alg = algorithm(pmuca, i)
        MonteCarloX.update!(ensemble(alg); mode=:simple)
        set!(logweight(alg), (E_right, E_max),
             E -> logweight(alg)(E_right) - beta_bound * (E - E_right))
        next!(prog)
    end
    distribute_logweight!(pmuca)
end

# Production run with final weights.
pmuca_xs = zeros(Float64, size(pmuca), n_samples)
pmuca_Es = zeros(Float64, size(pmuca), n_samples)
time_prod = @elapsed with_parallel(pmuca) do i, alg
    sys = pmuca_sys[i]
    for j in 1:n_samples
        for _ in 1:n_sweep; update!(sys, alg, delta=0.05); end
        pmuca_xs[i, j] = sys.x
        pmuca_Es[i, j] = E(sys)
    end
end

# No need to gather samples, we have shared memory.
pmuca_xs = vec(pmuca_xs)
pmuca_Es = vec(pmuca_Es)

on_root(pmuca) do 
    println("\n## Parallel Multicanonical took $(round(time_iter; digits=2)) seconds for iterations and $(round(time_prod; digits=2)) seconds for production runs")
    println("Acceptance rates (per chain): ",
            round.([acceptance_rate(algorithm(pmuca, i)) for i in 1:size(pmuca)]; digits=2))
end



# ## Comparison plot
#
# Independent chains at high beta get trapped in one well.
# Parallel tempering at the same beta recovers the bimodal distribution.
# The plotting is wrapped in a function so all `c, w` assignments live
# in hard scope (no soft-scope warnings).

using StatsBase, LinearAlgebra

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

idx_cold = findmin(abs.(betas_pt .- beta_ref))[2]
p = make_comparison_plot(;
    pt_cold_xs  = pt_xs[idx_cold, :],
    pt_cold_Es  = pt_Es[idx_cold, :],
    pc_chain_xs = [pc_xs[i, :] for i in 1:size(pc)],
    pc_chain_Es = [pc_Es[i, :] for i in 1:size(pc)],
    pmuca_xs    = pmuca_xs,
    pmuca_Es    = pmuca_Es,
    lw_muca     = logweight(algorithm(pmuca, 1)),
    beta_ref    = beta_ref,
    xbins       = xbins,
    Ebins       = Ebins)

savefig(p, "parallel_chains_threads.png")