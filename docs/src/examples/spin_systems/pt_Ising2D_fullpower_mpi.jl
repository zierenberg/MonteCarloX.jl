# # Parallel Tempering (MPI, full-power schedule) for 2D Ising
#
# Run with:
# mpiexec -n 8 julia --project=docs docs/src/examples/spin_systems/pt_Ising2D_fullpower_mpi.jl
#
# This example follows the practical strategy from
# Bittner, Nussbaumer, Janke (PRL 101, 130603, 2008):
#
# 1) keep a fixed beta ladder,
# 2) monitor exchange acceptance (target around 50%),
# 3) adapt local decorrelation effort between exchanges using tau_int.
#
# Here, each rank hosts one replica. After each exchange attempt, the
# replica performs an index-dependent number of local sweeps. That sweep
# budget is periodically recomputed from measured energy autocorrelation
# times per ladder index.

# %% #src
import Pkg
Pkg.activate(joinpath(@__FILE__, "../../../../"))
Pkg.instantiate()
include(joinpath(@__DIR__, "..", "defaults.jl"))

using Random, Statistics, StatsBase
using MonteCarloX, SpinSystems, MPI

function sweep_replica!(sys, alg, L)
    for _ in 1:(L * L)
        spin_flip!(sys, alg)
    end
    return nothing
end

function merge_samples_by_index(all_rank_samples::Vector{<:AbstractVector{<:Tuple{<:Integer,<:Real}}}, nindices::Int)
    merged = [Float64[] for _ in 1:nindices]
    for rank_samples in all_rank_samples
        for (idx, e) in rank_samples
            1 <= idx <= nindices || throw(ArgumentError("sample index $idx out of bounds for $nindices ladders"))
            push!(merged[idx], float(e))
        end
    end
    return merged
end

function main()
    L = 8
    nmeasurements = 20_000
    ntherm_init = 2_000

    # Fixed beta ladder (PRL strategy): acceptance is monitored, not dynamically retuned.
    Tmin = 1.5
    Tmax = 3.0

    # Local decorrelation schedule between exchange attempts.
    base_post_exchange_sweeps = 100
    min_post_exchange_sweeps = 10
    max_post_exchange_sweeps = 400

    # Re-estimate tau_int after this many exchange attempts.
    adapt_every_exchanges = 50
    tau_min_points = 400
    tau_lag_cap = 200

    seed = 42

    backend = init(:MPI)
    nreplicas = size(backend)
    nreplicas >= 2 || throw(ArgumentError("MPI PT requires at least 2 ranks"))

    betas = set_betas(nreplicas, inv(Tmax), inv(Tmin), :uniform)
    pt = ParallelTempering(betas; seed=seed, rng=MersenneTwister, backend=backend)

    sys = Ising([L, L])
    init!(sys, :random; rng=pt.alg.rng)

    if is_root(pt)
        println("════════════════════════════════════════")
        println(" PT Ising2D MPI full-power (L = $(L))  ")
        println(" Ranks = $(nreplicas), Measurements = $(nmeasurements)")
        println("════════════════════════════════════════")
    end

    for _ in 1:ntherm_init
        sweep_replica!(sys, pt.alg, L)
    end

    local_samples = Tuple{Int,Float64}[]
    sweeps_after_exchange = fill(base_post_exchange_sweeps, nreplicas)
    next_exchange_interval = base_post_exchange_sweeps
    sweeps_since_exchange = 0
    exchange_counter = 0

    for meas in 1:nmeasurements
        sweep_replica!(sys, pt.alg, L)
        e = energy(sys)
        push!(local_samples, (index(pt), e))

        sweeps_since_exchange += 1
        if sweeps_since_exchange >= next_exchange_interval
            update!(pt, e)
            exchange_counter += 1
            sweeps_since_exchange = 0

            if exchange_counter % adapt_every_exchanges == 0
                optimize_exchange_interval!(
                    pt,
                    local_samples,
                    sweeps_after_exchange;
                    base_sweeps=base_post_exchange_sweeps,
                    min_sweeps=min_post_exchange_sweeps,
                    max_sweeps=max_post_exchange_sweeps,
                    min_points=tau_min_points,
                    max_lag=tau_lag_cap,
                )

                # Propose a rank-local interval from the current ladder index,
                # then synchronize to one global interval via max-reduction.
                interval_buf = [sweeps_after_exchange[index(pt)]]
                MonteCarloX.allreduce!(interval_buf, max, pt)
                next_exchange_interval = only(interval_buf)
            end
        end
    end

    exch = exchange_stats_at_root(pt)
    all_samples = gather(local_samples, backend; root=pt.root)

    if is_root(pt)
        energy_samples = merge_samples_by_index(all_samples, nreplicas)
        taus_final = integrated_autocorrelation_times(energy_samples;
                                                      min_points=tau_min_points,
                                                      max_lag=tau_lag_cap)

        println("PT Ising2D MPI full-power finished")
        println("L = $(L), ranks = $(nreplicas), measurements = $(nmeasurements), exchanges = $(exchange_counter)")
        println("beta range = [$(round(minimum(betas), digits=4)), $(round(maximum(betas), digits=4))]")

        println("\nExchange acceptance by neighboring betas:")
        for i in eachindex(exch.rates)
            println("  (", round(betas[i], digits=4), ", ", round(betas[i + 1], digits=4), ")",
                " -> ", round(exch.rates[i], digits=4))
        end

        println("\nFinal tau_int(E) and post-exchange sweep schedule by beta index:")
        for i in 1:nreplicas
            tau_print = isfinite(taus_final[i]) ? round(taus_final[i], digits=3) : NaN
            println("  beta[$i] = ", round(betas[i], digits=4),
                    ": tau_int(E) = ", tau_print,
                    ", sweeps (last estimate) = ", sweeps_after_exchange[i],
                    ", samples (total) = ", length(energy_samples[i]))
        end

        println("\nMean energy comparison (measured vs exact):")
        for (i, β) in enumerate(betas)
            samples = energy_samples[i]
            dist_exact = distribution_exact_ising2D(L, β)
            E_exact = get_centers(dist_exact)
            mean_exact = sum(E_exact .* dist_exact.values)
            mean_measured = isempty(samples) ? NaN : mean(samples)
            delta_mean = abs(mean_measured - mean_exact)
            println("  beta[$i] = $(round(β, digits=4)): ",
                    "<E>_meas = $(round(mean_measured, digits=3)), ",
                    "<E>_exact = $(round(mean_exact, digits=3)), ",
                    "Δ = $(round(delta_mean, digits=4))")
        end
    end

    finalize!(backend)
    return nothing
end

main()
