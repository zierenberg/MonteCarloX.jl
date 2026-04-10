# # Parallel Tempering (MPI, full-power schedule) for 2D Ising
#
# Run with:
# mpiexec -n 8 julia --project=docs docs/src/examples/spin_systems/pt_Ising2D_mpi_fullpower.jl
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

function merge_samples_by_index(all_rank_samples::Vector{Vector{Vector{Float64}}}, nindices::Int)
    merged = [Float64[] for _ in 1:nindices]
    for rank_samples in all_rank_samples
        for i in 1:nindices
            append!(merged[i], rank_samples[i])
        end
    end
    return merged
end

function main()
    did_init = false
    if !MPI.Initialized()
        required = MPI.THREAD_FUNNELED
        provided = MPI.Init_thread(required)
        provided < required && error("MPI thread support insufficient: required THREAD_FUNNELED, got $(provided)")
        did_init = true
    end

    L = 8
    nmeasurements = 30_000
    ntherm_init = 2_000
    sweeps_between_exchange = 200

    # Fixed beta ladder (PRL strategy): acceptance is monitored, not dynamically retuned.
    Tmin = 1.5
    Tmax = 3.0

    # Local decorrelation schedule between exchange attempts.
    base_post_exchange_sweeps = 100
    min_post_exchange_sweeps = 1
    max_post_exchange_sweeps = 400

    # Re-estimate tau_int after this many exchange attempts.
    adapt_every_exchanges = 50
    tau_min_points = 400
    tau_lag_cap = 200

    seed = 42

    backend = MPIBackend(MPI.COMM_WORLD)
    pt = ParallelTempering(backend, root=0)
    nreplicas = size(pt)
    nreplicas >= 2 || throw(ArgumentError("MPI PT requires at least 2 ranks"))

    betas = set_betas(nreplicas, inv(Tmax), inv(Tmin), :uniform)
    beta0 = betas[rank(pt) + 1]

    sys = Ising([L, L])
    alg = Metropolis(MersenneTwister(seed + rank(pt)); β=beta0)
    init!(sys, :random; rng=alg.rng)

    for _ in 1:ntherm_init
        sweep_replica!(sys, alg, L)
    end

    local_samples = [Float64[] for _ in 1:nreplicas]
    sweeps_after_exchange = fill(base_post_exchange_sweeps, nreplicas)
    exchange_counter = 0

    for meas in 1:nmeasurements
        sweep_replica!(sys, alg, L)

        e = energy(sys)
        push!(local_samples[index(pt)], e)

        if meas % sweeps_between_exchange == 0
            update!(pt, alg, e)
            exchange_counter += 1

            idx = index(pt)
            @inbounds for _ in 1:sweeps_after_exchange[idx]
                sweep_replica!(sys, alg, L)
            end

            if exchange_counter % adapt_every_exchanges == 0
                all_rank_samples = gather(local_samples, pt.backend; root=pt.root)

                if is_root(pt)
                    merged = merge_samples_by_index(all_rank_samples, nreplicas)
                    taus = integrated_autocorrelation_times(merged;
                                                            min_points=tau_min_points,
                                                            max_lag=tau_lag_cap)

                    finite_taus = filter(isfinite, taus)
                    tau_ref = isempty(finite_taus) ? 1.0 : median(finite_taus)

                    @inbounds for i in eachindex(sweeps_after_exchange)
                        scale = isfinite(taus[i]) ? taus[i] / tau_ref : 1.0
                        target = round(Int, base_post_exchange_sweeps * scale)
                        sweeps_after_exchange[i] = clamp(target, min_post_exchange_sweeps, max_post_exchange_sweeps)
                    end
                end

                broadcast!(sweeps_after_exchange, pt.root, pt.backend)
            end
        end
    end

    exch = exchange_stats_at_root(pt)
    all_samples = gather(local_samples, pt.backend; root=pt.root)

    if is_root(pt)
        energy_samples = merge_samples_by_index(all_samples, nreplicas)
        taus_final = integrated_autocorrelation_times(energy_samples;
                                                      min_points=tau_min_points,
                                                      max_lag=tau_lag_cap)

        println("PT Ising2D MPI full-power finished")
        println("L = $(L), ranks = $(nreplicas), measurements = $(nmeasurements)")
        println("beta range = [$(round(minimum(betas), digits=4)), $(round(maximum(betas), digits=4))]")
        println("mean |m| per spin = $(round(mean_abs_m, digits=5))")

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
                    ", sweeps = ", sweeps_after_exchange[i],
                    ", samples = ", length(energy_samples[i]))
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

    if did_init
        finalize!(pt.backend)
    end
    return nothing
end

main()
