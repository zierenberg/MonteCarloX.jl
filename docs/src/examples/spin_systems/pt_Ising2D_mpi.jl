# # Parallel Tempering (MPI) for 2D Ising
#
# Run with:
# mpiexec -n 4 julia --project docs/src/examples/spin_systems/pt_Ising2D_mpi.jl
#
# One MPI rank hosts one replica. `pt` owns a temperature-label table
# (`pt.labels`) so every rank always knows who its neighbors are without
# a gather/broadcast. Only scalar data are exchanged per swap; no system
# state is transferred. The swap observable here is the energy `E`, passed
# as `update!(pt, alg, E)`.

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

function main()
    did_init = false
    if !MPI.Initialized()
        required = MPI.THREAD_FUNNELED
        provided = MPI.Init_thread(required)
        provided < required && error("MPI thread support insufficient: required THREAD_FUNNELED, got $(provided)")
        did_init = true
    end

    L = 8
    nmeasurements = 20_000
    ntherm_init = 2_000
    ntherm_after_exchange = 20
    sweeps_between_exchange = 200
    Tmin = 1.5
    Tmax = 3.0
    seed = 42

    pt = ParallelTempering(MPI.COMM_WORLD, root=0)
    pt.size >= 2 || throw(ArgumentError("MPI PT requires at least 2 ranks"))

    betas = set_betas(pt.size, inv(Tmax), inv(Tmin), :uniform)
    beta0 = betas[pt.rank + 1]

    sys = IsingLatticeOptim(L, L)
    alg = Metropolis(MersenneTwister(seed + pt.rank); β=beta0)
    init!(sys, :random; rng=alg.rng)

    for _ in 1:ntherm_init
        sweep_replica!(sys, alg, L)
    end

    local_samples = [Float64[] for _ in 1:pt.size]
    m_abs_sum_local = 0.0

    for meas in 1:nmeasurements
        sweep_replica!(sys, alg, L)

        e = energy(sys)
        m_abs_sum_local += abs(magnetization(sys))
        push!(local_samples[index(pt)], e)

        if meas % sweeps_between_exchange == 0
            update!(pt, alg, e)
            for _ in 1:ntherm_after_exchange
                sweep_replica!(sys, alg, L)
            end
        end
    end

    steps_total = MPI.Reduce(pt.steps, +, pt.root, pt.comm)
    accepted_total = MPI.Reduce(pt.accepted, +, pt.root, pt.comm)
    mag_abs_total = MPI.Reduce(m_abs_sum_local, +, pt.root, pt.comm)
    all_samples = MPI.gather(local_samples, pt.comm; root=pt.root)

    if is_root(pt)
        energy_samples = [Float64[] for _ in 1:pt.size]
        for rank_samples in all_samples
            for i in 1:pt.size  
                append!(energy_samples[i], rank_samples[i])
            end
        end

        rates = zeros(Float64, length(steps_total))
        for i in eachindex(rates)
            rates[i] = steps_total[i] > 0 ? accepted_total[i] / steps_total[i] : 0.0
        end

        mean_abs_m = mag_abs_total / (nmeasurements * pt.size * L * L)

        println("PT Ising2D MPI finished")
        println("L = $(L), ranks = $(pt.size), measurements = $(nmeasurements)")
        println("beta range = [$(round(minimum(betas), digits=4)), $(round(maximum(betas), digits=4))]")
        println("mean |m| per spin = $(round(mean_abs_m, digits=4))")
        total_steps = sum(steps_total)
        total_acceptance = total_steps > 0 ? sum(accepted_total) / total_steps : 0.0
        println("mean exchange acceptance = $(round(total_acceptance, digits=4))")

        for (i, β) in enumerate(betas)
            println("beta[$i] = ", round(β, digits=4),
                " (T = ", round(inv(β), digits=4), "), samples = ", length(energy_samples[i]))
        end

        println("\nMean energy comparison (measured vs exact):")
        for (i, β) in enumerate(betas)
            samples = energy_samples[i]
            dist_exact = distribution_exact_ising2D(L, β)
            E_exact = get_centers(dist_exact)
            mean_exact = sum(E_exact .* dist_exact.values)
            mean_measured = isempty(samples) ? NaN : mean(samples)
            delta_mean = mean_measured - mean_exact
            println("  beta[$i] = $(round(β, digits=4)): ",
                    "<E>_meas = $(round(mean_measured, digits=3)), ",
                    "<E>_exact = $(round(mean_exact, digits=3)), ",
                    "Δ = $(round(delta_mean, digits=3))")
        end
    end

    MPI.Barrier(pt.comm)
    if did_init && !MPI.Finalized()
        MPI.Finalize()
    end
    return nothing
end

main()
