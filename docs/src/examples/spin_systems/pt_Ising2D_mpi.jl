# # Parallel Tempering (MPI) for 2D Ising
#
# Run with:
# mpiexec -n 4 julia --project docs/src/examples/spin_systems/pt_Ising2D_mpi.jl
#
# One MPI rank hosts one replica. Exchanges always happen between
# neighboring temperatures in ladder space, while communication partners
# are chosen dynamically from temperature-label bookkeeping. Only scalar
# data are exchanged (beta/energy and acceptance flag), never full states.

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

@inline metropolis_accept_local(rng::AbstractRNG, log_ratio::Real) = (log_ratio >= 0) || (rand(rng) < exp(log_ratio))

function build_temperature_partner_plan(labels_by_rank::AbstractVector{<:Integer}, stage::Integer)
    nranks = length(labels_by_rank)
    partners = fill(Int32(-1), nranks)
    partner_labels = fill(Int32(0), nranks)

    rank_of_label = Vector{Int}(undef, nranks)
    @inbounds for r in 1:nranks
        lbl = labels_by_rank[r]
        rank_of_label[lbl] = r - 1
    end

    first_label = iseven(Int(stage)) ? 1 : 2
    @inbounds for l1 in first_label:2:(nranks - 1)
        l2 = l1 + 1
        r1 = rank_of_label[l1]
        r2 = rank_of_label[l2]
        partners[r1 + 1] = Int32(r2)
        partners[r2 + 1] = Int32(r1)
        partner_labels[r1 + 1] = Int32(l2)
        partner_labels[r2 + 1] = Int32(l1)
    end

    return partners, partner_labels
end

function attempt_exchange_with_partner!(pt::ParallelTempering,
                                        beta::Real,
                                        energy::Real,
                                        partner::Integer,
                                        stage::Integer;
                                        rng::AbstractRNG=Random.default_rng())
    partner < 0 && return (accepted=false, beta=beta, partner=-1, log_ratio=0.0)

    sendbuf = [float(beta), float(energy)]
    recvbuf = similar(sendbuf)
    tag_state = 100 + Int(stage)
    MPI.Sendrecv!(sendbuf, partner, tag_state, recvbuf, partner, tag_state, pt.comm)

    beta_partner = recvbuf[1]
    energy_partner = recvbuf[2]
    log_ratio = (beta - beta_partner) * (energy - energy_partner)

    tag_acc = 200 + Int(stage)
    if pt.rank < partner
        accepted = metropolis_accept_local(rng, log_ratio)
        MPI.Send([Int32(accepted)], partner, tag_acc, pt.comm)
    else
        recv_flag = Vector{Int32}(undef, 1)
        MPI.Recv!(recv_flag, partner, tag_acc, pt.comm)
        accepted = recv_flag[1] == 1
    end

    beta_new = accepted ? beta_partner : beta
    return (accepted=accepted, beta=beta_new, partner=Int(partner), log_ratio=log_ratio)
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
    rank = pt.rank
    nranks = pt.size
    nranks >= 2 || throw(ArgumentError("MPI PT requires at least 2 ranks"))

    betas = set_betas(nranks, inv(Tmax), inv(Tmin), :uniform)
    beta0 = betas[rank + 1]

    sys = IsingLatticeOptim(L, L)
    alg = Metropolis(MersenneTwister(seed + rank + 1); β=beta0)
    init!(sys, :random; rng=alg.rng)

    for _ in 1:ntherm_init
        sweep_replica!(sys, alg, L)
    end

    local_samples = [Float64[] for _ in 1:nranks]
    m_abs_sum_local = 0.0
    attempts_local = zeros(Int, nranks - 1)
    accepts_local = zeros(Int, nranks - 1)
    local_label = rank + 1

    for meas in 1:nmeasurements
        sweep_replica!(sys, alg, L)

        e = energy(sys)
        m = magnetization(sys)
        m_abs_sum_local += abs(m)
        push!(local_samples[local_label], e)

        if meas % sweeps_between_exchange == 0
            stage = pt.stage
            labels_by_rank = MPI.gather(local_label, pt.comm; root=pt.root)
            partners = fill(Int32(-1), nranks)
            partner_labels = fill(Int32(0), nranks)
            if is_root(pt)
                partners, partner_labels = build_temperature_partner_plan(labels_by_rank, stage)
            end
            MPI.Bcast!(partners, pt.root, pt.comm)
            MPI.Bcast!(partner_labels, pt.root, pt.comm)

            partner = Int(partners[rank + 1])
            partner_label = Int(partner_labels[rank + 1])
            label_before = local_label

            out = attempt_exchange_with_partner!(pt, inverse_temperature(alg), e, partner, stage; rng=alg.rng)
            if out.accepted
                set_inverse_temperature!(alg, out.beta)
                local_label = partner_label
            end

            if partner >= 0 && label_before < partner_label
                attempts_local[label_before] += 1
                accepts_local[label_before] += out.accepted
            end

            pt.stage = 1 - pt.stage
            for _ in 1:ntherm_after_exchange
                sweep_replica!(sys, alg, L)
            end
        end
    end

    attempts_total = MPI.Reduce(attempts_local, +, pt.root, pt.comm)
    accepts_total = MPI.Reduce(accepts_local, +, pt.root, pt.comm)
    mag_abs_total = MPI.Reduce(m_abs_sum_local, +, pt.root, pt.comm)
    all_samples = MPI.gather(local_samples, pt.comm; root=pt.root)

    if is_root(pt)
        energy_samples = [Float64[] for _ in 1:nranks]
        for rank_samples in all_samples
            for i in 1:nranks
                append!(energy_samples[i], rank_samples[i])
            end
        end

        rates = zeros(Float64, length(attempts_total))
        for i in eachindex(rates)
            rates[i] = attempts_total[i] > 0 ? accepts_total[i] / attempts_total[i] : 0.0
        end

        mean_abs_m = mag_abs_total / (nmeasurements * nranks * L * L)

        println("PT Ising2D MPI finished")
        println("L = $(L), ranks = $(nranks)")
        println("beta range = [$(round(minimum(betas), digits=4)), $(round(maximum(betas), digits=4))]")
        println("mean |m| per spin = $(round(mean_abs_m, digits=4))")
        println("mean exchange acceptance = $(round(mean(rates), digits=4))")

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
