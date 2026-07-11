# SiteEvents generator: rejection-free sampling (NFoldRates) and intrinsic dynamics through
# the ordinary KMC loop. Reference: a non-interacting paramagnet σᵢ ∈ {−1,+1} with E = h Σσ
# has the exact time-averaged magnetization ⟨σ⟩ = −tanh(βh).

mutable struct ToyParamagnet <: AbstractSystem
    spins::Vector{Int8}
    h::Float64
end

MonteCarloX.nsites(sys::ToyParamagnet) = length(sys.spins)
MonteCarloX.local_states(sys::ToyParamagnet, i) = (Int8(-sys.spins[i]),)
MonteCarloX.delta_energy(sys::ToyParamagnet, i, s_new) = sys.h * (Int(s_new) - Int(sys.spins[i]))
MonteCarloX.modify!(sys::ToyParamagnet, i::Int, s_new::Int8) = (sys.spins[i] = s_new; nothing)
MonteCarloX.partners(::ToyParamagnet, i) = ()

@testset "SiteEvents + NFoldRates (n-fold way)" begin
    β, h, N = 0.7, 0.8, 64

    # advance!: the generator is an ordinary event source; observe! sees the pre-jump state.
    sys = ToyParamagnet(ones(Int8, N), h)
    src = SiteEvents(sys, NFoldRates(β=β))
    alg = Gillespie(MersenneTwister(1))
    advance!(alg, src, 100.0)                                        # thermalize
    acc = zeros(2)                                                   # Σ m·Δt, t_prev
    advance!(alg, src, 5_000.0;
             observe! = (s, event, t) -> (acc[1] += (t - acc[2]) * sum(Int, sys.spins);
                                          acc[2] = t; nothing))
    @test abs(acc[1] / acc[2] / N - (-tanh(β * h))) < 0.02

    # Manual loop: the standard KMC pair, state untouched between step! and modify!.
    sys2 = ToyParamagnet(ones(Int8, N), h)
    src2 = SiteEvents(sys2, NFoldRates(BoltzmannEnsemble(β=β); balance=GlauberBalance()))
    alg2 = Gillespie(MersenneTwister(2))
    m_dt, t_prev, events_ok = 0.0, 0.0, true
    for _ in 1:200_000
        t, event = step!(alg2, src2)
        m_dt += (t - t_prev) * sum(Int, sys2.spins)                  # state still pre-jump here
        t_prev = t
        events_ok &= event isa Tuple{Int,Int8}
        modify!(src2, event, t)
    end
    @test events_ok
    @test abs(m_dt / alg2.time / N - (-tanh(β * h))) < 0.05
    @test alg2.steps == 200_000

    # Intrinsic rate rule: a plain callable — uniform flip rate μ regardless of state gives
    # exact mean waiting time 1/(μN) and relaxes the magnetization to zero.
    μ = 0.5
    sys3 = ToyParamagnet(ones(Int8, N), h)
    src3 = SiteEvents(sys3, (s, i, x) -> μ)
    @test total_rate(src3) ≈ μ * N
    alg3 = Gillespie(MersenneTwister(3))
    accm = zeros(2)
    advance!(alg3, src3, 2_000.0;
             observe! = (s, event, t) -> (accm[1] += (t - accm[2]) * sum(Int, sys3.spins);
                                          accm[2] = t; nothing))
    @test abs(accm[1] / accm[2] / N) < 0.02
    @test abs(alg3.time / alg3.steps - 1 / (μ * N)) < 0.001          # exact Poisson clock

    # Nonlinear ensembles are refused: local rates need logweight linear in ΔE.
    @test_throws ArgumentError NFoldRates(FunctionEnsemble(x -> -x^2))
end

mutable struct MMInfinityQueue <: AbstractSystem
    N::Int
    λ::Float64
    μ::Float64
end

MonteCarloX.nreactions(::MMInfinityQueue) = 2
MonteCarloX.modify!(sys::MMInfinityQueue, r::Int, t) =
    (r == 1 ? (sys.N += 1) : (sys.N -= 1); nothing)

@testset "ReactionEvents (M/M/∞ queue)" begin
    λ, μ = 2.0, 0.25                                                 # stationary ⟨N⟩ = λ/μ = 8
    sys = MMInfinityQueue(0, λ, μ)
    src = ReactionEvents(sys, (s, r) -> r == 1 ? s.λ : s.μ * s.N)
    alg = Gillespie(MersenneTwister(7))
    advance!(alg, src, 50.0)                                         # thermalize
    acc = zeros(2)
    advance!(alg, src, 20_000.0;
             observe! = (s, event, t) -> (acc[1] += (t - acc[2]) * sys.N;
                                          acc[2] = t; nothing))
    @test abs(acc[1] / acc[2] - λ / μ) < 0.15
    @test alg.steps > 10_000

    # manual step!/modify! loop, and propensities follow the state
    t, event = step!(alg, src)
    @test event in (1, 2)
    modify!(src, event, t)
    @test total_rate(src) ≈ λ + μ * sys.N
end

@testset "resample! (heat bath over the site interface)" begin
    β, h, N = 0.7, 0.8, 64
    sys = ToyParamagnet(ones(Int8, N), h)
    alg = HeatBathAlgorithm(MersenneTwister(11); β=β)
    heat_bath_update!(alg, sys) = begin                              # the MCMC template:
        i = rand(alg.rng, 1:nsites(sys))                             # pick, decide, apply
        s_new = resample!(alg, sys, i)
        s_new === nothing || modify!(sys, i, s_new)
    end
    for _ in 1:20_000                                                # thermalize
        heat_bath_update!(alg, sys)
    end
    m_sum, nsamp = 0.0, 200_000
    for _ in 1:nsamp
        heat_bath_update!(alg, sys)
        m_sum += sum(Int, sys.spins)
    end
    @test abs(m_sum / nsamp / N - (-tanh(β * h))) < 0.02
    @test alg.steps == 220_000

    # decide-only contract: resample! returns the drawn state (or nothing) and does NOT modify
    s_before = copy(sys.spins)
    rets = [resample!(alg, sys, rand(alg.rng, 1:N)) for _ in 1:200]
    @test sys.spins == s_before
    @test all(r -> r === nothing || r isa Int8, rets)
    @test any(isnothing, rets) && any(!isnothing, rets)

    # the shift trick needs linearity: nonlinear ensembles are refused at construction
    @test_throws ArgumentError HeatBathAlgorithm(MersenneTwister(12), FunctionEnsemble(x -> -x^2))
end
