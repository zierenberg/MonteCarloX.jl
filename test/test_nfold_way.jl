# n-fold way: rejection-free kinetic MC over the local-states protocol, driven by Gillespie.
# Reference: a non-interacting paramagnet σᵢ ∈ {−1,+1} with E = h Σσ has the exact
# time-averaged magnetization ⟨σ⟩ = −tanh(βh).

mutable struct ToyParamagnet
    spins::Vector{Int8}
    h::Float64
end

MonteCarloX.nsites(sys::ToyParamagnet) = length(sys.spins)
MonteCarloX.local_states(sys::ToyParamagnet, i) = (Int8(-sys.spins[i]),)
MonteCarloX.delta_energy(sys::ToyParamagnet, i, s_new) = sys.h * (Int(s_new) - Int(sys.spins[i]))
MonteCarloX.modify!(sys::ToyParamagnet, i::Int, s_new::Int8) = (sys.spins[i] = s_new; nothing)
MonteCarloX.partners(::ToyParamagnet, i) = ()

@testset "NFoldWay (rejection-free via Gillespie)" begin
    β, h, N = 0.7, 0.8, 64
    sys = ToyParamagnet(ones(Int8, N), h)
    nf = NFoldWay(sys, β)
    @test nf.n_alt == 1

    advance!(Gillespie(MersenneTwister(1)), nf, 100.0)               # thermalize
    alg = Gillespie(MersenneTwister(2))
    acc = zeros(2)                                                   # Σ m·Δt, t_prev
    advance!(alg, nf, 5_000.0;
             measure! = (n, event, t) -> (acc[1] += (t - acc[2]) * sum(Int, n.sys.spins);
                                          acc[2] = t; nothing))
    @test alg.steps > 1000
    @test abs(acc[1] / acc[2] / N - (-tanh(β * h))) < 0.02

    # Glauber balance is an equally valid rate choice: same stationary distribution.
    sysg = ToyParamagnet(ones(Int8, N), h)
    nfg = NFoldWay(sysg, β; balance=GlauberBalance())
    advance!(Gillespie(MersenneTwister(3)), nfg, 100.0)
    accg = zeros(2)
    advance!(Gillespie(MersenneTwister(4)), nfg, 5_000.0;
             measure! = (n, event, t) -> (accg[1] += (t - accg[2]) * sum(Int, n.sys.spins);
                                          accg[2] = t; nothing))
    @test abs(accg[1] / accg[2] / N - (-tanh(β * h))) < 0.02

    # Nonlinear ensembles are refused: local rates need logweight linear in ΔE.
    @test_throws ArgumentError NFoldWay(ToyParamagnet(ones(Int8, 4), h),
                                        FunctionEnsemble(x -> -x^2))
end
