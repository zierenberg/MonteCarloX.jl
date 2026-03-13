using Test
using Random
using SparseArrays
using Graphs
using MonteCarloX
using SpinSystems

@testset "BlumeCapel bookkeeping" begin
    sys = BlumeCapel([4, 4], J=1.0, D=0.5)

    @test energy(sys) == -24.0
    @test energy(sys; full=true) == -24.0
    @test magnetization(sys) == 16
    @test magnetization(sys; full=true) == 16

    i = 1
    s_new = Int8(0)
    Δpair, Δspin, Δspin2 = SpinSystems.propose_changes(sys, i, s_new)
    @test Δpair == -4.0
    @test Δspin == -1
    @test Δspin2 == -1

    ΔE = delta_energy(sys, i, s_new)
    @test ΔE == 3.5

    E_old = energy(sys)
    modify!(sys, i, s_new, Δpair, Δspin, Δspin2)
    @test energy(sys) == E_old + ΔE
    @test energy(sys; full=true) == energy(sys)
    @test magnetization(sys) == 15
end

@testset "BlumeCapel initialization" begin
    rng = MersenneTwister(11)
    sys = BlumeCapel([4, 4], J=1.0, D=0.2)

    init!(sys, :down)
    @test all(==(Int8(-1)), sys.spins)
    @test energy(sys) == energy(sys; full=true)

    init!(sys, :zero)
    @test all(==(Int8(0)), sys.spins)
    @test magnetization(sys) == 0
    @test energy(sys) == energy(sys; full=true)

    init!(sys, :random, rng=rng)
    @test all(s in Int8[-1, 0, 1] for s in sys.spins)
    @test energy(sys) == energy(sys; full=true)
end

@testset "BlumeCapel constructor variants" begin
    sys_g0 = BlumeCapel([2, 2], J=1.0, D=0.3)
    @test sys_g0 isa SpinSystems.BlumeCapelGraphCouplingNoField

    sys_gu = BlumeCapel([2, 2], J=1.0, D=0.3, h=0.2)
    @test sys_gu isa SpinSystems.BlumeCapelGraphCouplingUniformField

    sys_gv = BlumeCapel([2, 2], J=1.0, D=0.3, h=[0.1, -0.2, 0.3, 0.0])
    @test sys_gv isa SpinSystems.BlumeCapelGraphCouplingVectorField

    J = spzeros(Float64, 4, 4)
    J[1, 2] = J[2, 1] = 1.0
    J[2, 3] = J[3, 2] = 2.0
    J[3, 4] = J[4, 3] = 3.0
    J[4, 1] = J[1, 4] = 4.0

    sys_m0 = BlumeCapel(J, 0.5)
    @test sys_m0 isa SpinSystems.BlumeCapelMatrixCouplingNoField

    sys_mu = BlumeCapel(J, 0.5, h=0.2)
    @test sys_mu isa SpinSystems.BlumeCapelMatrixCouplingUniformField

    sys_mv = BlumeCapel(J, 0.5, h=[0.1, -0.2, 0.3, 0.0])
    @test sys_mv isa SpinSystems.BlumeCapelMatrixCouplingVectorField

    graph = Graphs.SimpleGraphs.grid([2, 2]; periodic=true)
    Jvec = collect(range(1.0, length=ne(graph)))
    sys_gsg = BlumeCapel(graph, Jvec, 0.5)
    @test sys_gsg isa SpinSystems.BlumeCapelMatrixCouplingNoField

    sys_dsg = BlumeCapel([2, 2], J=Jvec, D=0.5, periodic=true)
    @test sys_dsg isa SpinSystems.BlumeCapelMatrixCouplingNoField

    @test_throws AssertionError BlumeCapel(graph, Jvec[1:end-1], 0.5)
end

@testset "BlumeCapel Metropolis and HeatBath integration" begin
    rng = MersenneTwister(2026)
    sys = BlumeCapel([4, 4], J=1.0, D=0.2, h=0.1)
    init!(sys, :random, rng=rng)

    alg_m = Metropolis(rng, β=0.4)
    for _ in 1:100
        spin_flip!(sys, alg_m)
    end
    @test 0.0 <= acceptance_rate(alg_m) <= 1.0
    @test energy(sys) == energy(sys; full=true)
    @test magnetization(sys) == magnetization(sys; full=true)

    alg_h = HeatBath(rng, β=0.4)
    for _ in 1:100
        spin_flip!(sys, alg_h)
    end
    @test alg_h.steps == 100
    @test energy(sys) == energy(sys; full=true)
    @test magnetization(sys) == magnetization(sys; full=true)
end
