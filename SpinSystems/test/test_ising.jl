using Test
using Random
using SparseArrays
using Graphs
using MonteCarloX
using SpinSystems

@testset "Ising bookkeeping" begin
    sys = Ising([4, 4], J=1) 
    
    # default: all spins up
    # energy = -J * pairs = -32 for 4x4 lattice
    @test energy(sys) == -32
    @test energy(sys; full=true) == -32
    # magnetization = 16 for 16 spins all up
    @test magnetization(sys) == 16
    @test magnetization(sys; full=true) == 16

    # changing one spin to down changes 4 pair interactions, so delta_pair_interactions = 2*4 = 8
    delta_pair_interactions = 2 * local_pair_interactions(sys, 1)
    @test delta_pair_interactions == 8
    ΔE = delta_energy(sys, 1)
    @test ΔE == 8
    Δpair = -2 * local_pair_interactions(sys, 1)
    Δspin = -2 * sys.spins[1]
    @test Δpair == -8
    @test Δspin == -2

    modify!(sys, 1, Δpair, Δspin)
    @test energy(sys) == -24
    @test energy(sys; full=true) == energy(sys)
    @test magnetization(sys) == 14
    @test magnetization(sys; full=true) == magnetization(sys)
    @test sys.spins[1] == -1
end

@testset "Ising sparse J" begin
    J = spzeros(Float64, 4, 4)
    J[1, 2] = J[2, 1] = 1.0
    J[2, 3] = J[3, 2] = 2.0
    J[3, 4] = J[4, 3] = 3.0
    J[4, 1] = J[1, 4] = 4.0

    sys = Ising(J)

    @test energy(sys) == -10.0
    @test energy(sys; full=true) == -10.0
    @test magnetization(sys) == 4

    ΔE = delta_energy(sys, 1)
    @test ΔE == 10.0

    delta_pair_interactions = 2 * local_pair_interactions(sys, 1)
    @test ΔE == delta_pair_interactions
    Δpair = -2 * local_pair_interactions(sys, 1)
    Δspin = -2 * sys.spins[1]
    modify!(sys, 1, Δpair, Δspin)

    @test energy(sys) == 0.0
    @test energy(sys; full=true) == 0.0
    @test magnetization(sys) == 2

    J_bad = spzeros(Float64, 3, 3)
    J_bad[1, 2] = 1.0
    J_bad[2, 1] = 0.5
    @test_throws AssertionError Ising(J_bad)
end

@testset "Ising fields" begin
    sys = Ising([2, 2], J=1.0, h=0.5)
    pair_sum = sum(local_pair_interactions(sys, i) for i in eachindex(sys.spins)) / 2
    @test energy(sys) == -pair_sum - 0.5 * sum(sys.spins)
    @test energy(sys; full=true) == energy(sys)

    h_i = [1.0, -1.0, 0.5, 0.0]
    sys2 = Ising([2, 2], J=1.0, h=h_i)
    pair_sum2 = sum(local_pair_interactions(sys2, i) for i in eachindex(sys2.spins)) / 2
    @test energy(sys2) == -pair_sum2 - sum(h_i .* sys2.spins)
    @test energy(sys2; full=true) == energy(sys2)
end

@testset "Ising constructor variants" begin
    sys_g0 = Ising([2, 2], J=1)
    @test sys_g0 isa SpinSystems.IsingGraphCouplingNoField

    sys_gu = Ising([2, 2], J=1, h=0.2)
    @test sys_gu isa SpinSystems.IsingGraphCouplingUniformField

    sys_gv = Ising([2, 2], J=1, h=[0.1, -0.2, 0.3, 0.0])
    @test sys_gv isa SpinSystems.IsingGraphCouplingVectorField

    J = spzeros(Float64, 4, 4)
    J[1, 2] = J[2, 1] = 1.0
    J[2, 3] = J[3, 2] = 1.0
    J[3, 4] = J[4, 3] = 1.0
    J[4, 1] = J[1, 4] = 1.0

    sys_m0 = Ising(J)
    @test sys_m0 isa SpinSystems.IsingMatrixCouplingNoField

    sys_mu = Ising(J, h=0.2)
    @test sys_mu isa SpinSystems.IsingMatrixCouplingUniformField

    sys_mv = Ising(J, h=[0.1, -0.2, 0.3, 0.0])
    @test sys_mv isa SpinSystems.IsingMatrixCouplingVectorField

    sys_lat = IsingLatticeOptim(4, 4)
    @test sys_lat isa IsingLatticeOptim

    graph = Graphs.SimpleGraphs.grid([2, 2]; periodic=true)
    Jvec = collect(range(1.0, length=ne(graph)))
    sys_gsg = Ising(graph, Jvec)
    @test sys_gsg isa SpinSystems.IsingMatrixCouplingNoField

    sys_dsg = Ising([2, 2], J=Jvec, periodic=true)
    @test sys_dsg isa SpinSystems.IsingMatrixCouplingNoField

    @test_throws AssertionError Ising(graph, Jvec[1:end-1])
end

@testset "Ising delta_energy API consistency" begin
    # Graph variant
    sys_g = Ising([4, 4], J=1, h=0.1)
    i = 3
    Δpair = -2 * sys_g.J * local_pair_interactions(sys_g, i)
    Δspin = -2 * sys_g.spins[i]
    @test delta_energy(sys_g, i) == delta_energy(sys_g, Δpair, Δspin, i)

    # Matrix variant
    J = spzeros(Float64, 4, 4)
    J[1, 2] = J[2, 1] = 1.0
    J[2, 3] = J[3, 2] = 2.0
    J[3, 4] = J[4, 3] = 3.0
    J[4, 1] = J[1, 4] = 4.0
    sys_m = Ising(J, h=0.2)
    i = 2
    Δpair = -2 * local_pair_interactions(sys_m, i)
    Δspin = -2 * sys_m.spins[i]
    @test delta_energy(sys_m, i) == delta_energy(sys_m, Δpair, Δspin, i)

    # Optimized lattice variant
    sys_l = IsingLatticeOptim(4, 4)
    i = 5
    Δpair = -2 * local_pair_interactions(sys_l, i)
    Δspin = -2 * sys_l.spins[i]
    @test delta_energy(sys_l, i) == delta_energy(sys_l, Δpair, Δspin, i)
end

@testset "Ising Metropolis integration" begin
    rng = MersenneTwister(2024)
    sys = Ising([4, 4])
    init!(sys, :random, rng=rng)
    alg = Metropolis(rng, β=0.4)

    # test initial conditions
    @test acceptance_rate(alg) == 0.0

    # test single spin flip and energy consistency
    spin_flip!(sys, alg)
    @test alg.steps == 1
    @test 0.0 <= acceptance_rate(alg) <= 1.0
    @test energy(sys) == energy(sys; full=true)
    @test magnetization(sys) == magnetization(sys; full=true)

    # test multiple flips and energy consistency
    for _ in 1:50
        spin_flip!(sys, alg)
    end
    @test 0.0 <= acceptance_rate(alg) <= 1.0
    @test energy(sys) == energy(sys; full=true)
    @test magnetization(sys) == magnetization(sys; full=true)
end
