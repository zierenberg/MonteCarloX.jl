using Test
using Random
using SparseArrays
using Graphs
using MonteCarloX
using SpinSystems

@testset "Ising bookkeeping" begin
    sys = Ising([4, 4])

    # default: all spins up
    @test energy(sys) == -32
    @test energy(sys; full=true) == -32
    @test magnetization(sys) == 16
    @test magnetization(sys; full=true) == 16

    # flipping spin 1: changes 4 pair interactions
    lpi = local_pair_interactions(sys, 1)
    @test lpi == 4
    ΔE = delta_energy(sys, 1)
    @test ΔE == 8

    modify!(sys, 1, SpinSystems.delta_sys(sys, 1, Int8(-sys.spins[1])))
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

    modify!(sys, 1, SpinSystems.delta_sys(sys, 1, Int8(-sys.spins[1])))
    @test energy(sys) == 0.0
    @test energy(sys; full=true) == 0.0
    @test magnetization(sys) == 2

    J_bad = spzeros(Float64, 3, 3)
    J_bad[1, 2] = 1.0
    J_bad[2, 1] = 0.5
    @test_throws AssertionError Ising(J_bad)
end

@testset "Ising fields" begin
    sys = Ising([2, 2]; J=1.0, h=0.5)
    pair_sum = sum(local_pair_interactions(sys, i) for i in eachindex(sys.spins)) / 2
    @test energy(sys) == -pair_sum - 0.5 * sum(sys.spins)
    @test energy(sys; full=true) == energy(sys)

    h_i = [1.0, -1.0, 0.5, 0.0]
    sys2 = Ising([2, 2]; J=1.0, h=h_i)
    pair_sum2 = sum(local_pair_interactions(sys2, i) for i in eachindex(sys2.spins)) / 2
    @test energy(sys2) == -pair_sum2 - sum(h_i .* sys2.spins)
    @test energy(sys2; full=true) == energy(sys2)
end

@testset "Ising constructor variants" begin
    # Lattice (periodic, uniform J)
    sys_lat = Ising([2, 2])
    @test sys_lat isa IsingLattice

    # Lattice with uniform field
    sys_lh = Ising([2, 2]; h=0.2)
    @test sys_lh isa IsingLattice

    # Lattice with vector field
    sys_lv = Ising([2, 2]; h=[0.1, -0.2, 0.3, 0.0])
    @test sys_lv isa IsingLattice

    # Graph (non-periodic)
    sys_g = Ising([2, 2]; periodic=false)
    @test sys_g isa IsingGraph

    # Matrix (sparse J)
    J = spzeros(Float64, 4, 4)
    J[1, 2] = J[2, 1] = 1.0
    J[2, 3] = J[3, 2] = 1.0
    J[3, 4] = J[4, 3] = 1.0
    J[4, 1] = J[1, 4] = 1.0

    sys_m = Ising(J)
    @test sys_m isa IsingMatrix

    # Graph with explicit graph
    graph = Graphs.SimpleGraphs.grid([2, 2]; periodic=true)
    sys_gg = Ising(graph, 1.0)
    @test sys_gg isa IsingGraph

    # Vector J -> IsingMatrix
    Jvec = collect(range(1.0, length=ne(graph)))
    sys_gsg = Ising(graph, Jvec)
    @test sys_gsg isa IsingMatrix

    @test_throws AssertionError Ising(graph, Jvec[1:end-1])
end

@testset "Ising delta_energy with precomputed lpi" begin
    sys = Ising([4, 4]; h=0.1)
    i = 3
    lpi = local_pair_interactions(sys, i)
    @test delta_energy(sys, i) == delta_energy(sys, i, lpi)

    J = spzeros(Float64, 4, 4)
    J[1, 2] = J[2, 1] = 1.0
    J[2, 3] = J[3, 2] = 2.0
    J[3, 4] = J[4, 3] = 3.0
    J[4, 1] = J[1, 4] = 4.0
    sys_m = Ising(J; h=0.2)
    i = 2
    lpi_m = local_pair_interactions(sys_m, i)
    @test delta_energy(sys_m, i) == delta_energy(sys_m, i, lpi_m)
end

@testset "Ising Metropolis integration" begin
    rng = MersenneTwister(2024)
    sys = Ising([4, 4])
    init!(sys, :random, rng=rng)
    alg = Metropolis(rng; β=0.4)

    @test acceptance_rate(alg) == 0.0

    spin_flip!(sys, alg)
    @test alg.steps == 1
    @test 0.0 <= acceptance_rate(alg) <= 1.0
    @test energy(sys) == energy(sys; full=true)
    @test magnetization(sys) == magnetization(sys; full=true)

    for _ in 1:50
        spin_flip!(sys, alg)
    end
    @test 0.0 <= acceptance_rate(alg) <= 1.0
    @test energy(sys) == energy(sys; full=true)
    @test magnetization(sys) == magnetization(sys; full=true)
end

@testset "Exact 2D Ising logDOS API" begin
    binned = logdos_exact_ising2D(L=8)
    @test binned isa BinnedObject
    @test binned[-128] == log(2)
    @test binned[0] ≈ 42.41274640460084
    @test isnan(binned[-124])

    vec = logdos_exact_ising2D(8; format=:vector)
    @test vec isa Vector{Tuple{Int,Float64}}
    @test first(vec) == (-128, log(2))
    @test last(vec) == (128, log(2))

    dict = logdos_exact_ising2D(8; format=:dict)
    @test dict[0] ≈ 42.41274640460084

    @test_throws ErrorException logdos_exact_ising2D(L=10)
    @test_throws ErrorException logdos_exact_ising2D(8; format=:unknown)
end
