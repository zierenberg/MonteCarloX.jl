using MonteCarloX
using Test
using MPI
using Distributed

struct DummyMessageBackend <: AbstractMessageBackend end

function test_message_backend_api(; verbose=false)
    pass = true

    @test_throws ArgumentError rank(DummyMessageBackend())
    @test_throws ArgumentError size(DummyMessageBackend())
    @test_throws ArgumentError exchange_packet(DummyMessageBackend(), :packet, 0, 0, true)
    @test_throws ArgumentError allgather(1, DummyMessageBackend())
    @test_throws ArgumentError allreduce!([1], +, DummyMessageBackend())
    @test_throws ArgumentError reduce([1], +, 0, DummyMessageBackend())
    @test_throws ArgumentError MonteCarloX.bcast!([1], 0, DummyMessageBackend())
    @test_throws ArgumentError gather(1, DummyMessageBackend(); root=0)
    @test_throws ArgumentError barrier(DummyMessageBackend())

    @test_throws ArgumentError rank("not_a_comm")
    @test_throws ArgumentError size("not_a_comm")
    @test_throws ArgumentError init(:unknown_backend)

    # MPI backend (single-process safe checks)
    MPI.Initialized() || MPI.Init()
    mpi_backend = MPIBackend(MPI.COMM_WORLD)
    pass &= rank(mpi_backend) == MPI.Comm_rank(MPI.COMM_WORLD)
    pass &= size(mpi_backend) == MPI.Comm_size(MPI.COMM_WORLD)
    pass &= rank(MPI.COMM_WORLD) == rank(mpi_backend)
    pass &= size(MPI.COMM_WORLD) == size(mpi_backend)

    gathered_mpi = allgather(rank(mpi_backend), mpi_backend)
    pass &= length(gathered_mpi) == size(mpi_backend)

    reduced_mpi = [1]
    allreduce!(reduced_mpi, +, mpi_backend)
    pass &= reduced_mpi[1] == size(mpi_backend)

    reduced_root = reduce([1], +, 0, mpi_backend)
    pass &= reduced_root[1] == size(mpi_backend)

    bcast_val = [rank(mpi_backend) == 0 ? 7 : 0]
    MonteCarloX.bcast!(bcast_val, 0, mpi_backend)
    pass &= bcast_val[1] == 7

    gathered_at_root = gather(rank(mpi_backend), mpi_backend; root=0)
    if rank(mpi_backend) == 0
        pass &= length(gathered_at_root) == size(mpi_backend)
    end

    barrier(mpi_backend)

    # Distributed backend (single-process safe checks)
    dist_backend = DistributedBackend(; addprocs=0)
    pass &= size(dist_backend) >= 1
    pass &= rank(dist_backend) == Distributed.myid() - 1

    packet = (x=1,)
    pass &= exchange_packet(dist_backend, packet, rank(dist_backend), 0, true) == packet
    pass &= exchange_packet(dist_backend, packet, rank(dist_backend), 0, false) == packet

    gathered_dist = allgather(3, dist_backend)
    pass &= length(gathered_dist) == size(dist_backend)

    reduced_dist = [1]
    allreduce!(reduced_dist, +, dist_backend)
    pass &= reduced_dist[1] == size(dist_backend)

    reduced_dist_root = reduce([2], +, 0, dist_backend)
    pass &= reduced_dist_root[1] == 2 * size(dist_backend)

    bcast_dist = [rank(dist_backend) == 0 ? 9 : 0]
    MonteCarloX.bcast!(bcast_dist, 0, dist_backend)
    pass &= bcast_dist[1] == 9

    gathered_dist_root = gather(rank(dist_backend), dist_backend; root=0)
    pass &= length(gathered_dist_root) == size(dist_backend)

    barrier(dist_backend)

    # mode-based constructors
    pass &= init(:MPI) isa MPIBackend
    pass &= init(:Distributed; addprocs=0) isa DistributedBackend

    if verbose
        println("Message backend API test pass: $(pass)")
    end

    return pass
end

function run_message_backend_testsets(; verbose=false)
    @testset "Message backend" begin
        @test test_message_backend_api(verbose=verbose)
    end
    return true
end
