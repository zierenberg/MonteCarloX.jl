using MonteCarloX
using Test
using MPI
using Distributed

struct DummyMessageBackend <: AbstractMessageBackend end

function test_message_backend_api()
    pass = true

    # unimplemented backend throws ArgumentError
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
    pass &= check(rank(mpi_backend) == MPI.Comm_rank(MPI.COMM_WORLD), "MPI rank\n")
    pass &= check(size(mpi_backend) == MPI.Comm_size(MPI.COMM_WORLD), "MPI size\n")
    pass &= check(rank(MPI.COMM_WORLD) == rank(mpi_backend), "rank from comm\n")
    pass &= check(size(MPI.COMM_WORLD) == size(mpi_backend), "size from comm\n")

    gathered_mpi = allgather(rank(mpi_backend), mpi_backend)
    pass &= check(length(gathered_mpi) == size(mpi_backend), "MPI allgather length\n")

    reduced_mpi = [1]
    allreduce!(reduced_mpi, +, mpi_backend)
    pass &= check(reduced_mpi[1] == size(mpi_backend), "MPI allreduce\n")

    reduced_root = reduce([1], +, 0, mpi_backend)
    pass &= check(reduced_root[1] == size(mpi_backend), "MPI reduce\n")

    bcast_val = [rank(mpi_backend) == 0 ? 7 : 0]
    MonteCarloX.bcast!(bcast_val, 0, mpi_backend)
    pass &= check(bcast_val[1] == 7, "MPI bcast\n")

    gathered_at_root = gather(rank(mpi_backend), mpi_backend; root=0)
    if rank(mpi_backend) == 0
        pass &= check(length(gathered_at_root) == size(mpi_backend), "MPI gather at root\n")
    end

    barrier(mpi_backend)

    # Distributed backend (single-process safe checks)
    dist_backend = DistributedBackend(; addprocs=0)
    pass &= check(size(dist_backend) >= 1, "Distributed size >= 1\n")
    pass &= check(rank(dist_backend) == Distributed.myid() - 1, "Distributed rank\n")

    packet = (x=1,)
    pass &= check(exchange_packet(dist_backend, packet, rank(dist_backend), 0, true) == packet, "Distributed exchange active\n")
    pass &= check(exchange_packet(dist_backend, packet, rank(dist_backend), 0, false) == packet, "Distributed exchange inactive\n")

    gathered_dist = allgather(3, dist_backend)
    pass &= check(length(gathered_dist) == size(dist_backend), "Distributed allgather\n")

    reduced_dist = [1]
    allreduce!(reduced_dist, +, dist_backend)
    pass &= check(reduced_dist[1] == size(dist_backend), "Distributed allreduce\n")

    reduced_dist_root = reduce([2], +, 0, dist_backend)
    pass &= check(reduced_dist_root[1] == 2 * size(dist_backend), "Distributed reduce\n")

    bcast_dist = [rank(dist_backend) == 0 ? 9 : 0]
    MonteCarloX.bcast!(bcast_dist, 0, dist_backend)
    pass &= check(bcast_dist[1] == 9, "Distributed bcast\n")

    gathered_dist_root = gather(rank(dist_backend), dist_backend; root=0)
    pass &= check(length(gathered_dist_root) == size(dist_backend), "Distributed gather\n")

    barrier(dist_backend)

    # mode-based constructors
    pass &= check(init(:MPI) isa MPIBackend, "init(:MPI)\n")
    pass &= check(init(:Distributed; addprocs=0) isa DistributedBackend, "init(:Distributed)\n")

    return pass
end

@testset "Message backend" begin
    @test test_message_backend_api()
end
