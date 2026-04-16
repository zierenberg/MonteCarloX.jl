using MonteCarloX
using Test
using MPI

function test_parallel_backends()
    pass = true

    # ThreadsBackend
    tb = ThreadsBackend(4)
    pass &= check(rank(tb) == 0, "ThreadsBackend rank == 0\n")
    pass &= check(size(tb) == 4, "ThreadsBackend size == 4\n")
    pass &= check(is_root(tb), "ThreadsBackend is_root\n")

    tb_default = ThreadsBackend()
    pass &= check(size(tb_default) == Threads.nthreads(), "ThreadsBackend default size\n")

    # MPIBackend
    MPI.Initialized() || MPI.Init()
    mb = MPIBackend(MPI.COMM_WORLD)
    pass &= check(rank(mb) == MPI.Comm_rank(MPI.COMM_WORLD), "MPIBackend rank\n")
    pass &= check(size(mb) == MPI.Comm_size(MPI.COMM_WORLD), "MPIBackend size\n")
    pass &= check(mb.root == 0, "MPIBackend default root == 0\n")
    pass &= check(is_root(mb), "MPIBackend is_root (single process)\n")

    mb_root = MPIBackend(MPI.COMM_WORLD; root=0)
    pass &= check(mb_root.root == 0, "MPIBackend explicit root\n")

    # init convenience
    pass &= check(init(:MPI) isa MPIBackend, "init(:MPI)\n")
    pass &= check(init(:threads) isa ThreadsBackend, "init(:threads)\n")
    pass &= check(init(:threads; nthreads=2) isa ThreadsBackend, "init(:threads; nthreads=2)\n")
    pass &= check(size(init(:threads; nthreads=2)) == 2, "init(:threads) size\n")
    @test_throws ArgumentError init(:unknown_backend)

    # finalize! is safe to call on ThreadsBackend
    pass &= check(finalize!(tb) === nothing, "finalize! ThreadsBackend\n")

    return pass
end

@testset "Parallel backends" begin
    @test test_parallel_backends()
end
