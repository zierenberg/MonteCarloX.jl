"""
    Parallel backends for MonteCarloX.

Two backends determine how parallel algorithms are constructed:

- `ThreadsBackend`: shared-memory parallelism via Julia threads (single node).
- `MPIBackend`: distributed-memory parallelism via MPI (multi-node / HPC).

The backend is passed to parallel algorithm constructors (e.g. `ParallelChains`,
`ParallelMulticanonical`, `ParallelTempering`) to select the concrete variant.
"""

import Base: size, merge!
using MPI

function rank end

"""
    ThreadsBackend(nthreads::Int)
    ThreadsBackend()

Shared-memory backend. Defaults to `Threads.nthreads()` chains, but can be
overridden to run more (or fewer) chains than available threads.
"""
struct ThreadsBackend
    nthreads::Int
end

ThreadsBackend() = ThreadsBackend(Threads.nthreads())

@inline rank(::ThreadsBackend) = 0
@inline size(backend::ThreadsBackend) = backend.nthreads
@inline is_root(::ThreadsBackend) = true

"""
    MPIBackend(comm; root=0)
    MPIBackend(; comm=MPI.COMM_WORLD, root=0, required_thread_level=nothing)

MPI-backed parallelism. The keyword constructor initializes MPI on first use.

Parallel algorithm variants access `backend.comm` directly for MPI collective
operations (e.g. `MPI.Allreduce!`, `MPI.Bcast!`).
"""
struct MPIBackend{C}
    comm::C
    rank::Int
    size::Int
    root::Int
end

function MPIBackend(comm; root::Int=0)
    return MPIBackend{typeof(comm)}(comm, Int(MPI.Comm_rank(comm)), Int(MPI.Comm_size(comm)), root)
end

function MPIBackend(; comm=MPI.COMM_WORLD, root::Int=0, required_thread_level=nothing)
    if !MPI.Initialized()
        if required_thread_level === nothing
            MPI.Init()
        else
            provided = MPI.Init_thread(required_thread_level)
            provided < required_thread_level && throw(ArgumentError(
                "MPI thread support insufficient: required $(required_thread_level), got $(provided)"))
        end
    end
    return MPIBackend(comm; root=root)
end

@inline rank(backend::MPIBackend) = backend.rank
@inline size(backend::MPIBackend) = backend.size
@inline is_root(backend::MPIBackend) = (backend.rank == backend.root)

"""
    init(mode::Symbol; kwargs...)

Initialize and return a parallel backend.

Supported modes:
- `:MPI` / `:mpi` → `MPIBackend`
- `:threads` / `:Threads` → `ThreadsBackend`

Examples:
- `init(:MPI)`
- `init(:threads)`
- `init(:threads; nthreads=8)`
"""
function init(mode::Symbol; kwargs...)
    m = Symbol(lowercase(String(mode)))
    if m == :mpi
        return MPIBackend(; kwargs...)
    elseif m == :threads
        kw = (; kwargs...)
        n = get(kw, :nthreads, Threads.nthreads())
        return ThreadsBackend(n)
    else
        throw(ArgumentError("unknown backend mode $(mode); use :MPI or :threads"))
    end
end

"""
    finalize!(backend)

Tear down the communication layer. Safe to call multiple times.
"""
function finalize!(backend::MPIBackend)
    if MPI.Initialized() && !MPI.Finalized()
        MPI.Barrier(backend.comm)
        MPI.Finalize()
    end
    return nothing
end

finalize!(::ThreadsBackend) = nothing
