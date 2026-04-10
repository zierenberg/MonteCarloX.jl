using MPI
using Distributed
import Base: size, reduce

"""
    AbstractMessageBackend

Backend interface for message-passing coordination in parallel algorithms.
"""
abstract type AbstractMessageBackend end

rank(::AbstractMessageBackend) = throw(ArgumentError("rank is not implemented for this message backend"))
size(::AbstractMessageBackend) = throw(ArgumentError("size is not implemented for this message backend"))
exchange_packet(::AbstractMessageBackend, packet, partner_rank::Int, tag::Integer, is_owner::Bool) =
    throw(ArgumentError("exchange_packet is not implemented for this message backend"))
allgather(value, ::AbstractMessageBackend) =
    throw(ArgumentError("allgather is not implemented for this message backend"))
allreduce!(values, op, ::AbstractMessageBackend) =
    throw(ArgumentError("allreduce! is not implemented for this message backend"))
reduce(values, op, root::Int, ::AbstractMessageBackend) =
    throw(ArgumentError("reduce is not implemented for this message backend"))
broadcast!(values, root::Int, ::AbstractMessageBackend) =
    throw(ArgumentError("broadcast! is not implemented for this message backend"))
gather(value, ::AbstractMessageBackend; root::Int) =
    throw(ArgumentError("gather is not implemented for this message backend"))
barrier(::AbstractMessageBackend) =
    throw(ArgumentError("barrier is not implemented for this message backend"))

# Generic rank/size for MPI communicators
@inline rank(comm) = _safe_mpi_rank(comm)
@inline size(comm) = _safe_mpi_size(comm)

@inline function _safe_mpi_rank(comm)
    try
        return Int(MPI.Comm_rank(comm))
    catch
        throw(ArgumentError("rank requires MPI.Comm or a MessageBackend. Ensure MPI is available."))
    end
end

@inline function _safe_mpi_size(comm)
    try
        return Int(MPI.Comm_size(comm))
    catch
        throw(ArgumentError("size requires MPI.Comm or a MessageBackend. Ensure MPI is available."))
    end
end

"""
    MPIBackend(comm)
    MPIBackend(; comm=MPI.COMM_WORLD, required_thread_level=nothing)

MPI-backed message backend.

The keyword constructor initializes MPI on first use and then returns a backend
bound to `comm`.
"""
struct MPIBackend{C} <: AbstractMessageBackend
    comm::C
    rank::Int
    size::Int
end

function MPIBackend(comm)
    return MPIBackend{typeof(comm)}(comm, _safe_mpi_rank(comm), _safe_mpi_size(comm))
end

function MPIBackend(; comm=MPI.COMM_WORLD, required_thread_level=nothing)
    if !isdefined(@__MODULE__, :MPI)
        @warn "MPI backend unavailable in this session; skipping MPIBackend initialization"
        error("MPIBackend requires MPI module; install via: ] add MPI")
    end
    if !MPI.Initialized()
        if required_thread_level === nothing
            MPI.Init()
        else
            provided = MPI.Init_thread(required_thread_level)
            provided < required_thread_level && throw(ArgumentError("MPI thread support insufficient: required $(required_thread_level), got $(provided)"))
        end
    end
    return MPIBackend(comm)
end

@inline rank(backend::MPIBackend) = backend.rank
@inline size(backend::MPIBackend) = backend.size

function exchange_packet(backend::MPIBackend, packet, partner_rank::Int, tag::Integer, is_owner::Bool)
    if is_owner
        MPI.send(packet, backend.comm; dest=partner_rank, tag=tag)
        return MPI.recv(backend.comm; source=partner_rank, tag=tag)
    end

    recv_packet = MPI.recv(backend.comm; source=partner_rank, tag=tag)
    MPI.send(packet, backend.comm; dest=partner_rank, tag=tag)
    return recv_packet
end

function allgather(value, backend::MPIBackend)
    values = MPI.Allgather(value, backend.comm)
    return values
end

function allreduce!(values, op, backend::MPIBackend)
    return MPI.Allreduce!(values, op, backend.comm)
end

function reduce(values, op, root::Int, backend::MPIBackend)
    return MPI.Reduce(values, op, root, backend.comm)
end

function broadcast!(values, root::Int, backend::MPIBackend)
    return MPI.Bcast!(values, root, backend.comm)
end

function gather(value, backend::MPIBackend; root::Int)
    return MPI.gather(value, backend.comm; root=root)
end

function barrier(backend::MPIBackend)
    return MPI.Barrier(backend.comm)
end

"""
    DistributedBackend()
    DistributedBackend(; addprocs=0, exeflags=nothing)

Distributed.jl-backed message backend for Julia process-based parallelism.
Uses `Distributed.jl` for communication between multiple Julia processes.

The keyword constructor can add workers and ensures `MonteCarloX` is loaded on
workers so backend communication payloads deserialize correctly.
"""
struct DistributedBackend <: AbstractMessageBackend
    myid::Int
    nprocs::Int
end

@inline function _distributed_backend_state()
    return DistributedBackend(Distributed.myid(), Distributed.nprocs())
end

function DistributedBackend(; addprocs::Union{Nothing,Integer}=nothing, exeflags=nothing)
    if !isdefined(@__MODULE__, :Distributed)
        @warn "Distributed backend unavailable in this session; skipping DistributedBackend initialization"
        error("DistributedBackend requires Distributed module")
    end
    addprocs_n = addprocs === nothing ? 0 : Int(addprocs)
    launch_flags = exeflags
    if launch_flags === nothing
        active_project = Base.active_project()
        if active_project !== nothing
            launch_flags = "--project=$(dirname(active_project))"
        end
    end

    if addprocs_n > 0
        if launch_flags === nothing
            Distributed.addprocs(addprocs_n)
        else
            Distributed.addprocs(addprocs_n; exeflags=launch_flags)
        end
    end

    workers = Distributed.workers()
    if !isempty(workers)
        try
            Distributed.remotecall_eval(Main, workers, :(using MonteCarloX))
        catch err
            throw(ArgumentError("failed to load MonteCarloX on distributed workers; ensure workers use the active project (exeflags=$(repr(launch_flags))). Original error: $(err)"))
        end
    end

    return _distributed_backend_state()
end

@inline rank(backend::DistributedBackend) = backend.myid - 1  # 0-indexed rank
@inline size(backend::DistributedBackend) = backend.nprocs

function exchange_packet(backend::DistributedBackend, packet, partner_rank::Int, tag::Integer, is_owner::Bool)
    partner_id = partner_rank + 1
    if is_owner
        remote_packet = Distributed.remotecall_fetch(identity, partner_id, packet)
        return remote_packet
    else
        # Non-owner receives the packet from owner, holds onto it and returns it
        return packet
    end
end

function allgather(value, backend::DistributedBackend)
    # Gather value from all workers to all workers
    result = map(pid -> Distributed.remotecall_fetch(identity, pid, value), 1:backend.nprocs)
    return result
end

function allreduce!(values, op, backend::DistributedBackend)
    # Gather values from all workers, reduce, and scatter back
    all_values = Distributed.reduce(
        op,
        map(pid -> Distributed.remotecall_fetch(copy, pid, values), 1:backend.nprocs)
    )
    values .= all_values
    return values
end

function reduce(values, op, root::Int, backend::DistributedBackend)
    # Gather all values to root, reduce, return only relevant data
    root_id = root + 1
    
    # All workers gather their values to root
    gathered = Distributed.remotecall_fetch(root_id) do
        local_values = copy(values)
        for pid in setdiff(1:backend.nprocs, root_id)
            worker_values = Distributed.remotecall_fetch(identity, pid, values)
            local_values = map(op, local_values, worker_values)
        end
        return local_values
    end
    
    return gathered
end

function broadcast!(values, root::Int, backend::DistributedBackend)
    # Send values from root to all workers
    root_id = root + 1
    
    broadcast_values = Distributed.remotecall_fetch(identity, root_id, values)
    for pid in setdiff(1:backend.nprocs, root_id)
        Distributed.remotecall_fetch(pid) do
            return copy(broadcast_values)
        end
    end
    
    values .= broadcast_values
    return values
end

function gather(value, backend::DistributedBackend; root::Int)
    # Gather values from all workers to root worker
    root_id = root + 1
    
    gathered = Distributed.remotecall_fetch(root_id) do
        result = [copy(value)]
        for pid in setdiff(1:backend.nprocs, root_id)
            push!(result, Distributed.remotecall_fetch(identity, pid, value))
        end
        return result
    end
    
    return gathered
end

function barrier(backend::DistributedBackend)
    # Simple synchronization: all workers send a signal to root and wait for ack
    root_id = 1  # Use process 1 as a simple barrier coordinator
    
    Distributed.remotecall_fetch(root_id) do
        for pid in setdiff(1:backend.nprocs, root_id)
            Distributed.remotecall_fetch(identity, pid, nothing)
        end
    end
    
    return nothing
end

"""
    init(mode::Symbol; kwargs...)

Initialize and return a message backend.

Supported modes:
- `:MPI` / `:mpi`
- `:Distributed` / `:distributed`

Examples:
- `init(:MPI)`
- `init(:Distributed; addprocs=4)`
"""
function init(mode::Symbol; kwargs...)
    m = Symbol(lowercase(String(mode)))

    if m == :mpi
        return MPIBackend(; kwargs...)
    elseif m == :distributed
        backend_kwargs = (; kwargs...)
        if !haskey(backend_kwargs, :addprocs)
            backend_kwargs = merge((addprocs=0,), backend_kwargs)
        end
        return DistributedBackend(; backend_kwargs...)
    else
        throw(ArgumentError("unknown backend mode $(mode); use :MPI or :Distributed"))
    end
end

"""
    finalize!(backend::AbstractMessageBackend)

Tear down the communication layer associated with `backend`.

- `MPIBackend`: synchronizes ranks with `MPI.Barrier` and then calls
  `MPI.Finalize()` if MPI is initialized and not yet finalized.
- `DistributedBackend`: removes all worker processes via `rmprocs`.

Safe to call multiple times; subsequent calls are no-ops.
"""
function finalize!(backend::MPIBackend)
    isdefined(@__MODULE__, :MPI) || return nothing
    if MPI.Initialized() && !MPI.Finalized()
        barrier(backend)
        MPI.Finalize()
    end
    return nothing
end

function finalize!(backend::DistributedBackend)
    isdefined(@__MODULE__, :Distributed) || return nothing
    workers = Distributed.workers()
    isempty(workers) || Distributed.rmprocs(workers)
    return nothing
end