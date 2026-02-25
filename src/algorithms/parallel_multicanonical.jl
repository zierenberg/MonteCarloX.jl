using MPI
# ParallelMulticanonical: only handles MPI metadata and communication
mutable struct ParallelMulticanonical <: AbstractAlgorithm
    comm::Any # MPI communicator
    rank::Int
    size::Int
    root::Int
end

function ParallelMulticanonical(comm; root::Int=0)
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    0 <= root < size || throw(ArgumentError("`root` must satisfy 0 <= root < size"))
    return ParallelMulticanonical(comm, Int(rank), Int(size), Int(root))
end

@inline function is_root(pmuca::ParallelMulticanonical)
    return pmuca.rank == pmuca.root
end

# Merge local histogram across all ranks
function merge_histograms!(pmuca::ParallelMulticanonical, histogram)
    values = histogram.weights
    MPI.Allreduce!(values, MPI.SUM, pmuca.comm)
    histogram.weights .= values
end

# Distribute logweight from root to all ranks
function distribute_logweight!(pmuca::ParallelMulticanonical, logweight)
    # Only root rank has the updated logweight, broadcast to all ranks
    values = logweight.weights
    MPI.Bcast!(values, pmuca.root, pmuca.comm)
    logweight.weights .= values
    MPI.Barrier(pmuca.comm)
    return nothing
end
