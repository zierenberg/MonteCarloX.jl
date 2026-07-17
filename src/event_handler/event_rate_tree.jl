# Fenwick-tree (binary-indexed tree) event handler.
#
# Rates are addressable like an array, but both updating a rate and sampling an event are
# O(log N) instead of the O(N) linear scan of the list-based handlers. This is what makes the
# n-fold way (rejection-free kinetic Monte Carlo) scale: after each event only a handful of rates
# change, and drawing the next event is a single tree descent. Plugs into the core KMC loop via
# the `next` / `next_event` overloads below.

# Add δ to position i and all its Fenwick ancestors.
@inline function fenwick_add!(tree::Vector{Float64}, i::Int, δ::Float64)
    n = length(tree)
    while i <= n
        @inbounds tree[i] += δ
        i += i & (-i)
    end
    return nothing
end

# Prefix sum rates[1] + … + rates[i].
@inline function fenwick_prefix(tree::Vector{Float64}, i::Int)
    acc = 0.0
    while i > 0
        @inbounds acc += tree[i]
        i -= i & (-i)
    end
    return acc
end

# First index whose prefix sum reaches x ∈ (0, total]; a single top-down descent.
function fenwick_sample(tree::Vector{Float64}, x::Float64)
    i = 0
    mask = prevpow(2, length(tree))
    while mask > 0
        k = i + mask
        if k <= length(tree) && (@inbounds tree[k]) < x
            x -= @inbounds tree[k]
            i = k
        end
        mask >>= 1
    end
    return i + 1
end

"""
    EventRateTree(n) <: AbstractEventHandlerRate{Int}

Fenwick-tree event handler over `n` events. Rates are read/written by index
(`handler[i]`, `handler[i] = r`) in O(log n), and `next`/`next_event` sample an event
proportional to the current rates in O(log n). Suited to kinetic Monte Carlo where an event
changes only a few rates per step (e.g. the n-fold way). `total_rate(handler)` returns ΣR.

Float drift in the tree accumulates additively over very long runs; rebuild the handler if exact
totals matter.
"""
mutable struct EventRateTree <: AbstractEventHandlerRate{Int}
    rates::Vector{Float64}
    tree::Vector{Float64}
end
EventRateTree(n::Int) = EventRateTree(zeros(n), zeros(n))

Base.length(handler::EventRateTree) = length(handler.rates)
Base.getindex(handler::EventRateTree, i::Int) = @inbounds handler.rates[i]
function Base.setindex!(handler::EventRateTree, r::Float64, i::Int)
    fenwick_add!(handler.tree, i, r - (@inbounds handler.rates[i]))
    @inbounds handler.rates[i] = r
    return r
end

"""
    total_rate(handler::EventRateTree)

Total event rate ΣR, read from the Fenwick tree root in O(1) amortized (one prefix sum).
"""
total_rate(handler::EventRateTree) = fenwick_prefix(handler.tree, length(handler.tree))

next_event(rng::AbstractRNG, handler::EventRateTree) =
    fenwick_sample(handler.tree, rand(rng) * total_rate(handler))
