"""
    clusters(positions, L, r_cluster) -> Vector{Int}

Cluster sizes from grouping particles by geometric distance. Two particles are
in the same cluster if their minimum-image distance is less than `r_cluster`.
Uses union-find with path compression and union by rank.
Returned sorted in descending order.
"""
function clusters(positions::Vector{SVector{D,T}}, L, r_cluster) where {D,T}
    N = length(positions)
    N == 0 && return Int[]
    r_cluster_sq = r_cluster^2
    box = L isa PeriodicBox ? L : PeriodicBox{D}(L)

    parent = collect(1:N)
    rnk = zeros(Int, N)

    @inbounds for i in 1:N-1
        for j in i+1:N
            r_sq = minimum_image_sq(positions[i], positions[j], box)
            r_sq < r_cluster_sq && _union!(parent, rnk, i, j)
        end
    end

    sizes = zeros(Int, N)
    for i in 1:N
        sizes[_find!(parent, i)] += 1
    end
    return sort!(filter(!iszero, sizes); rev=true)
end

@inline function _find!(parent, x)
    @inbounds while parent[x] != x
        parent[x] = parent[parent[x]]
        x = parent[x]
    end
    return x
end

@inline function _union!(parent, rnk, a, b)
    @inbounds begin
        ra, rb = _find!(parent, a), _find!(parent, b)
        ra == rb && return
        if rnk[ra] < rnk[rb]; parent[ra] = rb
        elseif rnk[ra] > rnk[rb]; parent[rb] = ra
        else; parent[rb] = ra; rnk[ra] += 1
        end
    end
end

largest_cluster_size(c::Vector{Int}) = isempty(c) ? 0 : first(c)
second_largest_cluster_size(c::Vector{Int}) = length(c) < 2 ? 0 : c[2]

function cluster_size_distribution(c::Vector{Int})
    dist = Dict{Int,Int}()
    for s in c
        dist[s] = get(dist, s, 0) + 1
    end
    return dist
end
