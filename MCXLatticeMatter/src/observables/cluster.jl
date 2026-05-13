"""
    clusters(sys) -> Vector{Int}

Cluster sizes (in monomers) from grouping polymers by inter-polymer
nearest-neighbor contacts. Uses union-find on polymer indices.
Returned sorted in descending order.
"""
function clusters(sys::LatticePolymer)
    N = num_polymers(sys)
    N == 0 && return Int[]

    # set up a tree of parents and ranks from merges of polymers into clusters
    parent = collect(1:N)
    rnk = zeros(Int, N)

    # Merge polymers connected by inter-polymer neighbor contacts
    @inbounds for n in 1:N
        for m in 1:polymer_length(sys, n)
            site = coords_to_site(sys.polymers[n][m], sys.dims)
            for nb in sys.neighbors[site]
                n2 = sys.state[nb]
                n2 != 0 && n2 != n && _union!(parent, rnk, n, n2)
            end
        end
    end

    # Collect cluster sizes in monomers
    sizes = zeros(Int, N)
    for n in 1:N
        sizes[_find!(parent, n)] += polymer_length(sys, n)
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
    dist = Dict{Int, Int}()
    for s in c
        dist[s] = get(dist, s, 0) + 1
    end
    return dist
end
