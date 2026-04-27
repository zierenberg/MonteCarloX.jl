"""
    flood_fill_clusters(occupation, nbrs) -> Vector{Vector{Int}}

Find connected components of occupied sites on a lattice using union-find.

# Arguments
- `occupation::Vector{Bool}` -- per-site occupation
- `nbrs::Vector{NTuple{6,Int}}` -- neighbor table

# Returns
Vector of clusters, each cluster is a vector of site indices.
"""
function flood_fill_clusters(occupation::Vector{Bool}, nbrs::Vector{NTuple{6,Int}})
    N = length(occupation)
    parent = collect(1:N)
    rank = zeros(Int, N)

    function find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        end
        return x
    end

    function union!(a, b)
        ra, rb = find(a), find(b)
        ra == rb && return
        if rank[ra] < rank[rb]
            parent[ra] = rb
        elseif rank[ra] > rank[rb]
            parent[rb] = ra
        else
            parent[rb] = ra
            rank[ra] += 1
        end
    end

    # Union occupied neighbors
    for site in 1:N
        occupation[site] || continue
        for nb in nbrs[site]
            if occupation[nb]
                union!(site, nb)
            end
        end
    end

    # Collect clusters
    cluster_map = Dict{Int, Vector{Int}}()
    for site in 1:N
        occupation[site] || continue
        root = find(site)
        if haskey(cluster_map, root)
            push!(cluster_map[root], site)
        else
            cluster_map[root] = [site]
        end
    end

    return collect(values(cluster_map))
end

"""
    largest_cluster_size(clusters) -> Int

Size of the largest cluster. Returns 0 if no clusters.
"""
function largest_cluster_size(clusters::Vector{Vector{Int}})
    isempty(clusters) && return 0
    return maximum(length, clusters)
end

"""
    second_largest_cluster_size(clusters) -> Int

Size of the second largest cluster. Returns 0 if fewer than 2 clusters.
"""
function second_largest_cluster_size(clusters::Vector{Vector{Int}})
    length(clusters) < 2 && return 0
    sizes = sort!([length(c) for c in clusters]; rev=true)
    return sizes[2]
end

"""
    cluster_size_distribution(clusters) -> Dict{Int, Int}

Histogram of cluster sizes: size => count.
"""
function cluster_size_distribution(clusters::Vector{Vector{Int}})
    dist = Dict{Int, Int}()
    for c in clusters
        s = length(c)
        dist[s] = get(dist, s, 0) + 1
    end
    return dist
end
