"""
    geometric_clusters(positions, L, r_cluster) -> Vector{Vector{Int}}

Find clusters of particles based on geometric distance. Two particles are
in the same cluster if their minimum-image distance is less than `r_cluster`.

Uses union-find for O(N² α(N)) complexity.
"""
function geometric_clusters(positions::Vector{SVector{3,T}}, L, r_cluster) where T
    N = length(positions)
    r_cluster_sq = r_cluster^2

    parent = collect(1:N)
    rnk = zeros(Int, N)

    function find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end

    function union!(a, b)
        ra, rb = find(a), find(b)
        ra == rb && return
        if rnk[ra] < rnk[rb]
            parent[ra] = rb
        elseif rnk[ra] > rnk[rb]
            parent[rb] = ra
        else
            parent[rb] = ra
            rnk[ra] += 1
        end
    end

    for i in 1:N-1
        for j in i+1:N
            r_sq = minimum_image_sq(positions[i], positions[j], L)
            if r_sq < r_cluster_sq
                union!(i, j)
            end
        end
    end

    cluster_map = Dict{Int, Vector{Int}}()
    for i in 1:N
        root = find(i)
        if haskey(cluster_map, root)
            push!(cluster_map[root], i)
        else
            cluster_map[root] = [i]
        end
    end
    return collect(values(cluster_map))
end

"""
    largest_cluster_size(clusters) -> Int
"""
function largest_cluster_size(clusters::Vector{Vector{Int}})
    isempty(clusters) && return 0
    return maximum(length, clusters)
end

"""
    second_largest_cluster_size(clusters) -> Int
"""
function second_largest_cluster_size(clusters::Vector{Vector{Int}})
    length(clusters) < 2 && return 0
    sizes = sort!([length(c) for c in clusters]; rev=true)
    return sizes[2]
end

"""
    cluster_size_distribution(clusters) -> Dict{Int, Int}
"""
function cluster_size_distribution(clusters::Vector{Vector{Int}})
    dist = Dict{Int, Int}()
    for c in clusters
        s = length(c)
        dist[s] = get(dist, s, 0) + 1
    end
    return dist
end
