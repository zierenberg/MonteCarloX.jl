struct ClusterWolff end
# """
# Wolff single cluster update

# #Arguments:
# * `spins` : array of spin values
# * `nearest_neighbors` : function that returns a list of nearest neighbors to index i
# * `beta` : inverse temperature
# * `rng`  : random number generator

# Reference: U. Wolff, Phys. Rev. Lett. 62, 361 (1989)

# consider for Potts:
# https://stanford.edu/~kurinsky/ClassWork/Physics271_Final_Paper.pdf

# consider BlumeCapel implementation used in
# Fytas et al. Phys. Rev. E 97, 040102(R) (2018)

# #WARNING
# This is depricated
# """
function update(
        alg::ClusterWolff,
        rng::AbstractRNG,
        spins::Vector{Int},
        nearest_neighbors::Function,
        betaJ::Float64
    )
    N = length(spins)

    # get random spin and flip it
    index_i = rand(rng, 1:N)
    s_i = spins[index_i]
    spins[index_i] *= -1

    # probability (per bond!)
    p = 1.0 - exp(-2.0 * betaJ)

    # check all bonds and already flip cluster
    proposed = copy(nearest_neighbors(index_i))
    while !isempty(proposed)
        # check proposed sites for alignment with base spin s_i and add to cluster with probability p
        index_j = pop!(proposed)
        # no need to check if site is already in cluster, because spin flip disqualifies it already
        if spins[index_j] == s_i && rand(rng) < p
            spins[index_j] *= -1
            append!(proposed, nearest_neighbors(index_j))
        end
    end
end
