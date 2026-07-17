# ── Cluster updates (Wolff, Swendsen–Wang) ────────────────────────────────────
#
# Cluster moves are built from an INVOLUTION R on the spin type with the bilinear property
#     ⟨s,t⟩ − ⟨reflect(R,s), t⟩ = 2·component(R,s)·component(R,t),
# so a pair bond (i,j) with coupling J is activated with p = 1 − exp(logweight(ens, 2J·cᵢcⱼ))
# when that is positive (Boltzmann: 1 − exp(−2βJ·cᵢcⱼ)): spin flip σ → −σ for discrete spins,
# reflection s → s − 2⟨s,r⟩r about a random plane for continuous spins (Wolff's
# embedded-Ising trick). Bond activation is evaluated on the UNFLIPPED configuration (sites
# flip after growth), so each bond is tested at most once.
#
# Support is gated per term by assert_clusterable: pair terms provide bonds; CrystalField is
# R-invariant (σ² and |s| preserved) and rides along; an ExternalField breaks the flip
# symmetry (would need a ghost site) and non-Hamiltonian terms are refused. NOTE for
# Blume–Capel-like spin types: σ = 0 sites never join a cluster, so the number of zeros is
# conserved — interleave cluster moves with local spin_flip! sweeps for ergodicity.
#
# Caches are rebuilt by recompute_all! after each move: near criticality clusters are O(N),
# so the O(N·z) rebuild does not change the complexity class. (Incremental boundary-bond
# cache updates are a possible later optimization.)

struct SpinFlip end
@inline reflect(::SpinFlip, s::Int8) = -s
@inline component(::SpinFlip, s::Int8) = Int(s)

struct PlaneReflection{V}
    r::V
end
@inline reflect(R::PlaneReflection, s) = s - 2 * _dot(s, R.r) * R.r
@inline component(R::PlaneReflection, s) = _dot(s, R.r)

random_reflection(rng::AbstractRNG, ::Spin) = SpinFlip()
random_reflection(rng::AbstractRNG, ::XYSpin) = PlaneReflection(cis(2π * rand(rng)))
random_reflection(rng::AbstractRNG, spintype::HeisenbergSpin) =
    PlaneReflection(random_state(rng, spintype))

assert_clusterable(t::AbstractInteraction) =
    error("cluster updates: $(typeof(t)) is not supported")
assert_clusterable(::PairInteraction) = nothing
assert_clusterable(::PairInteractionMatrix{TJ,SymmetricCoupling}) where TJ = nothing
assert_clusterable(::CrystalField) = nothing
assert_clusterable(t::ExternalField) = iszero(t.h) ? nothing :
    error("cluster updates: external field breaks the flip symmetry (ghost site not implemented)")
assert_clusterable(ints::Tuple) =
    (assert_clusterable(first(ints)); assert_clusterable(Base.tail(ints)))
assert_clusterable(::Tuple{}) = nothing

# Wolff frontier: activate bonds from cluster site i (component c_i) to partners outside the
# cluster. On-site terms contribute no bonds (default).
@inline activate!(t::AbstractInteraction, rng, spins, i, c_i, ens, R, in_cluster, stack) = nothing
@inline function activate!(t::PairInteraction, rng, spins, i, c_i, ens, R, in_cluster, stack)
    twoJ = 2 * t.J
    @inbounds for j in t.partners[i]
        if !in_cluster[j]
            lw = logweight(ens, twoJ * c_i * component(R, spins[j]))
            if lw < 0 && rand(rng) < -expm1(lw)
                in_cluster[j] = true
                push!(stack, j)
            end
        end
    end
    return nothing
end
@inline function activate!(t::PairInteractionMatrix{TJ,SymmetricCoupling}, rng, spins, i, c_i,
                           ens, R, in_cluster, stack) where TJ
    J = t.J
    @inbounds for ptr in J.colptr[i]:(J.colptr[i+1]-1)
        j = J.rowval[ptr]
        (j == i || in_cluster[j]) && continue
        lw = logweight(ens, 2 * J.nzval[ptr] * c_i * component(R, spins[j]))
        if lw < 0 && rand(rng) < -expm1(lw)
            in_cluster[j] = true
            push!(stack, j)
        end
    end
    return nothing
end
@inline activate!(ints::Tuple, rng, spins, i, c_i, ens, R, in_cluster, stack) =
    (activate!(first(ints), rng, spins, i, c_i, ens, R, in_cluster, stack);
     activate!(Base.tail(ints), rng, spins, i, c_i, ens, R, in_cluster, stack))
@inline activate!(::Tuple{}, rng, spins, i, c_i, ens, R, in_cluster, stack) = nothing

# Cluster moves are ALGORITHM OBJECTS like MetropolisAlgorithm: rng, target ENSEMBLE (bond
# probabilities go through logweight; gated to linear log-weights by assert_linear_ensemble),
# counters, and their scratch buffers. The unified entry point cluster_update!(sys, alg)
# mirrors spin_flip!(sys, alg); the convenience keyword Wolff(rng; β) mirrors
# MetropolisAlgorithm(rng; β).
#@REVIEW: Again here the same as with heatbath and nfold. It would be better to have a clear separation of logweifght and update part. Not sure if this is possible. But worth consideration given the numbe rof examples we have. 
abstract type AbstractClusterUpdate <: MonteCarloX.AbstractAlgorithm end

MonteCarloX.steps(alg::AbstractClusterUpdate) = alg.steps

"""
    Wolff(rng, ensemble)
    Wolff(rng; β)

Wolff single-cluster algorithm object: owns the rng, the target ensemble (linear log-weight
required), counters, and scratch buffers. Apply with `cluster_update!(sys, alg)`.
"""
mutable struct Wolff{R<:AbstractRNG, E} <: AbstractClusterUpdate
    rng::R
    ensemble::E
    steps::Int
    summed_size::Int
    in_cluster::BitVector
    stack::Vector{Int}
end
function Wolff(rng::AbstractRNG, ensemble)
    assert_linear_ensemble(ensemble, "cluster updates")
    @warn "Wolff algorithm is experimental"
    return Wolff(rng, ensemble, 0, 0, falses(0), Int[])
end
Wolff(rng::AbstractRNG; β::Real) = Wolff(rng, BoltzmannEnsemble(β=β))

"Mean Wolff cluster size over all updates performed with this algorithm object."
mean_cluster_size(alg::Wolff) = alg.summed_size / alg.steps

"""
    cluster_update!(sys, alg::Wolff) -> cluster_size

One Wolff cluster move: grow a cluster from a random seed through the pair bonds and apply a
random involution to it. Scratch buffers live in the algorithm object (resized on demand).
"""
function cluster_update!(sys::SpinSystem, alg::Wolff)
    assert_clusterable(sys.interactions)
    spins = sys.spins
    rng = alg.rng
    ens = alg.ensemble
    length(alg.in_cluster) == length(spins) || (alg.in_cluster = falses(length(spins)))
    in_cluster = alg.in_cluster
    stack = alg.stack
    R = random_reflection(rng, sys.spintype)
    fill!(in_cluster, false)
    empty!(stack)
    seed = pick_site(rng, length(spins))
    in_cluster[seed] = true
    push!(stack, seed)
    cluster_size = 0
    while !isempty(stack)
        i = pop!(stack)
        cluster_size += 1
        c_i = component(R, @inbounds spins[i])
        activate!(sys.interactions, rng, spins, i, c_i, ens, R, in_cluster, stack)
    end
    @inbounds for i in eachindex(spins)
        in_cluster[i] && (spins[i] = reflect(R, spins[i]))
    end
    recompute_all!(sys)
    alg.steps += 1
    alg.summed_size += cluster_size
    return cluster_size
end

# Union-find with path halving (Swendsen–Wang cluster labels).
function find_root!(parent::Vector{Int}, i::Int)
    @inbounds while parent[i] != i
        parent[i] = parent[parent[i]]
        i = parent[i]
    end
    return i
end

# Swendsen–Wang bond pass: activate every satisfied bond once (j > i). On-site terms: none.
activate_bonds!(t::AbstractInteraction, rng, spins, ens, R, parent) = nothing
function activate_bonds!(t::PairInteraction, rng, spins, ens, R, parent)
    twoJ = 2 * t.J
    @inbounds for i in eachindex(spins), j in t.partners[i]
        j > i || continue
        lw = logweight(ens, twoJ * component(R, spins[i]) * component(R, spins[j]))
        if lw < 0 && rand(rng) < -expm1(lw)
            parent[find_root!(parent, i)] = find_root!(parent, j)
        end
    end
    return nothing
end
function activate_bonds!(t::PairInteractionMatrix{TJ,SymmetricCoupling}, rng, spins, ens, R,
                         parent) where TJ
    J = t.J
    @inbounds for i in eachindex(spins), ptr in J.colptr[i]:(J.colptr[i+1]-1)
        j = J.rowval[ptr]
        j > i || continue
        lw = logweight(ens, 2 * J.nzval[ptr] * component(R, spins[i]) * component(R, spins[j]))
        if lw < 0 && rand(rng) < -expm1(lw)
            parent[find_root!(parent, i)] = find_root!(parent, j)
        end
    end
    return nothing
end
activate_bonds!(ints::Tuple, rng, spins, ens, R, parent) =
    (activate_bonds!(first(ints), rng, spins, ens, R, parent);
     activate_bonds!(Base.tail(ints), rng, spins, ens, R, parent))
activate_bonds!(::Tuple{}, rng, spins, ens, R, parent) = nothing

"""
    SwendsenWang(rng, ensemble)
    SwendsenWang(rng; β)

Swendsen–Wang algorithm object; apply with `cluster_update!(sys, alg)`.
"""
mutable struct SwendsenWang{R<:AbstractRNG, E} <: AbstractClusterUpdate
    rng::R
    ensemble::E
    steps::Int
    summed_clusters::Int
    parent::Vector{Int}
    flip::BitVector
    decided::BitVector
end
function SwendsenWang(rng::AbstractRNG, ensemble)
    assert_linear_ensemble(ensemble, "cluster updates")
    @warn "Swendsen–Wang algorithm is experimental"
    return SwendsenWang(rng, ensemble, 0, 0, Int[], falses(0), falses(0))
end
SwendsenWang(rng::AbstractRNG; β::Real) = SwendsenWang(rng, BoltzmannEnsemble(β=β))

"Mean number of Swendsen–Wang clusters per sweep for this algorithm object."
mean_cluster_count(alg::SwendsenWang) = alg.summed_clusters / alg.steps

"""
    cluster_update!(sys, alg::SwendsenWang) -> n_clusters

One Swendsen–Wang sweep: percolate all satisfied pair bonds, then apply a random involution
to each cluster independently with probability 1/2 (embedded-Ising for continuous spins).
"""
function cluster_update!(sys::SpinSystem, alg::SwendsenWang)
    assert_clusterable(sys.interactions)
    spins = sys.spins
    rng = alg.rng
    ens = alg.ensemble
    N = length(spins)
    if length(alg.parent) != N
        alg.parent = collect(1:N)
        alg.flip = falses(N)
        alg.decided = falses(N)
    end
    parent = alg.parent
    R = random_reflection(rng, sys.spintype)
    @inbounds for i in 1:N
        parent[i] = i
    end
    activate_bonds!(sys.interactions, rng, spins, ens, R, parent)
    fill!(alg.decided, false)
    n_clusters = 0
    @inbounds for i in 1:N
        r = find_root!(parent, i)
        if !alg.decided[r]
            alg.decided[r] = true
            alg.flip[r] = rand(rng, Bool)
            n_clusters += 1
        end
        alg.flip[r] && (spins[i] = reflect(R, spins[i]))
    end
    recompute_all!(sys)
    alg.steps += 1
    alg.summed_clusters += n_clusters
    return n_clusters
end
