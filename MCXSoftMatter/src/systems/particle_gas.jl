"""
    ParticleGas(; D=3, N, L=nothing, rho=nothing, pair_potential)

Construct a `ParticleSystem` of `N` monatomic gas particles.
"""
function ParticleGas(; D::Int=3, N::Integer, L=nothing, rho=nothing,
                       pair_potential::AbstractPairPotential)
    @assert (L !== nothing) ⊻ (rho !== nothing) "Provide either `L` or `rho`, not both"
    rho !== nothing && (L = (N / rho)^(1/D))
    T = typeof(float(L))
    n = Int(N)
    TIdx = _index_type(n)
    cl = _make_cell_list(Val(D), n, T(L), pair_potential)
    ParticleSystem{D, T, typeof(pair_potential), Monatomic,
                   CacheMonatomic{T}, typeof(cl), TIdx}(
        [zero(SVector{D,T}) for _ in 1:n],
        [Monatomic() for _ in 1:n],
        TIdx.(1:n), ones(TIdx, n),
        T(L), pair_potential, CacheMonatomic(zero(T)), cl)
end

function init!(sys::ParticleSystem{D,T,P,Monatomic}, type::Symbol; rng=nothing) where {D,T,P}
    type == :random || error("Unknown initialization type: $type")
    @assert rng !== nothing "Random initialization requires rng"
    for i in 1:length(sys.positions)
        sys.positions[i] = SVector{D,T}(ntuple(_ -> rand(rng, T) * sys.L, Val(D)))
    end
    build!(sys.cell_list, sys.positions)
    _recompute_energy!(sys)
    return sys
end

function _recompute_energy!(sys::ParticleSystem{D,T,P,Monatomic}) where {D,T,P}
    sys.cache.pair = _compute_pair_energy(sys)
    return nothing
end

function _compute_pair_energy(sys::ParticleSystem{D,T,P,Monatomic}) where {D,T,P}
    N = length(sys.positions)
    E = zero(T)
    @inbounds for i in 1:N-1, j in i+1:N
        r_sq = minimum_image_sq(sys.positions[i], sys.positions[j], sys.L)
        E += sys.pair_potential(r_sq)
    end
    return E
end

@inline _monomer_energy(sys::ParticleSystem{D,T,P,Monatomic}, i::Int) where {D,T,P} =
    _local_pair_energy_no_excl(sys, i)
