"""
    ParticleGas(; D=3, N, L=nothing, rho=nothing, pair_potential)

Construct a `ParticleSystem` of `N` monatomic gas particles.
`L` can be a scalar (cubic box) or an `SVector` (anisotropic box).
"""
function ParticleGas(; D::Int=3, N::Integer, L=nothing, rho=nothing,
                       pair_potential::AbstractPairPotential)
    @assert (L !== nothing) ⊻ (rho !== nothing) "Provide either `L` or `rho`, not both"
    if rho !== nothing
        L_scalar = (N / rho)^(1/D)
        env = PeriodicBox{D}(L_scalar)
    elseif L isa Real
        env = PeriodicBox{D}(L)
    else
        env = PeriodicBox(SVector{D}(float.(L)))
    end
    T = eltype(env.L)
    n = Int(N)
    TIdx = _index_type(n)
    cl = _make_cell_list(Val(D), n, env, pair_potential)
    ParticleSystem{D, T, typeof(env), typeof(pair_potential), Monatomic,
                   CacheMonatomic{T}, typeof(cl), TIdx}(
        env,
        [zero(SVector{D,T}) for _ in 1:n],
        [Monatomic() for _ in 1:n],
        TIdx.(1:n), ones(TIdx, n),
        pair_potential, CacheMonatomic(zero(T)), cl)
end

function init!(sys::ParticleSystem{D,T,TEnv,P,Monatomic}, type::Symbol; rng=nothing) where {D,T,TEnv,P}
    type == :random || error("Unknown initialization type: $type")
    @assert rng !== nothing "Random initialization requires rng"
    for i in 1:length(sys.positions)
        sys.positions[i] = SVector{D,T}(ntuple(d -> rand(rng, T) * sys.env.L[d], Val(D)))
    end
    build!(sys.cell_list, sys.positions)
    _recompute_energy!(sys)
    return sys
end

function _recompute_energy!(sys::ParticleSystem{D,T,TEnv,P,Monatomic}) where {D,T,TEnv,P}
    sys.cache.pair = _compute_pair_energy(sys)
    return nothing
end

function _compute_pair_energy(sys::ParticleSystem{D,T,TEnv,P,Monatomic}) where {D,T,TEnv,P}
    N = length(sys.positions)
    E = zero(T)
    env = sys.env
    @inbounds for i in 1:N-1, j in i+1:N
        r_sq = distance_sq(env, sys.positions[i], sys.positions[j])
        E += sys.pair_potential(r_sq)
    end
    return E
end

@inline _monomer_energy(sys::ParticleSystem{D,T,TEnv,P,Monatomic}, i::Int) where {D,T,TEnv,P} =
    _local_pair_energy_no_excl(sys, i)

@inline _monomer_energy_change(sys::ParticleSystem{D,T,TEnv,P,Monatomic}, i::Int, new_pos::SVector{D,T}) where {D,T,TEnv,P} =
    _pair_energy_change(sys, i, new_pos)
