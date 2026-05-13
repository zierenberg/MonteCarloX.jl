abstract type AbstractMolecule end

mutable struct CacheMonatomic{T}
    pair::T
end

mutable struct CachePolymer{T}
    pair::T
    bond::T
    bend::T
end

@inline total_energy(c::CacheMonatomic) = c.pair
@inline total_energy(c::CachePolymer)   = c.pair + c.bond + c.bend

# ── Cache update from energy deltas ──────────────────────────────────────────
# Monatomic: dE is a scalar (pair only)
@inline _update_cache!(c::CacheMonatomic, dE) = (c.pair += dE; nothing)
@inline _total_dE(dE::Real) = dE

# Polymer monomer: dE is a NamedTuple (; pair, bond, bend)
@inline function _update_cache!(c::CachePolymer, dE::NamedTuple{(:pair, :bond, :bend)})
    c.pair += dE.pair
    c.bond += dE.bond
    c.bend += dE.bend
    return nothing
end
@inline _total_dE(dE::NamedTuple{(:pair, :bond, :bend)}) = dE.pair + dE.bond + dE.bend
