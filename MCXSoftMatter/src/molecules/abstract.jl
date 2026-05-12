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
