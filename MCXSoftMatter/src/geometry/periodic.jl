"""
    PeriodicBox{D, T}

D-dimensional periodic box with side lengths `L` and precomputed `inv_L = 1 ./ L`.
Supports anisotropic boxes (different Lx, Ly, Lz).
"""
struct PeriodicBox{D, T<:AbstractFloat}
    L::SVector{D,T}
    inv_L::SVector{D,T}
end

PeriodicBox(L::SVector{D,T}) where {D,T} = PeriodicBox{D,T}(L, inv.(L))

# Cubic box from scalar
PeriodicBox{D}(L::Real) where D = PeriodicBox(SVector{D}(ntuple(_ -> float(L), Val(D))))

"""
    wrap_position(pos::SVector{D,T}, box::PeriodicBox{D,T}) -> SVector{D,T}

Wrap all components of a position into [0, L_d) per dimension.
"""
@inline function wrap_position(pos::SVector{D,T}, box::PeriodicBox{D}) where {D,T}
    SVector{D,T}(ntuple(Val(D)) do d
        x = pos[d]
        x - box.L[d] * floor(x * box.inv_L[d])
    end)
end

"""
    minimum_image_displacement(ri, rj, box) -> SVector{D,T}

Displacement vector ri - rj under minimum image convention.
"""
@inline function minimum_image_displacement(ri::SVector{D,T}, rj::SVector{D,T}, box::PeriodicBox{D}) where {D,T}
    SVector{D,T}(ntuple(Val(D)) do d
        dx = ri[d] - rj[d]
        muladd(-box.L[d], round(dx * box.inv_L[d]), dx)
    end)
end

"""
    minimum_image_sq(ri, rj, box) -> T

Squared distance under minimum image convention.
"""
@inline function minimum_image_sq(ri::SVector{D,T}, rj::SVector{D,T}, box::PeriodicBox{D}) where {D,T}
    _sq_dist(ri, rj, box)
end

"""
    _sq_dist(ri, rj, box) -> T

Fast squared minimum-image distance via scalar accumulation (no intermediate SVector).
Uses precomputed `inv_L` to replace division with multiplication, and `muladd` for FMA.
"""
@inline function _sq_dist(ri::SVector{D,T}, rj::SVector{D,T}, box::PeriodicBox{D}) where {D,T}
    r_sq = zero(T)
    @inbounds for d in 1:D
        dx = ri[d] - rj[d]
        dx = muladd(-box.L[d], round(dx * box.inv_L[d]), dx)
        r_sq = muladd(dx, dx, r_sq)
    end
    return r_sq
end
