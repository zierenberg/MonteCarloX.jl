"""
    wrap_coordinate(x, L)

Wrap coordinate into [0, L).
"""
@inline wrap_coordinate(x, L) = x - L * floor(x / L)

"""
    wrap_position(pos::SVector{D,T}, L) -> SVector{D,T}

Wrap all components of a position into [0, L).
"""
@inline function wrap_position(pos::SVector{D,T}, L) where {D,T}
    SVector{D,T}(ntuple(d -> wrap_coordinate(pos[d], L), Val(D)))
end

"""
    minimum_image_displacement(ri, rj, L) -> SVector{D,T}

Displacement vector ri - rj under minimum image convention.
"""
@inline function minimum_image_displacement(ri::SVector{D,T}, rj::SVector{D,T}, L) where {D,T}
    SVector{D,T}(ntuple(d -> begin
        dx = ri[d] - rj[d]
        dx - T(L) * round(dx / T(L))
    end, Val(D)))
end

"""
    minimum_image_sq(ri, rj, L) -> T

Squared distance under minimum image convention.
"""
@inline function minimum_image_sq(ri::SVector{D,T}, rj::SVector{D,T}, L) where {D,T}
    _sq_dist(ri, rj, L)
end

"""
    _sq_dist(ri, rj, L) -> T

Fast squared minimum-image distance via scalar accumulation (no intermediate SVector).
"""
@inline function _sq_dist(ri::SVector{D,T}, rj::SVector{D,T}, L) where {D,T}
    r_sq = zero(T)
    @inbounds for d in 1:D
        dx = ri[d] - rj[d]
        dx -= T(L) * round(dx / T(L))
        r_sq += dx * dx
    end
    return r_sq
end
