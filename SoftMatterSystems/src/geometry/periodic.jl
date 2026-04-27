"""
    wrap_coordinate(x, L) -> T

Wrap coordinate into [0, L).
"""
@inline wrap_coordinate(x, L) = x - L * floor(x / L)

"""
    wrap_position(pos::SVector{3,T}, L) -> SVector{3,T}

Wrap all 3 components of a position into [0, L).
"""
@inline function wrap_position(pos::SVector{3,T}, L) where T
    SVector{3,T}(
        wrap_coordinate(pos[1], L),
        wrap_coordinate(pos[2], L),
        wrap_coordinate(pos[3], L)
    )
end

"""
    minimum_image_sq(ri, rj, L) -> T

Squared distance between two positions under minimum image convention.
"""
@inline function minimum_image_sq(ri::SVector{3,T}, rj::SVector{3,T}, L) where T
    dx = ri[1] - rj[1]
    dy = ri[2] - rj[2]
    dz = ri[3] - rj[3]
    dx -= L * round(dx / L)
    dy -= L * round(dy / L)
    dz -= L * round(dz / L)
    return dx*dx + dy*dy + dz*dz
end

"""
    minimum_image_displacement(ri, rj, L) -> SVector{3,T}

Vector displacement ri - rj under minimum image convention.
"""
@inline function minimum_image_displacement(ri::SVector{3,T}, rj::SVector{3,T}, L) where T
    dx = ri[1] - rj[1]
    dy = ri[2] - rj[2]
    dz = ri[3] - rj[3]
    dx -= L * round(dx / L)
    dy -= L * round(dy / L)
    dz -= L * round(dz / L)
    return SVector{3,T}(dx, dy, dz)
end
