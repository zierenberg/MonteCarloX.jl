"""
    site_index(x, y, z, Lx, Ly) -> Int

Convert 0-based 3D coordinates to 1-based linear index on an Lx x Ly x Lz lattice.
"""
@inline site_index(x, y, z, Lx, Ly) = z * Lx * Ly + y * Lx + x + 1

"""
    site_coords(site, Lx, Ly) -> (x, y, z)

Convert 1-based linear index to 0-based 3D coordinates.
"""
@inline function site_coords(site, Lx, Ly)
    s = site - 1
    x = s % Lx
    s = s ÷ Lx
    y = s % Ly
    z = s ÷ Ly
    return (x, y, z)
end

"""
    build_cubic_neighbors(Lx, Ly, Lz) -> Vector{NTuple{6,Int}}

Build a precomputed neighbor table for a 3D cubic lattice with periodic boundary
conditions. Returns a vector of length Lx*Ly*Lz where each entry is a 6-tuple
of 1-based neighbor indices: (+x, -x, +y, -y, +z, -z).
"""
function build_cubic_neighbors(Lx, Ly, Lz)
    N = Lx * Ly * Lz
    nbrs = Vector{NTuple{6,Int}}(undef, N)
    for site in 1:N
        x, y, z = site_coords(site, Lx, Ly)
        xp = site_index(mod(x + 1, Lx), y, z, Lx, Ly)
        xm = site_index(mod(x - 1, Lx), y, z, Lx, Ly)
        yp = site_index(x, mod(y + 1, Ly), z, Lx, Ly)
        ym = site_index(x, mod(y - 1, Ly), z, Lx, Ly)
        zp = site_index(x, y, mod(z + 1, Lz), Lx, Ly)
        zm = site_index(x, y, mod(z - 1, Lz), Lx, Ly)
        nbrs[site] = (xp, xm, yp, ym, zp, zm)
    end
    return nbrs
end

"""
    lattice_difference(c1, c2, L) -> Int

Minimum image displacement c1 - c2 on a periodic axis of length L.
Result is in (-L/2, L/2].
"""
@inline function lattice_difference(c1, c2, L)
    d = c1 - c2
    d -= L * round(Int, d / L)
    return d
end
