"""
    radius_of_gyration_sq(sys::LatticePolymer, m) -> Float64

Squared radius of gyration for polymer `m`:
  Rg² = (1/N) Σᵢ (rᵢ - r_cm)²
using minimum image convention for PBC.
"""
function radius_of_gyration_sq(sys::LatticePolymer, m::Int)
    N = sys.N
    poly = sys.polymers[m]
    cm = center_of_mass(sys, m)

    rg2 = 0.0
    for k in 1:N
        x, y, z = site_coords(poly[k], sys.Lx, sys.Ly)
        dx = lattice_difference(Float64(x), cm[1], sys.Lx)
        dy = lattice_difference(Float64(y), cm[2], sys.Ly)
        dz = lattice_difference(Float64(z), cm[3], sys.Lz)
        rg2 += dx^2 + dy^2 + dz^2
    end
    return rg2 / N
end

"""
    center_of_mass(sys::LatticePolymer, m) -> (Float64, Float64, Float64)

Center of mass of polymer `m`, computed using the middle monomer as reference
and minimum image convention for PBC.
"""
function center_of_mass(sys::LatticePolymer, m::Int)
    N = sys.N
    poly = sys.polymers[m]

    # Use middle monomer as reference point
    ref = poly[N ÷ 2 + 1]
    rx, ry, rz = site_coords(ref, sys.Lx, sys.Ly)

    cx, cy, cz = 0.0, 0.0, 0.0
    for k in 1:N
        x, y, z = site_coords(poly[k], sys.Lx, sys.Ly)
        cx += lattice_difference(Float64(x), Float64(rx), sys.Lx)
        cy += lattice_difference(Float64(y), Float64(ry), sys.Ly)
        cz += lattice_difference(Float64(z), Float64(rz), sys.Lz)
    end
    cx = _wrap(rx + cx / N, sys.Lx)
    cy = _wrap(ry + cy / N, sys.Ly)
    cz = _wrap(rz + cz / N, sys.Lz)
    return (cx, cy, cz)
end

"""
    end_to_end_distance_sq(sys::LatticePolymer, m) -> Float64

Squared end-to-end distance for polymer `m` with minimum image convention.
"""
function end_to_end_distance_sq(sys::LatticePolymer, m::Int)
    poly = sys.polymers[m]
    x1, y1, z1 = site_coords(poly[1], sys.Lx, sys.Ly)
    xN, yN, zN = site_coords(poly[end], sys.Lx, sys.Ly)
    dx = lattice_difference(Float64(xN), Float64(x1), sys.Lx)
    dy = lattice_difference(Float64(yN), Float64(y1), sys.Ly)
    dz = lattice_difference(Float64(zN), Float64(z1), sys.Lz)
    return dx^2 + dy^2 + dz^2
end
