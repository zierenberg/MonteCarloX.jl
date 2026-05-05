"""
    center_of_mass(sys, n) -> SVector{D,Float64}

Center of mass of polymer `n` with minimum-image convention.
"""
function center_of_mass(sys::LatticePolymer{D}, n::Int) where {D}
    poly = sys.polymers[n]
    M = polymer_length(sys, n)
    ref = poly[1]

    # Mean displacement from reference monomer (minimum-image)
    cm = zero(MVector{D,Float64})
    for m in 1:M
        cm .+= lattice_difference(poly[m], ref, sys.dims)
    end
    # Shift back to reference frame and wrap into [0, L)
    for d in 1:D
        cm[d] = mod(ref[d] + cm[d] / M, sys.dims[d])
    end
    return SVector{D,Float64}(cm)
end

"""
    radius_of_gyration_sq(sys, n) -> Float64

Squared radius of gyration for polymer `n` using minimum-image convention.
"""
function radius_of_gyration_sq(sys::LatticePolymer{D}, n::Int) where {D}
    poly = sys.polymers[n]
    M = polymer_length(sys, n)
    cm = MVector{D,Float64}(center_of_mass(sys, n))

    rg2 = 0.0
    for m in 1:M
        rg2 += sum(abs2, lattice_difference(poly[m], cm, sys.dims))
    end
    return rg2 / M
end

"""
    end_to_end_distance_sq(sys, n) -> Int

Squared end-to-end distance for polymer `n` with minimum-image convention.
"""
end_to_end_distance_sq(sys::LatticePolymer, n::Int) =
    lattice_distance_sq(sys.polymers[n][end], sys.polymers[n][1], sys.dims)

"""
    gyration_tensor(sys, n) -> Matrix{Float64}

DxD gyration tensor for polymer `n`. Satisfies `tr(G) == radius_of_gyration_sq`.
"""
function gyration_tensor(sys::LatticePolymer{D}, n::Int) where {D}
    poly = sys.polymers[n]
    M = polymer_length(sys, n)
    cm = MVector{D,Float64}(center_of_mass(sys, n))

    G = zeros(Float64, D, D)
    for m in 1:M
        r = lattice_difference(poly[m], cm, sys.dims)
        for i in 1:D, j in 1:D
            G[i, j] += r[i] * r[j]
        end
    end
    return G / M
end
