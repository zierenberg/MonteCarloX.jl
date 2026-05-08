"""
    center_of_mass(sys::BeadSpringPolymer, n) -> SVector{D,Float64}

Center of mass of polymer `n`, computed with minimum-image unwinding
relative to the first monomer.
"""
function center_of_mass(sys::BeadSpringPolymer{D,T}, n::Int) where {D,T}
    M   = sys.lengths[n]
    off = sys.offsets[n]
    ref = sys.positions[off + 1]
    cm  = zero(MVector{D,Float64})
    for k in 1:M
        pos = sys.positions[off + k]
        d = minimum_image_displacement(pos, ref, sys.L)
        cm .+= d
    end
    cm ./= M
    for d in 1:D
        cm[d] = wrap_coordinate(ref[d] + cm[d], sys.L)
    end
    return SVector{D,Float64}(cm)
end

"""
    radius_of_gyration_sq(sys::BeadSpringPolymer, n) -> Float64

Squared radius of gyration of polymer `n`.
"""
function radius_of_gyration_sq(sys::BeadSpringPolymer{D,T}, n::Int) where {D,T}
    M   = sys.lengths[n]
    off = sys.offsets[n]
    cm  = center_of_mass(sys, n)
    rg2 = 0.0
    for k in 1:M
        pos = sys.positions[off + k]
        d = minimum_image_displacement(pos, cm, sys.L)
        rg2 += sum(abs2, d)
    end
    return rg2 / M
end

"""
    end_to_end_distance_sq(sys::BeadSpringPolymer, n) -> Float64

Squared end-to-end distance of polymer `n` under minimum image convention.
"""
function end_to_end_distance_sq(sys::BeadSpringPolymer{D,T}, n::Int) where {D,T}
    M   = sys.lengths[n]
    off = sys.offsets[n]
    r1  = sys.positions[off + 1]
    rN  = sys.positions[off + M]
    return Float64(minimum_image_sq(r1, rN, sys.L))
end

"""
    gyration_tensor(sys::BeadSpringPolymer, n) -> SMatrix{D,D,Float64}

Gyration tensor of polymer `n`. Trace equals radius_of_gyration_sq.
"""
function gyration_tensor(sys::BeadSpringPolymer{D,T}, n::Int) where {D,T}
    M   = sys.lengths[n]
    off = sys.offsets[n]
    cm  = center_of_mass(sys, n)
    G   = zeros(MMatrix{D,D,Float64})
    for k in 1:M
        pos = sys.positions[off + k]
        d = minimum_image_displacement(pos, cm, sys.L)
        for i in 1:D, j in 1:D
            G[i,j] += d[i] * d[j]
        end
    end
    G ./= M
    return SMatrix{D,D,Float64}(G)
end
