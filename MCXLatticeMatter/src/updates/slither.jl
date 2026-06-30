"""
    _shift_polymer!(poly, new_coords, forward)

Shift polymer array: remove one end, insert `new_coords` at the other.
Forward: remove first, append. Backward: remove last, prepend.
"""
@inline function _shift_polymer!(poly, new_coords, grow_at_end::Bool)
    M = length(poly)
    if grow_at_end
        for m in 1:M-1; poly[m] = poly[m+1]; end
        poly[M] = new_coords
    else
        for m in M:-1:2; poly[m] = poly[m-1]; end
        poly[1] = new_coords
    end
end

"""
    slither!(sys, alg)

Reptation move: remove one end monomer and grow at the other end.
"""
function slither!(sys::LatticePolymer{D}, alg::AbstractMarkovChainMonteCarlo) where {D}
    n = rand(alg.rng, 1:num_polymers(sys))
    M = polymer_length(sys, n)
    M < 2 && return nothing
    poly = sys.polymers[n]

    grow_at_end = rand(alg.rng, Bool)
    remove_idx = grow_at_end ? 1 : M
    grow_idx   = grow_at_end ? M : 1

    remove_site = coords_to_site(poly[remove_idx], sys.dims)
    grow_site   = coords_to_site(poly[grow_idx], sys.dims)
    nbrs = sys.neighbors[grow_site]
    new_site = nbrs[rand(alg.rng, 1:length(nbrs))]

    new_site != remove_site && sys.state[new_site] != 0 && return nothing

    intra_b, inter_b = site_contacts(sys, remove_site)
    old_end = poly[remove_idx]

    sys.state[remove_site] = 0
    sys.state[new_site] = n
    _shift_polymer!(poly, site_to_coords(new_site, sys.dims), grow_at_end)

    intra_a, inter_a = site_contacts(sys, new_site)
    E_old = -sys.J_intra * intra_b - sys.J_inter * inter_b
    E_new = -sys.J_intra * intra_a - sys.J_inter * inter_a
    if accept!(alg, E_new, E_old)
        sys.cached_intra += intra_a - intra_b
        sys.cached_inter += inter_a - inter_b
    else
        sys.state[new_site] = 0
        _shift_polymer!(poly, old_end, !grow_at_end)
        sys.state[remove_site] = n
    end
    return nothing
end
