"""
    pivot_move!(sys, alg)

Pivot move: rotate one end of a chain by ±90° in a random plane.
Works in D=2 and D=3.
"""
function pivot_move!(sys::LatticePolymer{D}, alg::AbstractImportanceSampling) where {D}
    (D == 2 || D == 3) || return nothing

    n = rand(alg.rng, 1:num_polymers(sys))
    M = polymer_length(sys, n)
    poly = sys.polymers[n]

    pivot_m = rand(alg.rng, 1:M)
    pivot_at_end = rand(alg.rng, Bool)
    axis = rand(alg.rng, 1:D)
    sin_a = rand(alg.rng, Bool) ? -1 : 1

    start_m = pivot_at_end ? pivot_m + 1 : 1
    end_m   = pivot_at_end ? M : pivot_m - 1
    start_m > end_m && return nothing

    n_move = end_m - start_m + 1
    sec_axis = axis % D + 1
    pivot_c = poly[pivot_m]

    # Collect old sites
    changed = zeros(Int, 2 * n_move)
    for (i, m) in enumerate(start_m:end_m)
        changed[i] = coords_to_site(poly[m], sys.dims)
    end

    # Compute rotated sites with inline collision checks
    for (i, m) in enumerate(start_m:end_m)
        mc = poly[m]
        dx_ax  = lattice_difference(mc[axis], pivot_c[axis], sys.dims[axis])
        dx_sec = lattice_difference(mc[sec_axis], pivot_c[sec_axis], sys.dims[sec_axis])
        nc = SVector{D,Int}(ntuple(d -> begin
            if d == axis
                apply_pbc(pivot_c[axis] - sin_a * dx_sec, sys.dims[axis])
            elseif d == sec_axis
                apply_pbc(pivot_c[sec_axis] + sin_a * dx_ax, sys.dims[sec_axis])
            else
                mc[d]
            end
        end, Val(D)))
        new_site = coords_to_site(nc, sys.dims)

        # Self-overlap in rotated segment
        for j in 1:i-1
            changed[n_move + j] == new_site && return nothing
        end

        # Collision with other polymers or unmoved monomers
        occ = sys.state[new_site]
        if occ != 0
            is_old = false
            for j in 1:n_move
                changed[j] == new_site && (is_old = true; break)
            end
            is_old || return nothing
        end

        changed[n_move + i] = new_site
    end

    # Contacts before
    intra_b, inter_b = 0, 0
    for i in 1:n_move
        ci, ce = site_contacts(sys, changed[i]); intra_b += ci; inter_b += ce
    end

    # Apply: clear old, write new coords from stored sites
    for i in 1:n_move; sys.state[changed[i]] = 0; end
    for (i, m) in enumerate(start_m:end_m)
        poly[m] = site_to_coords(changed[n_move + i], sys.dims)
        sys.state[changed[n_move + i]] = n
    end

    # Contacts after
    intra_a, inter_a = 0, 0
    for i in 1:n_move
        ci, ce = site_contacts(sys, changed[n_move + i]); intra_a += ci; inter_a += ce
    end

    E_old = -sys.J_intra * intra_b - sys.J_inter * inter_b
    E_new = -sys.J_intra * intra_a - sys.J_inter * inter_a
    if accept!(alg, E_new, E_old)
        sys.cached_intra += intra_a - intra_b
        sys.cached_inter += inter_a - inter_b
    else
        for i in 1:n_move; sys.state[changed[n_move + i]] = 0; end
        for (i, m) in enumerate(start_m:end_m)
            poly[m] = site_to_coords(changed[i], sys.dims)
            sys.state[changed[i]] = n
        end
    end
    return nothing
end
