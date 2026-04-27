# ── Helper: wrap coordinate into [0, L) ─────────────────────────────────────
@inline _wrap(x, L) = mod(x, L)

# ── Translation ─────────────────────────────────────────────────────────────

"""
    _try_translate!(sys::LatticePolymer, m, rng) -> Bool

Translate polymer `m` by one lattice step in a random direction.
Returns true if the move is geometrically valid (no overlap), false otherwise.
On failure the system state is unchanged.
"""
function _try_translate!(sys::LatticePolymer, m::Int, rng)
    dir = rand(rng, 1:6)
    N = sys.N
    id = m
    poly = sys.polymers[m]

    # Compute new positions
    new_sites = Vector{Int}(undef, N)
    for k in 1:N
        new_sites[k] = sys.nbrs[poly[k]][dir]
    end

    # Vacate old positions
    for k in 1:N
        sys.site_occupant[poly[k]] = 0
    end

    # Check for collisions
    for k in 1:N
        if sys.site_occupant[new_sites[k]] != 0
            for j in 1:N
                sys.site_occupant[poly[j]] = id
            end
            return false
        end
    end

    # Apply
    for k in 1:N
        sys.site_occupant[new_sites[k]] = id
        poly[k] = new_sites[k]
    end
    return true
end

# ── Local Flip ──────────────────────────────────────────────────────────────

"""
    _try_flip!(sys::LatticePolymer, m, rng) -> Bool

Flip a random interior monomer to its mirror position across the axis
connecting its two backbone neighbors.

new_pos = pos[k-1] + pos[k+1] - pos[k]  (in lattice coordinates with PBC)
"""
function _try_flip!(sys::LatticePolymer, m::Int, rng)
    N = sys.N
    N < 3 && return false
    id = m
    poly = sys.polymers[m]

    k = rand(rng, 2:N-1)
    old_site = poly[k]

    x0, y0, z0 = site_coords(old_site, sys.Lx, sys.Ly)
    x1, y1, z1 = site_coords(poly[k-1], sys.Lx, sys.Ly)
    x2, y2, z2 = site_coords(poly[k+1], sys.Lx, sys.Ly)

    dx1 = lattice_difference(x1, x0, sys.Lx)
    dy1 = lattice_difference(y1, y0, sys.Ly)
    dz1 = lattice_difference(z1, z0, sys.Lz)
    dx2 = lattice_difference(x2, x0, sys.Lx)
    dy2 = lattice_difference(y2, y0, sys.Ly)
    dz2 = lattice_difference(z2, z0, sys.Lz)

    nx = _wrap(x0 + dx1 + dx2, sys.Lx)
    ny = _wrap(y0 + dy1 + dy2, sys.Ly)
    nz = _wrap(z0 + dz1 + dz2, sys.Lz)
    new_site = site_index(nx, ny, nz, sys.Lx, sys.Ly)

    new_site == old_site && return false
    sys.site_occupant[new_site] != 0 && return false

    sys.site_occupant[old_site] = 0
    sys.site_occupant[new_site] = id
    poly[k] = new_site
    return true
end

# ── Slither (Reptation) ────────────────────────────────────────────────────

"""
    _try_slither!(sys::LatticePolymer, m, rng) -> Bool

Reptation move: remove one end monomer and grow at the other end.
"""
function _try_slither!(sys::LatticePolymer, m::Int, rng)
    N = sys.N
    id = m
    poly = sys.polymers[m]

    forward = rand(rng, Bool)
    nb_dir = rand(rng, 1:6)

    if forward
        remove_site = poly[1]
        new_site = sys.nbrs[poly[N]][nb_dir]

        new_site == remove_site && return false
        sys.site_occupant[new_site] != 0 && return false

        sys.site_occupant[remove_site] = 0
        sys.site_occupant[new_site] = id
        for k in 1:N-1
            poly[k] = poly[k+1]
        end
        poly[N] = new_site
    else
        remove_site = poly[N]
        new_site = sys.nbrs[poly[1]][nb_dir]

        new_site == remove_site && return false
        sys.site_occupant[new_site] != 0 && return false

        sys.site_occupant[remove_site] = 0
        sys.site_occupant[new_site] = id
        for k in N:-1:2
            poly[k] = poly[k-1]
        end
        poly[1] = new_site
    end
    return true
end

# ── Pivot ───────────────────────────────────────────────────────────────────

@inline function _rotate_3d(dx, dy, dz, axis, sin_a, cos_a)
    if axis == 1      # rotate around x
        return (dx, cos_a*dy - sin_a*dz, sin_a*dy + cos_a*dz)
    elseif axis == 2  # rotate around y
        return (cos_a*dx + sin_a*dz, dy, -sin_a*dx + cos_a*dz)
    else              # rotate around z
        return (cos_a*dx - sin_a*dy, sin_a*dx + cos_a*dy, dz)
    end
end

"""
    _try_pivot!(sys::LatticePolymer, m, rng) -> Bool

Pivot move: rotate one end of the chain by ±90° around a random axis.
"""
function _try_pivot!(sys::LatticePolymer, m::Int, rng)
    N = sys.N
    id = m
    poly = sys.polymers[m]

    pivot_k = rand(rng, 1:N)
    forward = rand(rng, Bool)
    axis = rand(rng, 1:3)
    sin_a = rand(rng, Bool) ? -1 : 1
    cos_a = 0

    pivot_site = poly[pivot_k]
    ox, oy, oz = site_coords(pivot_site, sys.Lx, sys.Ly)

    if forward
        start_k, end_k = pivot_k + 1, N
    else
        start_k, end_k = 1, pivot_k - 1
    end
    start_k > end_k && return false

    n_move = end_k - start_k + 1
    new_sites = Vector{Int}(undef, n_move)

    for (i, k) in enumerate(start_k:end_k)
        sx, sy, sz = site_coords(poly[k], sys.Lx, sys.Ly)
        dx = lattice_difference(sx, ox, sys.Lx)
        dy = lattice_difference(sy, oy, sys.Ly)
        dz = lattice_difference(sz, oz, sys.Lz)
        ndx, ndy, ndz = _rotate_3d(dx, dy, dz, axis, sin_a, cos_a)
        nx = _wrap(ox + ndx, sys.Lx)
        ny = _wrap(oy + ndy, sys.Ly)
        nz = _wrap(oz + ndz, sys.Lz)
        new_sites[i] = site_index(nx, ny, nz, sys.Lx, sys.Ly)
    end

    # Vacate old positions
    for k in start_k:end_k
        sys.site_occupant[poly[k]] = 0
    end

    # Check collisions
    for i in 1:n_move
        if sys.site_occupant[new_sites[i]] != 0
            for k in start_k:end_k
                sys.site_occupant[poly[k]] = id
            end
            return false
        end
    end

    # Apply
    for (i, k) in enumerate(start_k:end_k)
        sys.site_occupant[new_sites[i]] = id
        poly[k] = new_sites[i]
    end
    return true
end

# ── Double Bridge ───────────────────────────────────────────────────────────

@inline function _are_lattice_neighbors(sys::LatticePolymer, site1, site2)
    for nb in sys.nbrs[site1]
        nb == site2 && return true
    end
    return false
end

"""
    _try_double_bridge!(sys, m1, rng) -> (Bool, Int, Vector{Int}, Vector{Int})

Returns (success, m2, old_poly1, old_poly2). old_poly1/old_poly2 are copies
of the polymer positions before the swap (empty if move failed).
"""
function _try_double_bridge!(sys::LatticePolymer, m1::Int, rng)
    N = sys.N
    M = sys.M
    empty_vec = Int[]
    (M < 2 || N < 4) && return (false, 0, empty_vec, empty_vec)

    id1 = m1
    poly1 = sys.polymers[m1]
    position = rand(rng, 2:N-2)

    partners = Tuple{Int,Int}[]
    site1 = poly1[position]

    for nb in sys.nbrs[site1]
        occ = sys.site_occupant[nb]
        (occ == 0 || occ == id1) && continue
        id2 = occ
        poly2 = sys.polymers[id2]

        if position + 1 <= N && poly2[position + 1] == nb
            s2 = poly2[position]
            s1_next = poly1[position + 1]
            if _are_lattice_neighbors(sys, s2, s1_next)
                push!(partners, (id2, position + 1))
            end
        end
        if position - 1 >= 1 && poly2[position - 1] == nb
            s2 = poly2[position]
            s1_prev = poly1[position - 1]
            if _are_lattice_neighbors(sys, s2, s1_prev)
                push!(partners, (id2, position))
            end
        end
    end

    isempty(partners) && return (false, 0, empty_vec, empty_vec)

    id2, pos_cut = partners[rand(rng, 1:length(partners))]
    poly2 = sys.polymers[id2]

    n_forward = length(partners)
    n_reverse = _count_bridge_partners(sys, id2, position)
    n_forward != n_reverse && return (false, 0, empty_vec, empty_vec)

    # Save pre-swap state
    old_poly1 = copy(poly1)
    old_poly2 = copy(poly2)

    if position > N ÷ 2
        for k in pos_cut:N
            poly1[k], poly2[k] = poly2[k], poly1[k]
            sys.site_occupant[poly1[k]] = id1
            sys.site_occupant[poly2[k]] = id2
        end
    else
        for k in 1:pos_cut-1
            poly1[k], poly2[k] = poly2[k], poly1[k]
            sys.site_occupant[poly1[k]] = id1
            sys.site_occupant[poly2[k]] = id2
        end
    end
    return (true, id2, old_poly1, old_poly2)
end

function _count_bridge_partners(sys::LatticePolymer, m, position)
    poly = sys.polymers[m]
    N = sys.N
    site = poly[position]
    count = 0
    for nb in sys.nbrs[site]
        occ = sys.site_occupant[nb]
        (occ == 0 || occ == m) && continue
        other_poly = sys.polymers[occ]
        if position + 1 <= N && other_poly[position + 1] == nb
            s2 = other_poly[position]
            s1_next = poly[position + 1]
            if _are_lattice_neighbors(sys, s2, s1_next)
                count += 1
            end
        end
        if position - 1 >= 1 && other_poly[position - 1] == nb
            s2 = other_poly[position]
            s1_prev = poly[position - 1]
            if _are_lattice_neighbors(sys, s2, s1_prev)
                count += 1
            end
        end
    end
    return count
end

# ── Composite polymer_move! ─────────────────────────────────────────────────

function _select_move_type(sys::LatticePolymer, rng)
    N = sys.N
    M = sys.M
    if N == 1
        return :translate
    elseif N <= 3
        return rand(rng, Bool) ? :translate : :flip
    else
        r = rand(rng)
        if M >= 2
            if r < 0.2;     return :translate
            elseif r < 0.4; return :flip
            elseif r < 0.6; return :slither
            elseif r < 0.8; return :pivot
            else;            return :double_bridge
            end
        else
            if r < 0.25;     return :translate
            elseif r < 0.5;  return :flip
            elseif r < 0.75; return :slither
            else;             return :pivot
            end
        end
    end
end

"""
    polymer_move!(sys::LatticePolymer, alg::AbstractMetropolis)

Perform a random polymer move with Metropolis acceptance.
"""
function polymer_move!(sys::LatticePolymer, alg::AbstractMetropolis)
    rng = alg.rng
    m = rand(rng, 1:sys.M)

    old_intra = sys.num_intra_contacts
    old_inter = sys.num_inter_contacts
    E_old = sys.cached_energy

    move_type = _select_move_type(sys, rng)

    # Save state for undo
    old_poly_m = copy(sys.polymers[m])
    other_m = 0
    old_poly_m_pre = Int[]
    old_poly_other_pre = Int[]

    moved = if move_type == :double_bridge
        success, other_m, old_poly_m_pre, old_poly_other_pre = _try_double_bridge!(sys, m, rng)
        success
    elseif move_type == :translate
        _try_translate!(sys, m, rng)
    elseif move_type == :flip
        _try_flip!(sys, m, rng)
    elseif move_type == :slither
        _try_slither!(sys, m, rng)
    elseif move_type == :pivot
        _try_pivot!(sys, m, rng)
    else
        false
    end

    !moved && return nothing

    _recompute_energy!(sys)
    ΔE = sys.cached_energy - E_old

    if accept!(alg, ΔE)
        return nothing
    end

    # Reject: restore
    if move_type == :double_bridge
        _restore_polymer!(sys, m, old_poly_m_pre)
        _restore_polymer!(sys, other_m, old_poly_other_pre)
    else
        _restore_polymer!(sys, m, old_poly_m)
    end

    sys.num_intra_contacts = old_intra
    sys.num_inter_contacts = old_inter
    sys.cached_energy = E_old
    return nothing
end

"""
    polymer_move!(sys::LatticePolymer, alg::AbstractImportanceSampling)

Perform a random polymer move with generic importance sampling acceptance.
"""
function polymer_move!(sys::LatticePolymer, alg::AbstractImportanceSampling)
    rng = alg.rng
    m = rand(rng, 1:sys.M)

    E_old = energy(sys)
    old_intra = sys.num_intra_contacts
    old_inter = sys.num_inter_contacts

    move_type = _select_move_type(sys, rng)

    old_poly_m = copy(sys.polymers[m])
    other_m = 0
    old_poly_m_pre = Int[]
    old_poly_other_pre = Int[]

    moved = if move_type == :double_bridge
        success, other_m, old_poly_m_pre, old_poly_other_pre = _try_double_bridge!(sys, m, rng)
        success
    elseif move_type == :translate
        _try_translate!(sys, m, rng)
    elseif move_type == :flip
        _try_flip!(sys, m, rng)
    elseif move_type == :slither
        _try_slither!(sys, m, rng)
    elseif move_type == :pivot
        _try_pivot!(sys, m, rng)
    else
        false
    end

    !moved && return nothing

    _recompute_energy!(sys)
    E_new = sys.cached_energy

    if accept!(alg, E_new, E_old)
        return nothing
    end

    # Reject: restore
    if move_type == :double_bridge
        _restore_polymer!(sys, m, old_poly_m_pre)
        _restore_polymer!(sys, other_m, old_poly_other_pre)
    else
        _restore_polymer!(sys, m, old_poly_m)
    end

    sys.num_intra_contacts = old_intra
    sys.num_inter_contacts = old_inter
    sys.cached_energy = E_old
    return nothing
end

function _restore_polymer!(sys::LatticePolymer, m, old_sites)
    for site in sys.polymers[m]
        sys.site_occupant[site] = 0
    end
    sys.polymers[m] = old_sites
    for site in old_sites
        sys.site_occupant[site] = m
    end
end
