"""
    _find_bridge_partners(sys, n, pos)

Find valid double-bridge partner candidates at monomer position `pos` on polymer `n`.

A double bridge reconnects two polymers that cross at adjacent positions:

    Before:  n1: ...A—B...    n2: ...C—D...
    After:   n1: ...A—D...    n2: ...C—B...

Forward (+1): A=n[pos] adjacent to D=n2[pos+1], and C=n2[pos] adjacent to B=n[pos+1]
Backward (-1): A=n[pos] adjacent to D=n2[pos-1], and C=n2[pos] adjacent to B=n[pos-1]

Returns vector of `(partner_id, sign)` tuples where sign ∈ {+1, -1}.
"""
function _find_bridge_partners(sys::LatticePolymer{D}, n::Int, pos::Int) where {D}
    poly = sys.polymers[n]
    M = polymer_length(sys, n)
    site_A = coords_to_site(poly[pos], sys.dims)

    partners = Tuple{Int, Int}[]

    @inbounds for nb in sys.neighbors[site_A]
        n2 = sys.state[nb]
        (n2 == 0 || n2 == n) && continue
        polymer_length(sys, n2) == M || continue

        poly2 = sys.polymers[n2]

        # Forward: A adjacent to D=poly2[pos+1], check C=poly2[pos] adjacent to B=poly[pos+1]
        if pos + 1 <= M && coords_to_site(poly2[pos + 1], sys.dims) == nb
            site_C = coords_to_site(poly2[pos], sys.dims)
            site_B = coords_to_site(poly[pos + 1], sys.dims)
            @inbounds for nb2 in sys.neighbors[site_C]
                nb2 == site_B && push!(partners, (n2, +1))
            end
        end

        # Backward: A adjacent to D=poly2[pos-1], check C=poly2[pos] adjacent to B=poly[pos-1]
        if pos - 1 >= 1 && coords_to_site(poly2[pos - 1], sys.dims) == nb
            site_C = coords_to_site(poly2[pos], sys.dims)
            site_B = coords_to_site(poly[pos - 1], sys.dims)
            @inbounds for nb2 in sys.neighbors[site_C]
                nb2 == site_B && push!(partners, (n2, -1))
            end
        end
    end

    return partners
end

"""
    _swap_segments!(sys, n1, n2, range)

Swap monomer coordinates and ownership between polymers `n1` and `n2` over `range`.
Self-inverse: calling twice restores the original state.
"""
@inline function _swap_segments!(sys::LatticePolymer, n1::Int, n2::Int, range)
    poly1 = sys.polymers[n1]
    poly2 = sys.polymers[n2]
    for m in range
        s1 = coords_to_site(poly1[m], sys.dims)
        s2 = coords_to_site(poly2[m], sys.dims)
        sys.state[s1] = n2
        sys.state[s2] = n1
        poly1[m], poly2[m] = poly2[m], poly1[m]
    end
end

"""
    double_bridge!(sys, alg)

Exchange segments between two equal-length polymers at a randomly chosen cut.
Requires at least 2 polymers of length >= 4.
"""
function double_bridge!(sys::LatticePolymer{D}, alg::AbstractMarkovChainMonteCarlo) where {D}
    N = num_polymers(sys)
    N < 2 && return nothing

    n1 = rand(alg.rng, 1:N)
    M = polymer_length(sys, n1)
    M < 4 && return nothing

    pos = rand(alg.rng, 2:M-2)

    partners = _find_bridge_partners(sys, n1, pos)
    isempty(partners) && return nothing

    n2, sign = partners[rand(alg.rng, 1:length(partners))]

    # Detailed balance: reverse proposal count must match
    reverse_partners = _find_bridge_partners(sys, n2, pos)
    length(partners) != length(reverse_partners) && return nothing

    # Cut position depends on bridge direction; swap the shorter segment
    pos_cut = sign == +1 ? pos + 1 : pos
    if pos > M ÷ 2
        range_start, range_end = pos_cut, M
    else
        range_start, range_end = 1, pos_cut - 1
    end

    # Collect sites for contact counting
    range = range_start:range_end
    n_swap = length(range)
    sites = Vector{Int}(undef, 2 * n_swap)
    for (i, m) in enumerate(range)
        sites[i]          = coords_to_site(sys.polymers[n1][m], sys.dims)
        sites[n_swap + i] = coords_to_site(sys.polymers[n2][m], sys.dims)
    end

    intra_b, inter_b = 0, 0
    for i in eachindex(sites)
        ci, ce = site_contacts(sys, sites[i]); intra_b += ci; inter_b += ce
    end

    _swap_segments!(sys, n1, n2, range)

    intra_a, inter_a = 0, 0
    for i in eachindex(sites)
        ci, ce = site_contacts(sys, sites[i]); intra_a += ci; inter_a += ce
    end

    E_old = -sys.J_intra * intra_b - sys.J_inter * inter_b
    E_new = -sys.J_intra * intra_a - sys.J_inter * inter_a
    if accept!(alg, E_new, E_old)
        sys.cached_intra += intra_a - intra_b
        sys.cached_inter += inter_a - inter_b
    else
        _swap_segments!(sys, n1, n2, range)
    end
    return nothing
end
