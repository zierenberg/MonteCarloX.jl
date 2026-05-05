"""
    translate_move!(sys, alg; Δ=1)

Translate an entire polymer by a random displacement where each
component is drawn uniformly from `{-Δ, …, Δ}`.
"""
function translate_move!(sys::LatticePolymer{D}, alg::AbstractImportanceSampling; Δ::Int=1) where {D}
    n = rand(alg.rng, 1:num_polymers(sys))
    M = polymer_length(sys, n)
    poly = sys.polymers[n]

    shift = MVector{D,Int}(ntuple(_ -> rand(alg.rng, -Δ:Δ), Val(D)))
    all(iszero, shift) && return nothing

    # Old and new sites + collision check
    changed = zeros(Int, 2M)
    for m in 1:M
        changed[m] = coords_to_site(poly[m], sys.dims)
        nc = MVector{D,Int}(poly[m])
        for d in 1:D; nc[d] = apply_pbc(nc[d] + shift[d], sys.dims[d]); end
        changed[M+m] = coords_to_site(nc, sys.dims)
        occ = sys.state[changed[M+m]]
        occ != 0 && occ != n && return nothing
    end

    # Contacts before
    intra_b, inter_b = 0, 0
    for m in 1:M
        ci, ce = site_contacts(sys, changed[m]); intra_b += ci; inter_b += ce
    end

    # Apply: clear all old first, then write new (order matters when old/new overlap)
    for m in 1:M; sys.state[changed[m]] = 0; end
    for m in 1:M
        for d in 1:D; poly[m][d] = apply_pbc(poly[m][d] + shift[d], sys.dims[d]); end
        sys.state[changed[M+m]] = n
    end

    # Contacts after
    intra_a, inter_a = 0, 0
    for m in 1:M
        ci, ce = site_contacts(sys, changed[M+m]); intra_a += ci; inter_a += ce
    end

    E_old = -sys.J_intra * intra_b - sys.J_inter * inter_b
    E_new = -sys.J_intra * intra_a - sys.J_inter * inter_a
    if accept!(alg, E_new, E_old)
        sys.cached_intra += intra_a - intra_b
        sys.cached_inter += inter_a - inter_b
    else
        for m in 1:M; sys.state[changed[M+m]] = 0; end
        for m in 1:M
            for d in 1:D; poly[m][d] = apply_pbc(poly[m][d] - shift[d], sys.dims[d]); end
            sys.state[changed[m]] = n
        end
    end
    return nothing
end
