"""
    LatticePolymer{TJ<:Real} <: AbstractLatticeParticleSystem

Lattice polymer system: M self-avoiding polymers of N monomers each on a 3D
cubic lattice with periodic boundary conditions.

Energy:
  E = -J_intra * (intra-polymer contacts - bonds) - J_inter * inter-polymer contacts

where contacts count nearest-neighbor pairs of occupied sites belonging to the
same polymer (intra) or different polymers (inter), and bonds are the N-1
backbone connections per polymer that are always present.

# Fields
- `site_occupant::Vector{Int}` -- site -> polymer ID (1-based) or 0
- `polymers::Vector{Vector{Int}}` -- polymers[m][k] = site of k-th monomer (1-indexed)
- `M::Int` -- number of polymers
- `N::Int` -- monomers per polymer
- `Lx, Ly, Lz, N_sites::Int` -- lattice dimensions
- `nbrs::Vector{NTuple{6,Int}}` -- precomputed neighbor table
- `J_intra::TJ`, `J_inter::TJ` -- coupling constants
- `num_intra_contacts::Int` -- intra-polymer neighbor pairs / 2 minus bonds
- `num_inter_contacts::Int` -- inter-polymer neighbor pairs / 2
- `cached_energy::TJ`
"""
mutable struct LatticePolymer{TJ<:Real} <: AbstractLatticeParticleSystem
    site_occupant::Vector{Int}
    polymers::Vector{Vector{Int}}
    M::Int
    N::Int
    Lx::Int
    Ly::Int
    Lz::Int
    N_sites::Int
    nbrs::Vector{NTuple{6,Int}}
    J_intra::TJ
    J_inter::TJ
    num_intra_contacts::Int
    num_inter_contacts::Int
    cached_energy::TJ
end

"""
    LatticePolymer(Lx, Ly, Lz; M, N, J_intra=0.0, J_inter=1.0)

Construct an uninitialized LatticePolymer system. Call `init!` to place polymers.
"""
function LatticePolymer(Lx::Int, Ly::Int, Lz::Int; M::Int, N::Int,
                         J_intra::Real=0.0, J_inter::Real=1.0)
    N_sites = Lx * Ly * Lz
    @assert M * N <= N_sites "Not enough sites for $M polymers of length $N"
    nbrs = build_cubic_neighbors(Lx, Ly, Lz)
    TJ = promote_type(typeof(J_intra), typeof(J_inter))
    LatticePolymer{TJ}(
        zeros(Int, N_sites),
        [Int[] for _ in 1:M],
        M, N, Lx, Ly, Lz, N_sites, nbrs,
        TJ(J_intra), TJ(J_inter),
        0, 0, zero(TJ)
    )
end

LatticePolymer(L::Int; kwargs...) = LatticePolymer(L, L, L; kwargs...)

"""
    init!(sys::LatticePolymer, type::Symbol; rng=nothing)

Initialize polymer positions.

- `:ordered` -- place polymers as straight rods along the z-axis
- `:random` -- place polymers via random self-avoiding walks
"""
function init!(sys::LatticePolymer, type::Symbol; rng=nothing)
    fill!(sys.site_occupant, 0)
    for m in 1:sys.M
        empty!(sys.polymers[m])
    end

    if type == :ordered
        _init_ordered!(sys)
    elseif type == :random
        @assert rng !== nothing "Random initialization requires rng"
        _init_random!(sys, rng)
    else
        error("Unknown initialization type: $type")
    end

    _recompute_energy!(sys)
    return sys
end

function _init_ordered!(sys::LatticePolymer)
    # Place polymers as straight rods along z-axis, equally spaced
    placed = 0
    for m in 1:sys.M
        # Find a starting position that fits N monomers along z
        # Distribute polymers in a grid on the xy-plane
        grid_size = ceil(Int, sys.M^(1/2))
        mx = (m - 1) % grid_size
        my = ((m - 1) ÷ grid_size) % grid_size
        x = mx * (sys.Lx ÷ max(grid_size, 1))
        y = my * (sys.Ly ÷ max(grid_size, 1))

        for k in 1:sys.N
            z = (k - 1) % sys.Lz
            site = site_index(x % sys.Lx, y % sys.Ly, z, sys.Lx, sys.Ly)
            @assert sys.site_occupant[site] == 0 "Collision during ordered init at polymer $m, monomer $k"
            sys.site_occupant[site] = m
            push!(sys.polymers[m], site)
        end
        placed += 1
    end
end

function _init_random!(sys::LatticePolymer, rng)
    for m in 1:sys.M
        # Try to grow a self-avoiding walk
        success = false
        for attempt in 1:1000
            _clear_polymer!(sys, m)
            if _grow_saw!(sys, m, rng)
                success = true
                break
            end
        end
        if !success
            error("Failed to place polymer $m after 1000 attempts")
        end
    end
end

function _clear_polymer!(sys::LatticePolymer, m)
    for site in sys.polymers[m]
        sys.site_occupant[site] = 0
    end
    empty!(sys.polymers[m])
end

function _grow_saw!(sys::LatticePolymer, m, rng)
    # Pick random starting site
    start = rand(rng, 1:sys.N_sites)
    sys.site_occupant[start] != 0 && return false
    sys.site_occupant[start] = m
    push!(sys.polymers[m], start)

    for k in 2:sys.N
        # Try random neighbor of last monomer
        last_site = sys.polymers[m][end]
        # Shuffle neighbor order
        nb_indices = collect(1:6)
        for i in 6:-1:2
            j = rand(rng, 1:i)
            nb_indices[i], nb_indices[j] = nb_indices[j], nb_indices[i]
        end
        placed = false
        for ni in nb_indices
            nb = sys.nbrs[last_site][ni]
            if sys.site_occupant[nb] == 0
                sys.site_occupant[nb] = m
                push!(sys.polymers[m], nb)
                placed = true
                break
            end
        end
        if !placed
            _clear_polymer!(sys, m)
            return false
        end
    end
    return true
end

"""
    _recompute_energy!(sys::LatticePolymer)

Full recomputation of contacts and cached energy from scratch.
"""
function _recompute_energy!(sys::LatticePolymer)
    intra = 0
    inter = 0
    for m in 1:sys.M
        id = m
        for site in sys.polymers[m]
            for nb in sys.nbrs[site]
                occ = sys.site_occupant[nb]
                if occ == id
                    intra += 1
                elseif occ > 0
                    inter += 1
                end
            end
        end
    end
    # Each pair counted twice; subtract backbone bonds from intra
    intra = intra ÷ 2 - sys.M * (sys.N - 1)
    inter = inter ÷ 2
    sys.num_intra_contacts = intra
    sys.num_inter_contacts = inter
    sys.cached_energy = -sys.J_intra * intra - sys.J_inter * inter
    return nothing
end

"""
    energy(sys::LatticePolymer; full=false)

Return the system energy. If `full=true`, recompute from scratch.
"""
@inline function energy(sys::LatticePolymer; full=false)
    if full
        _recompute_energy!(sys)
    end
    return sys.cached_energy
end

"""
    _compute_polymer_contacts(sys::LatticePolymer, id) -> (intra, inter)

Count contacts for polymer `id`. Intra contacts are counted once
(by temporarily marking sites as empty). Inter contacts are counted once.
"""
function _compute_polymer_contacts(sys::LatticePolymer, id::Int)
    intra = 0
    inter = 0
    for site in sys.polymers[id]
        for nb in sys.nbrs[site]
            occ = sys.site_occupant[nb]
            if occ == id
                intra += 1
            elseif occ > 0
                inter += 1
            end
        end
    end
    # intra: each pair counted twice
    return (intra ÷ 2, inter)
end

"""
    _delta_energy_polymer_move!(sys, id, new_sites)

Compute delta energy for moving the monomers of polymer `id` from their
current positions to `new_sites`. Does NOT apply the move.

Returns (ΔE, Δintra, Δinter).
"""
function _delta_energy_full(sys::LatticePolymer)
    intra_old = sys.num_intra_contacts
    inter_old = sys.num_inter_contacts
    _recompute_energy!(sys)
    Δintra = sys.num_intra_contacts - intra_old
    Δinter = sys.num_inter_contacts - inter_old
    ΔE = -sys.J_intra * Δintra - sys.J_inter * Δinter
    return (ΔE, Δintra, Δinter)
end
