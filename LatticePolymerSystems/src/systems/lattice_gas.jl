"""
    LatticeGas{TJ<:Real} <: AbstractLatticeParticleSystem

Lattice gas on a 3D cubic lattice with periodic boundary conditions.

Energy: E = -J * num_contacts, where num_contacts counts nearest-neighbor
occupied pairs. Dynamics conserve particle number (canonical ensemble)
via Kawasaki swap moves.

# Fields
- `occupation::Vector{Bool}` -- per-site occupation
- `occupied_sites::Vector{Int}` -- indices of occupied sites (for O(1) random selection)
- `empty_sites::Vector{Int}` -- indices of empty sites
- `site_to_occupied_idx::Vector{Int}` -- reverse map: site -> position in occupied_sites (0 if empty)
- `site_to_empty_idx::Vector{Int}` -- reverse map: site -> position in empty_sites (0 if occupied)
- `Lx, Ly, Lz::Int` -- lattice dimensions
- `N_sites::Int` -- total number of sites
- `nbrs::Vector{NTuple{6,Int}}` -- precomputed neighbor table
- `J::TJ` -- coupling constant
- `N_particles::Int` -- number of particles (conserved)
- `cached_energy::TJ` -- cached total energy
- `num_contacts::Int` -- number of nearest-neighbor occupied pairs
"""
mutable struct LatticeGas{TJ<:Real} <: AbstractLatticeParticleSystem
    occupation::Vector{Bool}
    occupied_sites::Vector{Int}
    empty_sites::Vector{Int}
    site_to_occupied_idx::Vector{Int}
    site_to_empty_idx::Vector{Int}
    Lx::Int
    Ly::Int
    Lz::Int
    N_sites::Int
    nbrs::Vector{NTuple{6,Int}}
    J::TJ
    N_particles::Int
    cached_energy::TJ
    num_contacts::Int
end

"""
    LatticeGas(Lx, Ly, Lz; J=1.0, N_particles)

Construct an uninitialized LatticeGas. Call `init!` to place particles.
"""
function LatticeGas(Lx::Int, Ly::Int, Lz::Int; J::Real=1.0, N_particles::Int)
    N_sites = Lx * Ly * Lz
    @assert 0 < N_particles <= N_sites "Need 0 < N_particles <= N_sites"
    nbrs = build_cubic_neighbors(Lx, Ly, Lz)
    TJ = typeof(J)
    LatticeGas{TJ}(
        zeros(Bool, N_sites),
        Int[], Int[],
        zeros(Int, N_sites), zeros(Int, N_sites),
        Lx, Ly, Lz, N_sites, nbrs,
        J, N_particles,
        zero(TJ), 0
    )
end

LatticeGas(L::Int; kwargs...) = LatticeGas(L, L, L; kwargs...)

"""
    init!(sys::LatticeGas, type::Symbol; rng)

Initialize particle positions.

- `:random` -- place particles uniformly at random
- `:ordered` -- fill the first N_particles sites
"""
function init!(sys::LatticeGas, type::Symbol; rng=nothing)
    fill!(sys.occupation, false)
    empty!(sys.occupied_sites)
    empty!(sys.empty_sites)
    fill!(sys.site_to_occupied_idx, 0)
    fill!(sys.site_to_empty_idx, 0)

    if type == :random
        @assert rng !== nothing "Random initialization requires rng"
        sites = randperm(rng, sys.N_sites)
        for i in 1:sys.N_particles
            sys.occupation[sites[i]] = true
        end
    elseif type == :ordered
        for i in 1:sys.N_particles
            sys.occupation[i] = true
        end
    else
        error("Unknown initialization type: $type")
    end

    # Build index arrays
    for site in 1:sys.N_sites
        if sys.occupation[site]
            push!(sys.occupied_sites, site)
            sys.site_to_occupied_idx[site] = length(sys.occupied_sites)
        else
            push!(sys.empty_sites, site)
            sys.site_to_empty_idx[site] = length(sys.empty_sites)
        end
    end

    _recompute_energy!(sys)
    return sys
end

"""
    _recompute_energy!(sys::LatticeGas)

Full recomputation of num_contacts and cached_energy from scratch.
"""
function _recompute_energy!(sys::LatticeGas)
    contacts = 0
    for site in 1:sys.N_sites
        sys.occupation[site] || continue
        for nb in sys.nbrs[site]
            if sys.occupation[nb]
                contacts += 1
            end
        end
    end
    sys.num_contacts = contacts ÷ 2  # each pair counted twice
    sys.cached_energy = -sys.J * sys.num_contacts
    return nothing
end

"""
    energy(sys::LatticeGas; full=false)

Return the system energy. If `full=true`, recompute from scratch.
"""
@inline function energy(sys::LatticeGas; full=false)
    if full
        _recompute_energy!(sys)
    end
    return sys.cached_energy
end

"""
    _count_occupied_neighbors(sys::LatticeGas, site) -> Int

Count occupied neighbors of a site.
"""
@inline function _count_occupied_neighbors(sys::LatticeGas, site)
    count = 0
    for nb in sys.nbrs[site]
        count += sys.occupation[nb]
    end
    return count
end

"""
    delta_energy(sys::LatticeGas, occ_site, emp_site) -> TJ

Energy change for swapping an occupied site with an empty site (Kawasaki move).
"""
@inline function delta_energy(sys::LatticeGas, occ_site::Int, emp_site::Int)
    # Contacts lost by removing particle from occ_site
    contacts_old = _count_occupied_neighbors(sys, occ_site)
    # Contacts gained by placing particle at emp_site
    contacts_new = _count_occupied_neighbors(sys, emp_site)
    # Correction: if occ_site and emp_site are neighbors, emp_site sees
    # occ_site as occupied, but the particle is leaving occ_site
    if _are_neighbors(sys, occ_site, emp_site)
        contacts_new -= 1
    end
    delta_contacts = contacts_new - contacts_old
    return -sys.J * delta_contacts
end

@inline function _are_neighbors(sys::LatticeGas, site1, site2)
    for nb in sys.nbrs[site1]
        nb == site2 && return true
    end
    return false
end

"""
    modify!(sys::LatticeGas, occ_site, emp_site, delta_contacts)

Apply a Kawasaki swap: move particle from occ_site to emp_site.
Updates occupation, index arrays, and cached energy.
"""
function modify!(sys::LatticeGas, occ_site::Int, emp_site::Int, delta_contacts::Int)
    # Update occupation
    sys.occupation[occ_site] = false
    sys.occupation[emp_site] = true

    # Update occupied_sites: replace occ_site with emp_site
    idx_occ = sys.site_to_occupied_idx[occ_site]
    sys.occupied_sites[idx_occ] = emp_site
    sys.site_to_occupied_idx[emp_site] = idx_occ
    sys.site_to_occupied_idx[occ_site] = 0

    # Update empty_sites: replace emp_site with occ_site
    idx_emp = sys.site_to_empty_idx[emp_site]
    sys.empty_sites[idx_emp] = occ_site
    sys.site_to_empty_idx[occ_site] = idx_emp
    sys.site_to_empty_idx[emp_site] = 0

    # Update energy
    sys.num_contacts += delta_contacts
    sys.cached_energy = -sys.J * sys.num_contacts
    return nothing
end
