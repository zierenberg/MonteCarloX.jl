# ── Ising-family systems (Spin(1//2)) ─────────────────────────────────────────

"""
    IsingSystem(topology; J=1, h=0)
    IsingSystem(J::SparseMatrixCSC; h=0, geometry=nothing)

Ising system −J Σ_{<ij>} σσ − h Σσ. The topology is a dims vector (periodic hypercubic
lattice, `[L, L]`-style), a `SimpleGraph`, or a sparse coupling matrix J_ij (spin glasses,
directed couplings — then the term is −½ Σ_{ij} J_ij σσ and the uniform `J` is absorbed
into the matrix).
"""
function IsingSystem(topo::Union{AbstractVector{<:Integer}, SimpleGraph}; J=1, h=0)
    prt = _init_partners(topo)
    pair = PairInteraction(J, prt)
    ints = iszero(h) ? (pair,) : (pair, ExternalField(h))
    return SpinSystem(Spin(1//2), ints, length(prt); geometry=_geometry(topo))
end

function IsingSystem(J::SparseMatrixCSC; h=0, geometry=nothing)
    pair = PairInteractionMatrix(J)
    ints = iszero(h) ? (pair,) : (pair, ExternalField(h))
    return SpinSystem(Spin(1//2), ints, size(J, 1); geometry=geometry)
end

"""
    VisionConeIsingSystem(dims; κ, J=1)

Vision-cone (nonreciprocal) Ising system (Garcés–Levis): reciprocal coupling `J` to all
lattice neighbors plus the extra cone coupling `κ` forward — composed as
`(PairInteraction(J), VisionConeInteraction(κ))`. The pair term remains a proper
Hamiltonian term (see `hamiltonian_energy`); the full `energy` throws — energy-like
diagnostics of the nonreciprocal dynamics are observables.
"""
function VisionConeIsingSystem(dims::AbstractVector{<:Integer}; κ, J=1)
    d = Tuple(Int.(dims))
    ints = (PairInteraction(J, _build_lattice_neighbors(d)), VisionConeInteraction(κ, d))
    return SpinSystem(Spin(1//2), ints, prod(d); geometry=d)
end

"""
    HopfieldSystem(patterns::AbstractMatrix{<:Integer})

Hopfield associative memory: Ising spins with J_ij = (1/N) Σ_μ ξ_i^μ ξ_j^μ from the pattern
matrix ξ (N×P, entries ±1). Experimental: interface and normalization may change.
"""
function HopfieldSystem(patterns::AbstractMatrix{<:Integer})
    @warn "HopfieldSystem is experimental" maxlog=1
    return SpinSystem(Spin(1//2), (PairInteractionMatrix(hopfield_J(patterns)),),
                      size(patterns, 1))
end

"""
    EdwardsAndersonSystem(dims; rng, dist=:bimodal)

Edwards–Anderson spin glass on a periodic hypercubic lattice with quenched random bonds:
`dist=:bimodal` draws J_ij = ±1, `dist=:gaussian` draws J_ij ~ N(0,1).
"""
function EdwardsAndersonSystem(dims::AbstractVector{<:Integer}; rng, dist::Symbol=:bimodal)
    draw = dist === :bimodal ? (r -> rand(r, (-1.0, 1.0))) :
           dist === :gaussian ? randn :
           throw(ArgumentError("dist must be :bimodal or :gaussian (got $dist)"))
    return IsingSystem(lattice_random_J(Tuple(Int.(dims)), rng; dist=draw);
                       geometry=Tuple(Int.(dims)))
end
