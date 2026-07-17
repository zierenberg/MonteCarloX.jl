# ── Blume–Capel systems (Spin(1)) ─────────────────────────────────────────────

"""
    BlumeCapelSystem(topology; J=1, D=0, h=0)
    BlumeCapelSystem(J::SparseMatrixCSC; D=0, h=0, geometry=nothing)

Blume–Capel system −J Σ_{<ij>} σσ + D Σσ² − h Σσ with σ ∈ {−1,0,+1}. The topology is a
dims vector (periodic hypercubic lattice), a `SimpleGraph`, or a sparse coupling matrix
J_ij.
"""
function BlumeCapelSystem(topo::Union{AbstractVector{<:Integer}, SimpleGraph}; J=1, D=0, h=0)
    prt = _init_partners(topo)
    pair = PairInteraction(J, prt)
    ints = iszero(h) ? (pair, CrystalField(D)) : (pair, ExternalField(h), CrystalField(D))
    return SpinSystem(Spin(1), ints, length(prt); geometry=_geometry(topo))
end

function BlumeCapelSystem(J::SparseMatrixCSC; D=0, h=0, geometry=nothing)
    pair = PairInteractionMatrix(J)
    ints = iszero(h) ? (pair, CrystalField(D)) : (pair, ExternalField(h), CrystalField(D))
    return SpinSystem(Spin(1), ints, size(J, 1); geometry=geometry)
end

"""
    VisionConeBlumeCapelSystem(dims; κ, D=0, J=1)

Vision-cone (nonreciprocal) Blume–Capel system, composed as `(PairInteraction(J),
VisionConeInteraction(κ), CrystalField(D))`. The pair and crystal-field terms remain proper
Hamiltonian terms (see `hamiltonian_energy`) — valid multicanonical coordinates.
"""
function VisionConeBlumeCapelSystem(dims::AbstractVector{<:Integer}; κ, D=0, J=1)
    d = Tuple(Int.(dims))
    ints = (PairInteraction(J, _build_lattice_neighbors(d)), VisionConeInteraction(κ, d),
            CrystalField(D))
    return SpinSystem(Spin(1), ints, prod(d); geometry=d)
end
