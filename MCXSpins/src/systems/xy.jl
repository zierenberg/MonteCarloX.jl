# ── XY systems (planar rotors) ────────────────────────────────────────────────

"""
    XYSystem(topology; J=1, h=0)

XY (planar rotor) system −J Σ_{<ij>} cos(θi−θj); a complex `h` is an in-plane field.
The rotation proposal half-width is an update keyword: `spin_flip!(sys, alg; Δθ)`.
"""
function XYSystem(topo::Union{AbstractVector{<:Integer}, SimpleGraph}; J=1, h=0)
    prt = _init_partners(topo)
    pair = PairInteraction(float(J), prt, Cache(0.0))
    ints = iszero(h) ? (pair,) : (pair, ExternalField(complex(float(h))))
    return SpinSystem(XYSpin(), ints, length(prt); geometry=_geometry(topo))
end
