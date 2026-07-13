# ── Heisenberg systems (unit 3-vector spins) ──────────────────────────────────

"""
    HeisenbergSystem(topology; J=1)

Classical Heisenberg system −J Σ_{<ij>} s⃗ᵢ·s⃗ⱼ with unit-vector spins.
"""
function HeisenbergSystem(topo::Union{AbstractVector{<:Integer}, SimpleGraph}; J=1)
    prt = _init_partners(topo)
    return SpinSystem(HeisenbergSpin(), (PairInteraction(float(J), prt, Cache(0.0)),),
                      length(prt); geometry=_geometry(topo))
end
