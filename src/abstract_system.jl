"""
    AbstractSystem

Base type for all systems in MonteCarloX.
"""
abstract type AbstractSystem end

"""
    init!(sys::AbstractSystem, type::Symbol; kwargs...)

Initialize the state of a system. Concrete methods are provided by
companion packages (MCXSpins, MCXSoftMatter, MCXLatticeMatter).
"""
function init! end

# ── Site interface ───────────────────────────────────────────────────────────
# The contract a model implements to drive local-transition dynamics: SiteEvents /
# NFoldRates (n-fold way, contact processes) and HeatBath. Concrete methods are
# provided by companion packages; NFoldRates and HeatBath additionally need
# `delta_energy`, and applying a move goes through `modify!(sys, i, s_new)`.

"""
    nsites(sys)

Number of sites of `sys`.
"""
function nsites end

"""
    local_states(sys, i)

Tuple of alternative states available at site `i` (the events `(i, s)` iterate over these).
"""
function local_states end

"""
    partners(sys, i)

Sites whose rates are invalidated when site `i` changes (its interaction neighbours).
"""
function partners end

"""
    delta_energy(sys, i, s_new)

Energy change from setting site `i` to `s_new`. Needed by rate rules that read a Boltzmann
weight (`NFoldRates`, `HeatBath`).
"""
function delta_energy end