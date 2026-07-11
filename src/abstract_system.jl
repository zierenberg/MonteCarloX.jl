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