module MonteCarloX
# dependencies
using Random
using StatsBase
using LinearAlgebra
using StaticArrays
# TODO: ?exports that are relevant to run simulations in MonteCarloX
# using Reexports
# @reexport using Random
# @reexport using StatsBase

include("utils.jl")
include("event_handler.jl")
include("rng.jl")

# Equilibrium
include("importance_sampling.jl")
include("reweighting.jl")

# Non-equilibrium
include("kinetic_monte_carlo.jl")
include("poisson_process.jl")
include("gillespie.jl")

# TODO: move to external SpinSystems.jl package
include("cluster_wolff.jl")

# algorithms
export  Metropolis,
        Gillespie,
        KineticMonteCarlo,
        InhomogeneousPoisson,
        InhomogeneousPoissonPiecewiseDecreasing

# functions
export  # base
        # equilibrium
        accept,
        sweep,
        # update -> will be moved to test/utils.jl for now and later to SpinSystems.jl
        # non-equilibrium
        next_event,
        next_time,
        next,
        advance!,
        initialize

# reweighting (needs makeover)
export  # reweighting
        expectation_value_from_timeseries,
        distribution_from_timeseries

# helper
export  log_sum,
        binary_search,
        random_element,
        kldivergence,
        # event handler
        AbstractEventHandlerRate,
        ListEventRateSimple,
        ListEventRateActiveMask,
        # rng
        MutableRandomNumbers,
        reset




# Check georges 2nd workshop notebook on github
#    i::Int = StatsBase.binindex(d.h, x) -> remember this for custom things similar to EmpiricalDistributions.jl (not well documented though)


end # module

# Maybe embedd this into StatisticalPhysics.jl the including SpinSystems.jl PolymerSystems.jl etc ;)
# TODO: external modules obviously in external modules
# ("DirectedPercolation.jl") [inlcuding ContactProcess, CellularAutomatoa, etc but not as modules but as models]
# ("NeuralNetworks.jl) -> maybe NeuralSystems.jl?
# ("ComplexNetworks.jl)
