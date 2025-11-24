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

#TODO List
# * Think about the interface of algorithmic structs; should they include basic
#   elements such as weight functions or event handler?
# * Move cluster_wolff.jl to designated SpinSystems.jl, this is an update not
#   MonteCarlo

# Equilibrium
include("importance_sampling.jl")
include("reweighting.jl")

# Non-equilibrium
include("kinetic_monte_carlo.jl")
include("poisson_process.jl")
include("gillespie.jl")

# move to external
include("cluster_wolff.jl")

# algorithms
export  Metropolis,
        Gillespie,
        KineticMonteCarlo,
        PoissonProcess
        # InhomogeneousPoisson,
        # InhomogeneousPoissonPiecewiseDecreasing

# functions (equilibrium)
export accept,
       sweep
        # update -> will be moved to test/utils.jl for now and later to SpinSystems.jl

# funtions (non-equilibrium)
export  next_event,
        next_time,
        next,
        advance!,
        init
        # initialize,

# reweighting (needs makeover)
export  expectation_value_from_timeseries,
        distribution_from_timeseries
        # reweighting

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




end # module

# Maybe embedd this into StatisticalPhysics.jl, which could include
# SpinSystems.jl, PolymerSystems.jl, DirectedPercolation, (NeuralNetworks,)
# ComplexSystems, ComlexNetworks etc ;)
