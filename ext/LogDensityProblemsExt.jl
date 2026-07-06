module LogDensityProblemsExt

# Bridge a MonteCarloX `FunctionEnsemble` to the LogDensityProblems.jl interface, so a
# Bayesian target expressed as a `logweight` over a parameter vector can be handed to
# gradient-based samplers (AdvancedHMC/NUTS, DynamicHMC) and AD packages. Requires the
# ensemble's `dimension` to be set. Auto-loaded when both MonteCarloX and
# LogDensityProblems are present. Order-0 only — an AD package (e.g. via
# LogDensityProblemsAD) supplies the gradient.

using MonteCarloX: FunctionEnsemble, logweight
import LogDensityProblems as LDP

LDP.capabilities(::Type{<:FunctionEnsemble}) = LDP.LogDensityOrder{0}()
LDP.logdensity(e::FunctionEnsemble, x)       = logweight(e, x)
function LDP.dimension(e::FunctionEnsemble)
    e.dimension === nothing && throw(ArgumentError(
        "this FunctionEnsemble has no dimension; construct it as " *
        "FunctionEnsemble(logdensity; dimension = d) to use it with gradient-based samplers"))
    return e.dimension
end

end # module LogDensityProblemsExt
