module LogDensityProblemsExt

# Bridge a MonteCarloX `LogDensityTarget` to the LogDensityProblems.jl interface, so a
# Bayesian target expressed as a `logweight` over a parameter vector can be handed to
# gradient-based samplers (AdvancedHMC/NUTS, DynamicHMC) and AD packages. Auto-loaded
# when both MonteCarloX and LogDensityProblems are present. Order-0 only — an AD
# package (e.g. via LogDensityProblemsAD) supplies the gradient.

using MonteCarloX: LogDensityTarget, logweight
import LogDensityProblems as LDP

LDP.capabilities(::Type{<:LogDensityTarget}) = LDP.LogDensityOrder{0}()
LDP.dimension(t::LogDensityTarget)           = t.dimension
LDP.logdensity(t::LogDensityTarget, x)       = logweight(t, x)

end # module LogDensityProblemsExt
