module DistributionsExt

# Bridge Distributions.jl into the ensemble/logweight protocol: a normalized
# `Distribution` acts as a source or target in `reweight` by exposing its `logpdf`
# as a logweight. Auto-loaded when both MonteCarloX and Distributions are present.

using MonteCarloX
using Distributions: Distribution, logpdf

MonteCarloX._as_ensemble(d::Distribution) = FunctionEnsemble(x -> logpdf(d, x))

end # module DistributionsExt
