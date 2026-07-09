using MonteCarloX
using LogDensityProblems
using Random
using Test

# LogDensityTarget bridges to LogDensityProblems (loaded here → extension active) and is
# also usable directly by MonteCarloX's own sampler.
function test_log_density_target()
    pass = true
    t = LogDensityTarget(θ -> -0.5 * sum(abs2, θ), 3)

    pass &= check(LogDensityProblems.dimension(t) == 3, "ldt: dimension\n")
    pass &= check(isapprox(LogDensityProblems.logdensity(t, [1.0, 0.0, 0.0]), -0.5),
                  "ldt: logdensity == logweight\n")
    pass &= check(LogDensityProblems.capabilities(typeof(t)) === LogDensityProblems.LogDensityOrder{0}(),
                  "ldt: capabilities order 0\n")

    # An uphill move (toward higher density) is always accepted by the MonteCarloX sampler.
    alg = MetropolisHastingsAlgorithm(Xoshiro(1), t)
    pass &= check(accept!(alg, [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]), "ldt: usable by MCX accept!\n")
    return pass
end

@testset "LogDensityTarget" begin
    @test test_log_density_target()
end
