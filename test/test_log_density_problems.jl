using MonteCarloX
using LogDensityProblems
using Random
using Test

# A FunctionEnsemble with `dimension` set bridges to LogDensityProblems (loaded here →
# extension active) and remains usable directly by MonteCarloX's own sampler.
function test_log_density_problems()
    pass = true
    t = FunctionEnsemble(θ -> -0.5 * sum(abs2, θ); dimension = 3)

    pass &= check(LogDensityProblems.dimension(t) == 3, "ldp: dimension\n")
    pass &= check(isapprox(LogDensityProblems.logdensity(t, [1.0, 0.0, 0.0]), -0.5),
                  "ldp: logdensity == logweight\n")
    pass &= check(LogDensityProblems.capabilities(typeof(t)) === LogDensityProblems.LogDensityOrder{0}(),
                  "ldp: capabilities order 0\n")

    # Without a dimension the bridge refuses with an informative error.
    bare = FunctionEnsemble(θ -> -0.5 * sum(abs2, θ))
    pass &= check(
        try
            LogDensityProblems.dimension(bare); false
        catch e
            e isa ArgumentError
        end, "ldp: missing dimension throws\n")

    # An uphill move (toward higher density) is always accepted by the MonteCarloX sampler.
    alg = MarkovChainMonteCarlo(Xoshiro(1), t)
    pass &= check(accept!(alg, [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]), "ldp: usable by MCX accept!\n")
    return pass
end

@testset "LogDensityProblems bridge" begin
    @test test_log_density_problems()
end
