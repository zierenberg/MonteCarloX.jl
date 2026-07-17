using MonteCarloX
using MonteCarloX: _sample_accept
using Random
using Test

@testset "balance functions" begin
    logRs = (-3.0, -1.0, -0.2, 0.0, 0.2, 1.0, 3.0)

    @testset "detailed balance f(logR)/f(-logR) = e^logR" begin
        for bal in (MetropolisBalance(), GlauberBalance())
            for x in logRs
                fx  = acceptance_probability(bal, x)
                fmx = acceptance_probability(bal, -x)
                @test isapprox(fx / fmx, exp(x); rtol=1e-12)
            end
        end
    end

    @testset "acceptance_probability == transition_rate (shipped members)" begin
        for bal in (MetropolisBalance(), GlauberBalance())
            for x in logRs
                @test acceptance_probability(bal, x) == transition_rate(bal, x)
            end
        end
    end

    @testset "explicit formulas" begin
        for x in logRs
            @test acceptance_probability(MetropolisBalance(), x) == min(1.0, exp(x))
            @test isapprox(acceptance_probability(GlauberBalance(), x), 1 / (1 + exp(-x)); rtol=1e-12)
        end
        # uphill move: Metropolis accepts with probability 1, Glauber with 1/2 at logR=0
        @test acceptance_probability(MetropolisBalance(), 5.0) == 1.0
        @test acceptance_probability(GlauberBalance(), 0.0) == 0.5
    end

    @testset "n-fold rate parity with the historical rule" begin
        # old hard-coded n-fold Metropolis rate: logR >= 0 ? 1.0 : exp(logR)
        for x in logRs
            @test transition_rate(MetropolisBalance(), x) == (x >= 0 ? 1.0 : exp(x))
        end
    end

    @testset "sampling preserves RNG streams" begin
        # Metropolis uphill (logR > 0) draws NO random number (short-circuit)
        r = Xoshiro(1); before = copy(r)
        @test _sample_accept(MetropolisBalance(), r, 2.0) == true
        @test r == before
        # Metropolis downhill draws exactly one
        r = Xoshiro(1); ref = Xoshiro(1); _ = rand(ref)
        _sample_accept(MetropolisBalance(), r, -1.0)
        @test r == ref
        # Glauber always draws exactly one (even uphill)
        r = Xoshiro(1); ref = Xoshiro(1); _ = rand(ref)
        _sample_accept(GlauberBalance(), r, 2.0)
        @test r == ref
    end

    @testset "empirical acceptance matches probability" begin
        for (bal, x) in ((MetropolisBalance(), -0.7), (GlauberBalance(), 0.3))
            rng = Xoshiro(42)
            n = 200_000
            acc = 0
            for _ in 1:n
                acc += _sample_accept(bal, rng, x)
            end
            @test isapprox(acc / n, acceptance_probability(bal, x); atol=5e-3)
        end
    end

    @testset "MetropolisAlgorithm vs GlauberAlgorithm carry the right balance" begin
        @test balance(MetropolisAlgorithm(Xoshiro(1); β=1.0)) === MetropolisBalance()
        @test balance(GlauberAlgorithm(Xoshiro(1); β=1.0)) === GlauberBalance()
    end
end
