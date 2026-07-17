using MonteCarloX
using Random
using Test

@testset "EventRateTree (Fenwick)" begin
    @testset "construction & indexing" begin
        h = EventRateTree(5)
        @test length(h) == 5
        @test total_rate(h) == 0.0
        @test all(h[i] == 0.0 for i in 1:5)

        h[2] = 3.0
        h[4] = 1.5
        @test (h[2], h[4]) == (3.0, 1.5)
        @test total_rate(h) ≈ 4.5
    end

    @testset "total_rate tracks Σrates through random updates" begin
        rng = MersenneTwister(1)
        h = EventRateTree(16)
        rates = zeros(16)
        ok = true
        for _ in 1:300
            i = rand(rng, 1:16)
            r = rand(rng)                       # occasionally lowers a rate → negative Fenwick δ
            h[i] = r
            rates[i] = r
            ok &= total_rate(h) ≈ sum(rates)
        end
        @test ok
    end

    @testset "prefix sums (fenwick_prefix)" begin
        vals = [0.5, 1.0, 0.0, 2.0, 0.25, 0.75, 3.0, 1.0]
        h = EventRateTree(8)
        for (i, v) in enumerate(vals)
            h[i] = v
        end
        @test all(MonteCarloX.fenwick_prefix(h.tree, i) ≈ sum(@view vals[1:i]) for i in 1:8)
    end

    @testset "next_event: deterministic single non-zero rate" begin
        h = EventRateTree(4)
        h[3] = 2.0
        @test all(next_event(MersenneTwister(k), h) == 3 for k in 1:25)
        h[3] = 0.0
        h[1] = 1.0
        @test all(next_event(MersenneTwister(k), h) == 1 for k in 1:25)
    end

    @testset "next_event: samples ∝ rates" begin
        rng = MersenneTwister(42)
        h = EventRateTree(4)
        rates = [1.0, 2.0, 0.0, 4.0]            # event 3 has zero rate
        for (i, r) in enumerate(rates)
            h[i] = r
        end
        counts = zeros(Int, 4)
        n = 200_000
        for _ in 1:n
            counts[next_event(rng, h)] += 1
        end
        @test isapprox(counts ./ n, rates ./ sum(rates); atol = 0.01)
        @test counts[3] == 0                    # zero-rate event never fires
    end

    @testset "drives a kinetic Monte Carlo step" begin
        alg = Gillespie(MersenneTwister(7))
        h = EventRateTree(3)
        h[1] = 1.0; h[2] = 0.0; h[3] = 3.0
        dt, ev = next(alg, h)
        @test dt > 0 && isfinite(dt)
        @test ev in (1, 3)                      # event 2 has zero rate
        t_new, _ = step!(alg, h)
        @test t_new > 0 && alg.steps == 1
    end
end
