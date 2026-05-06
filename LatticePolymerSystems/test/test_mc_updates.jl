using LatticePolymerSystems
using MonteCarloX: Metropolis, acceptance_rate
using Test
using Random

"""
    verify_invariants(sys)

Check all system invariants: state↔polymers consistency, self-avoidance,
connectivity, energy consistency. Throws on failure.
"""
function verify_invariants(sys)
    N = num_polymers(sys)
    # state matches polymer coordinates
    expected_state = zeros(Int, length(sys.state))
    for n in 1:N
        for m in 1:polymer_length(sys, n)
            site = coords_to_site(sys.polymers[n][m], sys.dims)
            @assert expected_state[site] == 0 "Overlap at site $site: polymers $(expected_state[site]) and $n"
            expected_state[site] = n
        end
    end
    @assert expected_state == sys.state "state[] out of sync with polymer coordinates"
    # connectivity
    for n in 1:N
        for m in 1:polymer_length(sys, n)-1
            s1 = coords_to_site(sys.polymers[n][m], sys.dims)
            s2 = coords_to_site(sys.polymers[n][m+1], sys.dims)
            @assert s2 ∈ sys.neighbors[s1] "Polymer $n: monomers $m and $(m+1) not connected"
        end
    end
    # energy
    @assert energy(sys) ≈ energy(sys; full=true) "Cached energy $(energy(sys)) != full $(energy(sys; full=true))"
end

@testset "MC Updates" begin
    # ── Per-step invariant checks ────────────────────────────────────────────
    @testset "Slither: per-step invariants" begin
        sys = LatticePolymer(; dims=[8, 8], num_poly=4, length_poly=8, J_intra=1.0, J_inter=1.0)
        init!(sys, :random; rng=Xoshiro(42))
        alg = Metropolis(Xoshiro(1); β=0.5)
        for _ in 1:200
            slither_move!(sys, alg)
            verify_invariants(sys)
        end
        @test acceptance_rate(alg) > 0.0
    end

    @testset "Translate: per-step invariants" begin
        sys = LatticePolymer(; dims=[8, 8], num_poly=4, length_poly=8, J_intra=1.0, J_inter=1.0)
        init!(sys, :random; rng=Xoshiro(42))
        alg = Metropolis(Xoshiro(2); β=0.5)
        for _ in 1:200
            translate!(sys, alg)
            verify_invariants(sys)
        end
        @test acceptance_rate(alg) > 0.0
    end

    @testset "Pivot: per-step invariants" begin
        sys = LatticePolymer(; dims=[8, 8], num_poly=4, length_poly=8, J_intra=1.0, J_inter=1.0)
        init!(sys, :random; rng=Xoshiro(42))
        alg = Metropolis(Xoshiro(3); β=0.5)
        for _ in 1:200
            pivot_move!(sys, alg)
            verify_invariants(sys)
        end
        @test acceptance_rate(alg) > 0.0
    end

    @testset "Double bridge: per-step invariants" begin
        # Dense system to increase bridge opportunities
        sys = LatticePolymer(; dims=[6, 6], num_poly=6, length_poly=4, J_intra=1.0, J_inter=1.0)
        init!(sys, :random; rng=Xoshiro(42))
        alg = Metropolis(Xoshiro(4); β=0.5)
        # Mix in slither to rearrange and create bridge opportunities
        alg_mix = Metropolis(Xoshiro(5); β=0.5)
        for _ in 1:500
            slither_move!(sys, alg_mix)
            double_bridge_move!(sys, alg)
            verify_invariants(sys)
        end
        @test acceptance_rate(alg) > 0.0
    end

    # ── Mixed moves stress test ──────────────────────────────────────────────
    @testset "Mixed moves: long run" begin
        sys = LatticePolymer(; dims=[10, 10], num_poly=4, length_poly=10, J_intra=0.5, J_inter=1.0)
        init!(sys, :random; rng=Xoshiro(99))
        alg = Metropolis(Xoshiro(0); β=0.3)
        moves = [slither_move!, translate!, pivot_move!, double_bridge_move!]
        for _ in 1:2000
            moves[rand(alg.rng, 1:4)](sys, alg)
        end
        verify_invariants(sys)
        @test 0.0 < acceptance_rate(alg) < 1.0
    end

end
