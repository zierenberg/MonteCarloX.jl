using Random
using Statistics
using Printf
using MonteCarloX
using LatticePolymerSystems

# Optional CLI args: julia --project=. benchmark.jl [steps_per_sample] [samples]
const STEPS_PER_SAMPLE = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 6_000
const SAMPLES = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8
const WARMUP_STEPS = max(1_000, STEPS_PER_SAMPLE ÷ 5)

@inline function run_updates!(sys, alg, move!::Function, nsteps::Int)
    for _ in 1:nsteps
        move!(sys, alg)
    end
    return nothing
end

function benchmark_case(name::AbstractString, setup::Function, move!::Function)
    sys, alg = setup()

    # JIT warmup and cache warmup
    run_updates!(sys, alg, move!, WARMUP_STEPS)

    times_ns = Vector{Float64}(undef, SAMPLES)
    for s in 1:SAMPLES
        t0 = time_ns()
        run_updates!(sys, alg, move!, STEPS_PER_SAMPLE)
        t1 = time_ns()
        times_ns[s] = (t1 - t0) / STEPS_PER_SAMPLE
    end

    med_ns = median(times_ns)
    mean_ns = mean(times_ns)
    upd_s = 1.0e9 / med_ns
    acc = acceptance_rate(alg)

    @printf("%-44s median: %10.1f ns  mean: %10.1f ns  throughput: %10.2e updates/s  acceptance: %.3f\n",
            name, med_ns, mean_ns, upd_s, acc)
    return nothing
end

println("LatticePolymerSystems benchmark")
println("Configuration: steps/sample=$(STEPS_PER_SAMPLE), samples=$(SAMPLES), warmup=$(WARMUP_STEPS)")
println()

benchmark_case("LatticePolymer(24x24, 8x20) slither_move!", () -> begin
    rng = Xoshiro(2040)
    sys = LatticePolymer(; dims=[24, 24], num_poly=8, length_poly=20,
        J_intra=0.5, J_inter=1.0)
    init!(sys, :random; rng=rng)
    alg = Metropolis(Xoshiro(21); β=0.5)
    return sys, alg
end, slither_move!)

benchmark_case("LatticePolymer(24x24, 8x20) translate!", () -> begin
    rng = Xoshiro(2041)
    sys = LatticePolymer(; dims=[24, 24], num_poly=8, length_poly=20,
        J_intra=0.5, J_inter=1.0)
    init!(sys, :random; rng=rng)
    alg = Metropolis(Xoshiro(22); β=0.5)
    return sys, alg
end, translate!)

benchmark_case("LatticePolymer(24x24, 8x20) pivot_move!", () -> begin
    rng = Xoshiro(2042)
    sys = LatticePolymer(; dims=[24, 24], num_poly=8, length_poly=20,
        J_intra=0.5, J_inter=1.0)
    init!(sys, :random; rng=rng)
    alg = Metropolis(Xoshiro(23); β=0.5)
    return sys, alg
end, pivot_move!)

benchmark_case("LatticePolymer(20x20, 12x12) double_bridge_move!", () -> begin
    rng = Xoshiro(2043)
    sys = LatticePolymer(; dims=[20, 20], num_poly=12, length_poly=12,
        J_intra=0.5, J_inter=1.0)
    init!(sys, :random; rng=rng)
    alg = Metropolis(Xoshiro(24); β=0.5)
    return sys, alg
end, double_bridge_move!)
