using Random
using Statistics
using Printf
using MonteCarloX
using SpinSystems

# Optional CLI args: julia --project=. benchmark.jl [steps_per_sample] [samples]
const STEPS_PER_SAMPLE = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 500_000
const SAMPLES = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
const WARMUP_STEPS = max(2_000, STEPS_PER_SAMPLE ÷ 10)

mutable struct TableMetropolis{R<:AbstractRNG} <: AbstractMetropolis
    rng::R
    p4::Float64
    p8::Float64
    steps::Int
    accepted::Int
end

TableMetropolis(rng::AbstractRNG; β::Real) =
    TableMetropolis(rng, exp(-4β), exp(-8β), 0, 0)

@inline function MonteCarloX.accept!(alg::TableMetropolis, dE::Int)
    alg.steps += 1
    dE <= 0 && (alg.accepted += 1; return true)
    p = dE == 4 ? alg.p4 : alg.p8
    accepted = rand(alg.rng) < p
    alg.accepted += accepted
    return accepted
end

@inline function run_updates!(sys, alg, nsteps::Int)
    for _ in 1:nsteps
        spin_flip!(sys, alg)
    end
    return nothing
end

function benchmark_case(name::AbstractString, sys, alg)
    # JIT warmup and cache warmup
    run_updates!(sys, alg, WARMUP_STEPS)

    times_ns = Vector{Float64}(undef, SAMPLES)
    for s in 1:SAMPLES
        t0 = time_ns()
        run_updates!(sys, alg, STEPS_PER_SAMPLE)
        t1 = time_ns()
        times_ns[s] = (t1 - t0) / STEPS_PER_SAMPLE
    end

    med_ns = median(times_ns)
    mean_ns = mean(times_ns)
    upd_s = 1.0e9 / med_ns
    if applicable(acceptance_rate, alg)
        acc = acceptance_rate(alg)
        @printf("%-34s median: %9.1f ns  mean: %9.1f ns  throughput: %10.2e updates/s  acceptance: %.3f\n",
                name, med_ns, mean_ns, upd_s, acc)
    else
        @printf("%-34s median: %9.1f ns  mean: %9.1f ns  throughput: %10.2e updates/s  acceptance: n/a\n",
                name, med_ns, mean_ns, upd_s)
    end
    return nothing
end

function main()
    println("SpinSystems benchmark")
    println("Configuration: steps/sample=$(STEPS_PER_SAMPLE), samples=$(SAMPLES), warmup=$(WARMUP_STEPS)")
    println()

    rng_ising_table = Xoshiro(2026)
    sys_ising_table = Ising([64, 64])
    init!(sys_ising_table, :random; rng=rng_ising_table)
    alg_ising_table = TableMetropolis(Xoshiro(1); β=0.3)
    benchmark_case("Ising(64x64) Lattice/Int + TableMetropolis", sys_ising_table, alg_ising_table)

    rng_ising_fast = Xoshiro(2026)
    sys_ising_fast = Ising([64, 64])
    init!(sys_ising_fast, :random; rng=rng_ising_fast)
    alg_ising_fast = Metropolis(Xoshiro(1); β=0.3)
    benchmark_case("Ising(64x64) Lattice/Int + Metropolis", sys_ising_fast, alg_ising_fast)

    rng_ising = Xoshiro(2026)
    sys_ising = Ising([64, 64]; J=1.0, h=0.0)
    init!(sys_ising, :random; rng=rng_ising)
    alg_ising = Metropolis(Xoshiro(1); β=0.3)
    benchmark_case("Ising(64x64) + Metropolis", sys_ising, alg_ising)

    rng_bc = Xoshiro(2028)
    sys_bc = BlumeCapel([48, 48]; J=1.0, D=0.5, h=0.1)
    init!(sys_bc, :random; rng=rng_bc)
    alg_bc = Metropolis(Xoshiro(3); β=0.40)
    benchmark_case("BlumeCapel(48x48) + Metropolis", sys_bc, alg_bc)

    return nothing
end

main()
