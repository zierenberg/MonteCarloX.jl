using Random
using Statistics
using Printf
using MonteCarloX
using SoftMatterSystems

# Optional CLI args: julia --project=. benchmark.jl [steps_per_sample] [samples]
const STEPS_PER_SAMPLE = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5_000
const SAMPLES = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8
const WARMUP_STEPS = max(1_000, STEPS_PER_SAMPLE ÷ 5)

@inline function run_updates!(sys, alg, move!::Function, nsteps::Int; chain::Bool=false)
    if chain
        for _ in 1:nsteps
            move!(sys, alg; chain=true)
        end
    else
        for _ in 1:nsteps
            move!(sys, alg)
        end
    end
    return nothing
end

function benchmark_case(name::AbstractString, setup::Function, move!::Function; chain::Bool=false)
    sys, alg = setup()

    # JIT warmup and cache warmup
    run_updates!(sys, alg, move!, WARMUP_STEPS; chain=chain)

    times_ns = Vector{Float64}(undef, SAMPLES)
    for s in 1:SAMPLES
        t0 = time_ns()
        run_updates!(sys, alg, move!, STEPS_PER_SAMPLE; chain=chain)
        t1 = time_ns()
        times_ns[s] = (t1 - t0) / STEPS_PER_SAMPLE
    end

    med_ns = median(times_ns)
    mean_ns = mean(times_ns)
    upd_s = 1.0e9 / med_ns
    acc = acceptance_rate(alg)

    @printf("%-46s median: %10.1f ns  mean: %10.1f ns  throughput: %10.2e updates/s  acceptance: %.3f\n",
            name, med_ns, mean_ns, upd_s, acc)
    return nothing
end

println("SoftMatterSystems benchmark")
println("Configuration: steps/sample=$(STEPS_PER_SAMPLE), samples=$(SAMPLES), warmup=$(WARMUP_STEPS)")
println()

benchmark_case("ParticleGas(D=3,N=256,LJ) translate!", () -> begin
    rng = Xoshiro(2030)
    lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
    sys = ParticleGas(; D=3, N=256, L=32.0, pair_potential=lj, delta=0.20)
    init!(sys, :random; rng=rng)
    alg = Metropolis(Xoshiro(11); β=1.0)
    return sys, alg
end, translate!)

benchmark_case("BeadSpring(D=3,6x24,LJ+FENE) monomer translate!", () -> begin
    rng = Xoshiro(2031)
    lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
    fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
    sys = BeadSpringPolymer(; D=3, num_poly=6, length_poly=24, L=48.0,
        pair_potential=lj, bond_potential=fene, delta=0.08)
    init!(sys, :random_walk; rng=rng)
    alg = Metropolis(Xoshiro(12); β=1.0)
    return sys, alg
end, translate!)

benchmark_case("BeadSpring(D=3,6x24,LJ+FENE) chain translate!", () -> begin
    rng = Xoshiro(2032)
    lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
    fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
    sys = BeadSpringPolymer(; D=3, num_poly=6, length_poly=24, L=28.0,
        pair_potential=lj, bond_potential=fene, delta=0.05)
    init!(sys, :random_walk; rng=rng)
    alg = Metropolis(Xoshiro(13); β=2.0)
    return sys, alg
end, translate!; chain=true)

benchmark_case("BeadSpring(D=3,4x20,LJ+FENE+bending) monomer translate!", () -> begin
    rng = Xoshiro(2033)
    lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
    fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
    bend = CosineBendingPotential(3.0)
    sys = BeadSpringPolymer(; D=3, num_poly=4, length_poly=20, L=42.0,
        pair_potential=lj, bond_potential=fene, bending_potential=bend, delta=0.08)
    init!(sys, :random_walk; rng=rng)
    alg = Metropolis(Xoshiro(14); β=1.0)
    return sys, alg
end, translate!)
