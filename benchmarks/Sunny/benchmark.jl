import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Random, DelimitedFiles
using StatsBase: weights, mean
using MonteCarloX, MCXSpins
import Sunny

datadir = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "data")))
factors_file = joinpath(datadir, "benchmarks.tsv")
bench_file(name) = joinpath(datadir, "bench_$(name).tsv")

function recfactor(comparison, case, ref_ns, mcx_ns)
    new = !isfile(factors_file)
    open(factors_file, "a") do io
        new && println(io, "comparison\tcase\tcpu\tref_ns_per_flip\tmcx_ns_per_flip\tspeedup_mcx")
        println(io, string(comparison, "\t", case, "\t", Sys.CPU_NAME, "\t",
                           ref_ns, "\t", mcx_ns, "\t", ref_ns / mcx_ns))
    end
end

ns_per_flip(run, nflips; reps=3) = 1e9 * minimum(@elapsed(run()) for _ in 1:reps) / nflips
mcx_sweep!(sys, alg, n) = (for _ in 1:n; spin_flip!(sys, alg); end)

function mcx_ising_em(L, β; therm=2_000, sweeps=5_000)
    sys = IsingSystem([L, L]); N = L * L
    init!(sys, :random, rng=MersenneTwister(3))
    alg = MetropolisAlgorithm(Xoshiro(4); β=β)
    mcx_sweep!(sys, alg, therm * N)
    e = m = 0.0
    for _ in 1:sweeps
        mcx_sweep!(sys, alg, N)
        e += energy(sys) / N; m += abs(magnetization(sys)) / N
    end
    return e / sweeps, m / sweeps
end

function mcx_ising_time(L, β; sweeps=5_000)
    sys = IsingSystem([L, L]); N = L * L
    init!(sys, :random, rng=MersenneTwister(2))
    alg = MetropolisAlgorithm(Xoshiro(1); β=β)
    mcx_sweep!(sys, alg, 200 * N)
    return ns_per_flip(() -> mcx_sweep!(sys, alg, sweeps * N), sweeps * N)
end

function sunny_ising_system(L; B=0.0)
    latvecs = Sunny.lattice_vectors(1, 1, 10, 90, 90, 90)
    crystal = Sunny.Crystal(latvecs, [[0, 0, 0]])
    sys = Sunny.System(crystal, [1 => Sunny.Moment(s=1, g=-1)], :dipole; dims=(L, L, 1))
    Sunny.polarize_spins!(sys, (0, 0, 1))
    Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, (1, 0, 0)))
    Sunny.set_field!(sys, (0.0, 0.0, B))
    return sys
end

sunny_magnetization(sys) = abs(sum(S[3] for S in sys.dipoles)) / length(sys.dipoles)

function sunny_ising_em(L, T; therm=2_000, sweeps=5_000)
    sys = sunny_ising_system(L)
    sampler = Sunny.LocalSampler(kT=T, propose=Sunny.propose_flip)
    for _ in 1:therm
        Sunny.step!(sys, sampler)
    end
    e = m = 0.0
    for _ in 1:sweeps
        Sunny.step!(sys, sampler)
        e += Sunny.energy_per_site(sys)
        m += sunny_magnetization(sys)
    end
    return e / sweeps, m / sweeps
end

function sunny_ising_time(L, T; sweeps=5_000)
    N = L * L
    sys = sunny_ising_system(L)
    sampler = Sunny.LocalSampler(kT=T, propose=Sunny.propose_flip)
    for _ in 1:200
        Sunny.step!(sys, sampler)
    end
    return ns_per_flip(() -> (for _ in 1:sweeps; Sunny.step!(sys, sampler); end), sweeps * N)
end

function main()
    L, Ts = 8, [1.4, 1.8, 2.269, 3.0, 3.8]
    logdos = logdos_exact_ising2D(L)
    E = get_centers(logdos)
    e_exact = [mean(E, weights(reweight(logdos, -E ./ T))) / L^2 for T in Ts]
    e_ref = [sunny_ising_em(L, T; therm=2_000, sweeps=5_000)[1] for T in Ts]
    e_mcx = [mcx_ising_em(L, 1 / T; therm=2_000, sweeps=5_000)[1] for T in Ts]

    writedlm(bench_file("sunny_ising"),
             [["T" "e_ref" "e_mcx" "e_exact"]; hcat(Ts, e_ref, e_mcx, e_exact)], '\t')

    for T in (1.4, 2.269, 3.8)
        sweeps, Lb = 5_000, 64
        recfactor("Sunny.jl (2D Ising 64×64)", "T=$T",
                  sunny_ising_time(Lb, T; sweeps=sweeps),
                  mcx_ising_time(Lb, 1 / T; sweeps=sweeps))
    end
end

main()
