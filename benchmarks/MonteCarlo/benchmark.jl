import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Random, DelimitedFiles
using StatsBase: weights, mean
using MonteCarloX, MCXSpins
import MonteCarlo

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
mcx_sweep_seq!(sys, alg, nsweeps) = (for _ in 1:nsweeps, i in eachindex(sys.spins); spin_flip!(sys, alg, i); end)

function mcx_ising_em(L, β; therm=10^4, sweeps=10^3)
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

function mcx_ising_time_seq(L, β; sweeps=5_000)
    sys = IsingSystem([L, L]); N = L * L
    init!(sys, :random, rng=MersenneTwister(2))
    alg = MetropolisAlgorithm(Xoshiro(1); β=β)
    mcx_sweep_seq!(sys, alg, 200)
    return ns_per_flip(() -> mcx_sweep_seq!(sys, alg, sweeps), sweeps * N)
end

function mc_em(L, β; therm=10^4, sweeps=10^3)
    mc = MonteCarlo.MC(MonteCarlo.IsingModel(dims=2, L=L), beta=β)
    MonteCarlo.run!(mc; sweeps=sweeps, thermalization=therm, verbose=false)
    meas = MonteCarlo.measurements(mc)
    return mean(meas[:Energy].e), mean(meas[:Magn].m)
end

function mc_kernel_time(L, T; sweeps=5_000)
    mc = MonteCarlo.MC(MonteCarlo.IsingModel(dims=2, L=L), beta=1 / T)
    for _ in 1:200
        MonteCarlo.sweep(mc)
    end
    return ns_per_flip(() -> (for _ in 1:sweeps; MonteCarlo.sweep(mc); end), sweeps * L * L)
end

function main()
    L, Ts = 8, [1.4, 1.8, 2.269, 3.0, 3.8]
    logdos = logdos_exact_ising2D(L)
    E = get_centers(logdos)
    e_exact = [mean(E, weights(reweight(logdos, -E ./ T))) / L^2 for T in Ts]
    e_ref = [mc_em(L, 1 / T)[1] for T in Ts]
    e_mcx = [mcx_ising_em(L, 1 / T)[1] for T in Ts]

    writedlm(bench_file("montecarlo"),
             [["T" "e_ref" "e_mcx" "e_exact"]; hcat(Ts, e_ref, e_mcx, e_exact)], '\t')

    for T in (1.4, 2.269, 3.8)
        sweeps, Lb = 5_000, 64
        recfactor("MonteCarlo.jl (2D Ising 64×64)", "T=$T",
                  mc_kernel_time(Lb, T; sweeps=sweeps),
                  mcx_ising_time_seq(Lb, 1 / T; sweeps=sweeps))
    end
end

main()
