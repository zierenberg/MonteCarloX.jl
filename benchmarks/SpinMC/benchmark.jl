import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Random, DelimitedFiles, Statistics
using MonteCarloX, MCXSpins
using SpinMC

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

quiet(f) = redirect_stdout(() -> redirect_stderr(f, devnull), devnull)
ns_per_flip(run, nflips; reps=3) = 1e9 * minimum(@elapsed(run()) for _ in 1:reps) / nflips
mcx_sweep!(sys, alg, n) = (for _ in 1:n; spin_flip!(sys, alg); end)

function spinmc_lattice(L)
    uc = UnitCell((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    b = addBasisSite!(uc, (0.0, 0.0, 0.0))
    M = [-1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 -1.0]
    for dir in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        addInteraction!(uc, b, b, M, dir)
    end
    return Lattice(uc, (L, L, L))
end

function spinmc_m(T; L=8, therm=2_000, sweeps=5_000)
    mc = SpinMC.MonteCarlo(spinmc_lattice(L), 1 / T, therm, sweeps; reportInterval=10^9)
    quiet(() -> SpinMC.run!(mc))
    return mean(mc.observables.magnetization), SpinMC.std_error(mc.observables.magnetization)
end

function mcx_heisenberg_m(T; L=8, therm=2_000, sweeps=5_000)
    sys = HeisenbergSystem([L, L, L]); N = L^3
    init!(sys, :random, rng=MersenneTwister(3))
    alg = MetropolisAlgorithm(Xoshiro(4); β=1 / T)
    mcx_sweep!(sys, alg, therm * N)
    ms = [(mcx_sweep!(sys, alg, N); sqrt(sum(abs2, magnetization(sys))) / N) for _ in 1:sweeps]
    return mean(ms), std(ms) / sqrt(length(ms) / 100)
end

function main()
    Ts = [10.0 * (0.1 / 10.0)^((i - 1) / 7) for i in 1:8]
    ref = spinmc_m.(Ts)
    mcx = mcx_heisenberg_m.(Ts)

    writedlm(bench_file("spinmc"),
             [["T" "m_ref" "dm_ref" "m_mcx" "dm_mcx"];
              hcat(Ts, first.(ref), last.(ref), first.(mcx), last.(mcx))], '\t')

    for T in (0.5, 1.443, 10.0)
        sweeps, N = 5_000, 512
        lat = spinmc_lattice(8)
        t_ref = ns_per_flip(sweeps * N) do
            quiet(() -> SpinMC.run!(SpinMC.MonteCarlo(lat, 1 / T, sweeps, 0; reportInterval=10^9)))
        end
        sys = HeisenbergSystem([8, 8, 8])
        init!(sys, :random, rng=MersenneTwister(2))
        alg = MetropolisAlgorithm(Xoshiro(1); β=1 / T)
        mcx_sweep!(sys, alg, 200 * N)
        t_mcx = ns_per_flip(() -> mcx_sweep!(sys, alg, sweeps * N), sweeps * N)
        recfactor("SpinMC.jl (Heisenberg 8×8×8)", "T=$T", t_ref, t_mcx)
    end
end

main()
