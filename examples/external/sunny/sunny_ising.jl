
import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

using Random, Statistics, Printf
using StatsBase: weights, mean
using Plots
using Sunny
using MCXSpins: logdos_exact_ising2D, IsingSystem, init!, energy, spin_flip!
using MonteCarloX: reweight, get_centers, MetropolisAlgorithm, accept!, acceptance_rate, reset!

L, therm, prod = 8, 5000, 200_000
Tc = 2 / log(1 + sqrt(2.0))
Ts = [1.8, Tc, 3.0]

const SEED = 42

logdos = logdos_exact_ising2D(L)
E = get_centers(logdos)
e_exact = [mean(E, weights(reweight(logdos, -E ./ T))) / L^2 for T in Ts]

function sweep_sunny!(sys, sampler)
    Sunny.step!(sys, sampler)
end

function sweep_mcx_sunny!(sys, alg, N)
    for _ in 1:N
        site = rand(sys.rng, Sunny.eachsite(sys))
        prop = Sunny.propose_flip(sys, site)
        accept!(alg, Sunny.local_energy_change(sys, site, prop)) && Sunny.setspin!(sys, prop, site)
    end
end

function sweep_mcx_native!(sys, alg, N)
    for _ in 1:N; spin_flip!(sys, alg); end
end

function run_sunny(Ts, therm, prod)
    latvecs = Sunny.lattice_vectors(1, 1, 10, 90, 90, 90)
    sys = Sunny.System(Sunny.Crystal(latvecs, [[0, 0, 0]]), [1 => Sunny.Moment(s=1, g=-1)], :dipole; dims=(L, L, 1))
    Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, (1, 0, 0)))
    es, ts = [], []
    for (i, T) in enumerate(Ts)
        Sunny.polarize_spins!(sys, (0, 0, 1))
        rng_init = Xoshiro(SEED + i)
        for site in Sunny.eachsite(sys)
            rand(rng_init, Bool) && Sunny.setspin!(sys, Sunny.propose_flip(sys, site), site)
        end
        copy!(sys.rng, Xoshiro(SEED + 100 + i))
        sampler = Sunny.LocalSampler(kT=T, nsweeps=1.0, propose=Sunny.propose_flip)
        for _ in 1:therm; sweep_sunny!(sys, sampler); end
        t = @elapsed begin
            e = 0.0
            for _ in 1:prod
                sweep_sunny!(sys, sampler)
                e += Sunny.energy_per_site(sys)
            end
            e /= prod
        end
        push!(es, e); push!(ts, t)
    end
    es, ts
end

function run_mcx_sunny(Ts, therm, prod)
    latvecs = Sunny.lattice_vectors(1, 1, 10, 90, 90, 90)
    sys = Sunny.System(Sunny.Crystal(latvecs, [[0, 0, 0]]), [1 => Sunny.Moment(s=1, g=-1)], :dipole; dims=(L, L, 1))
    Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, (1, 0, 0)))
    N = L^2
    es, ts = [], []
    for (i, T) in enumerate(Ts)
        Sunny.polarize_spins!(sys, (0, 0, 1))
        rng_init = Xoshiro(SEED + i)
        for site in Sunny.eachsite(sys)
            rand(rng_init, Bool) && Sunny.setspin!(sys, Sunny.propose_flip(sys, site), site)
        end
        copy!(sys.rng, Xoshiro(SEED + 100 + i))
        alg = MetropolisAlgorithm(Xoshiro(SEED + 200 + i); β=1/T)
        for _ in 1:therm; sweep_mcx_sunny!(sys, alg, N); end
        reset!(alg)
        t = @elapsed begin
            e = 0.0
            for _ in 1:prod
                sweep_mcx_sunny!(sys, alg, N)
                e += Sunny.energy_per_site(sys)
            end
            e /= prod
        end
        push!(es, e); push!(ts, t)
    end
    es, ts
end

function run_mcx_native(Ts, therm, prod)
    sys = IsingSystem([L, L])
    N = L^2
    es, ts = [], []
    for (i, T) in enumerate(Ts)
        init!(sys, :random, rng=MersenneTwister(SEED + i))
        alg = MetropolisAlgorithm(Xoshiro(SEED + 100 + i); β=1/T)
        for _ in 1:therm; sweep_mcx_native!(sys, alg, N); end
        reset!(alg)
        t = @elapsed begin
            e = 0.0
            for _ in 1:prod
                sweep_mcx_native!(sys, alg, N)
                e += energy(sys) / N
            end
            e /= prod
        end
        push!(es, e); push!(ts, t)
    end
    es, ts
end

# Warmup: call each run_* function fully to compile before real timing
run_sunny([Tc], 2, 2)
run_mcx_sunny([Tc], 2, 2)
run_mcx_native([Tc], 2, 2)

e_s, t_s = run_sunny(Ts, therm, prod)
e_bs, t_bs = run_mcx_sunny(Ts, therm, prod)
e_n, t_n = run_mcx_native(Ts, therm, prod)

println("T       e_exact      e_sunny      e_mcx_sunny  e_mcx_native  |Δ_sunny|     |Δ_bridge|    |Δ_native|    t_sunny(s)  t_mcx_sunny  t_native(s)")
for i in 1:3
    @printf "%-7.3f %12.6f %12.6f %12.6f %12.6f  %12.6f %12.6f %12.6f  %10.4f %11.4f %11.4f\n" Ts[i] e_exact[i] e_s[i] e_bs[i] e_n[i] abs(e_s[i]-e_exact[i]) abs(e_bs[i]-e_exact[i]) abs(e_n[i]-e_exact[i]) t_s[i] t_bs[i] t_n[i]
end

p = plot(Ts, e_exact; lw=2, ls=:dot, color=:black, label="exact", xlabel="T", ylabel="e per site")
plot!(p, Ts, e_s; marker=:o, label="Sunny")
plot!(p, Ts, e_bs; marker=:d, label="MCX on Sunny")
plot!(p, Ts, e_n; marker=:s, label="MCX native")
savefig(p, "ising_comparison.pdf")
