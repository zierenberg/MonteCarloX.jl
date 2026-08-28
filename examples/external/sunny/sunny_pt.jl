# Parallel Tempering: Sunny vs MCX Bridge vs MCX Native
#
# This example demonstrates parallel tempering (replica exchange) on the 2D Ising model
# in three implementations: Sunny native, MCX bridge on Sunny, and MCX native on MCXSpins.
# See examples/mcmc/pt_Ising2D.jl for a more detailed MCX implementation with reweighting.

using Random, Statistics, Plots, Printf
using Sunny, MonteCarloX, MCXSpins

const SEED = 42
const L = 8
const n_replicas = 4
const n_therm = 1_000
const n_prod = 10_000
const Tmin, Tmax = 1.5, 3.0
const measure_interval = 50

# ============================================================================
# SECTION 1: SUNNY PARALLEL TEMPERING
# ============================================================================
# Sunny uses ParallelTempering with replica exchange and step_ensemble!

function run_sunny_pt(n_therm, n_prod, measure_interval)
    # Setup: create system and temperature schedule
    latvecs = Sunny.lattice_vectors(1, 1, 10, 90, 90, 90)
    sys = Sunny.System(Sunny.Crystal(latvecs, [[0,0,0]]), 
                       [1 => Sunny.Moment(s=1, g=-1)], :dipole; dims=(L, L, 1))
    Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, (1, 0, 0)))
    Sunny.polarize_spins!(sys, (0, 0, 1))
    
    kT_sched = collect(range(Tmin, Tmax, length=n_replicas))
    sampler = Sunny.LocalSampler(; kT=0, propose=Sunny.propose_flip)
    pt = Sunny.ParallelTempering(sys, sampler, kT_sched)
    
    # Thermalization with frequent exchanges
    Sunny.step_ensemble!(pt, n_therm, 1)  # n_therm sweeps, exchange every sweep
    
    # Production: parallel sweeps + frequent exchanges
    energies_per_replica = [Float64[] for _ in 1:n_replicas]
    exch_interval = 5  # attempt exchange every N sweeps
    n_meas = n_prod ÷ exch_interval
    for _ in 1:n_meas
        Sunny.step_ensemble!(pt, exch_interval, 1)  # sweeps + frequent exchange
        for r in 1:n_replicas
            push!(energies_per_replica[r], Sunny.energy_per_site(pt.systems[r]))
        end
    end
    
    energies_per_replica
end

# ============================================================================
# SECTION 2: MCX BRIDGE (Sunny systems + MCX ParallelTempering)
# ============================================================================
# Same Sunny systems, but acceptance decisions made by MCX's ParallelTempering.
# Each replica's sweep uses Sunny for proposal/energy, MCX algorithm for accept!.
# See sunny_ising.jl for the single-temperature bridge pattern.

function sweep_bridge!(sys, alg, n)
    N = length(collect(Sunny.eachsite(sys)))
    for _ in 1:n, _ in 1:N
        site = rand(alg.rng, Sunny.eachsite(sys))
        prop = Sunny.propose_flip(sys, site)
        accept!(alg, Sunny.local_energy_change(sys, site, prop)) && Sunny.setspin!(sys, prop, site)
    end
end

function run_bridge_pt(n_therm, n_prod, measure_interval)
    betas = collect(range(inv(Tmax), inv(Tmin), length=n_replicas))
    latvecs = Sunny.lattice_vectors(1, 1, 10, 90, 90, 90)
    systems = [begin
        s = Sunny.System(Sunny.Crystal(latvecs, [[0,0,0]]),
                         [1 => Sunny.Moment(s=1, g=-1)], :dipole; dims=(L, L, 1))
        Sunny.set_exchange!(s, -1.0, Sunny.Bond(1, 1, (1, 0, 0)))
        Sunny.polarize_spins!(s, (0, 0, 1))
        s
    end for _ in 1:n_replicas]
    pt = ParallelTempering(betas; seed=SEED, rng=Xoshiro)

    with_parallel(pt) do r, alg
        copy!(systems[r].rng, Xoshiro(SEED + r))
        sweep_bridge!(systems[r], alg, n_therm)
    end

    energies_per_replica = [Float64[] for _ in 1:n_replicas]
    energies = zeros(Float64, n_replicas)
    exch_interval = 5
    n_meas = n_prod ÷ exch_interval
    for _ in 1:n_meas
        for _ in 1:exch_interval
            with_parallel(pt) do r, alg
                sweep_bridge!(systems[r], alg, 1)
                energies[r] = Sunny.energy_per_site(systems[r])
            end
            MonteCarloX.update!(pt, energies)
        end
        for r in 1:n_replicas
            push!(energies_per_replica[r], energies[r])
        end
    end

    energies_per_replica
end

# ============================================================================
# SECTION 3: MCX NATIVE (MCXSpins IsingSystem + MCX ParallelTempering)
# ============================================================================
# MCX uses ParallelTempering (backed by ThreadsBackend) with manual sweep loops
# and update! for replica exchange proposals. with_parallel uses @threads when
# Julia is started with -t N; serial otherwise.
# See examples/mcmc/pt_Ising2D.jl for the full pedagogical version.

sweep!(sys, alg, n) = (for _ in 1:n, _ in 1:length(sys.spins); spin_flip!(sys, alg); end)

function run_mcx_pt(n_therm, n_prod, measure_interval)    # Setup
    betas = collect(range(inv(Tmax), inv(Tmin), length=n_replicas))
    systems = [IsingSystem([L, L]) for _ in 1:n_replicas]
    pt = ParallelTempering(betas; seed=SEED, rng=Xoshiro)

    with_parallel(pt) do r, alg # @threads when julia -t N, serial otherwise
        init!(systems[r], :random; rng=alg.rng)
        sweep!(systems[r], alg, n_therm)
    end

    # Production: sweeps + frequent exchanges
    energies_per_replica = [Float64[] for _ in 1:n_replicas]
    energies = zeros(Float64, n_replicas)
    exch_interval = 5  # attempt exchange every N sweeps
    n_meas = n_prod ÷ exch_interval
    for _ in 1:n_meas
        for _ in 1:exch_interval
            with_parallel(pt) do r, alg  # @threads when julia -t N, serial otherwise
                sweep!(systems[r], alg, 1)
                energies[r] = MCXSpins.energy(systems[r])
            end
            MonteCarloX.update!(pt, energies)  # attempt exchange between neighbor replicas
        end
        for r in 1:n_replicas
            push!(energies_per_replica[r], energies[r])
        end
    end

    energies_per_replica
end

# ============================================================================
# MAIN: RUN ALL THREE IMPLEMENTATIONS WITH TIMING
# ============================================================================

println("Parallel Tempering: 2D Ising, L=$L, n_replicas=$n_replicas")
println("="^70)

# Warmup (compile all paths)
run_sunny_pt(2, 2, 1)
run_bridge_pt(2, 2, 1)
run_mcx_pt(2, 2, 1)

# Timing measurement
t_sunny  = @elapsed run_sunny_pt(n_therm, n_prod, measure_interval)
t_bridge = @elapsed run_bridge_pt(n_therm, n_prod, measure_interval)
t_mcx    = @elapsed run_mcx_pt(n_therm, n_prod, measure_interval)

println("Timing Comparison (n_therm=$n_therm, n_prod=$n_prod):")
println("-"^70)
println(@sprintf "  Sunny native:        %8.4f s" t_sunny)
println(@sprintf "  MCX bridge:          %8.4f s" t_bridge)
println(@sprintf "  MCX native:          %8.4f s" t_mcx)
println(@sprintf "  Speedup native/Sunny:  %.2f x" t_sunny / t_mcx)
println()
println("Notes:")
println("  - Sunny:  LocalSampler + step_ensemble! + replica exchange")
println("  - Bridge: MCX ParallelTempering driving Sunny systems (accept! + setspin!)")
println("  - MCX:    ParallelTempering + spin_flip! + with_parallel (threadable)")
println("  - For detailed PT with reweighting, see examples/mcmc/pt_Ising2D.jl")
