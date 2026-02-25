"""
ising_muca_mpi.jl

Multicanonical (MUCA) sampling for 2D Ising model using MPI for parallel replica exchange.
Run with: mpiexec -n 4 julia --project ising_muca_mpi.jl

Each MPI rank runs one independent replica and exchanges histograms via Allreduce.
"""

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Add local dev packages to load path
# push!(LOAD_PATH, joinpath(dirname(@__DIR__), "SpinSystems", "src"))

using Random
using StatsBase
using MonteCarloX
using SpinSystems
using MPI

# Initialize MPI with thread support
if !MPI.Initialized()
    required = MPI.THREAD_FUNNELED
    provided = MPI.Init_thread(required)
    if provided < required
        error("MPI thread support insufficient: required THREAD_FUNNELED, got level $(provided).")
    end
end

# Exact Beale logDOS for RMSE reference
log_dos_beale_8x8 = [
    (-128, 0.6931471805599453), (-120, 4.852030263919617), (-116, 5.545177444479562),
    (-112, 8.449342524508063), (-108, 9.793672686528922), (-104, 11.887298863200714),
    (-100, 13.477180596840947), (-96, 15.268195474147658), (-92, 16.912371686315282),
    (-88, 18.59085846191256), (-84, 20.230089202801466), (-80, 21.870810400320693),
    (-76, 23.498562234123614), (-72, 25.114602234581373), (-68, 26.70699035290573),
    (-64, 28.266152815389898), (-60, 29.780704423363996), (-56, 31.241053997806176),
    (-52, 32.63856452513369), (-48, 33.96613536105969), (-44, 35.217576663643314),
    (-40, 36.3873411250109), (-36, 37.47007844691906), (-32, 38.46041522581422),
    (-28, 39.35282710786369), (-24, 40.141667825183845), (-20, 40.82130289691285),
    (-16, 41.38631975325592), (-12, 41.831753810069756), (-8, 42.153328313883975),
    (-4, 42.34770636939425), (0, 42.41274640460084), (4, 42.34770636939425),
    (8, 42.153328313883975), (12, 41.831753810069756), (16, 41.38631975325592),
    (20, 40.82130289691285), (24, 40.141667825183845), (28, 39.35282710786369),
    (32, 38.46041522581422), (36, 37.47007844691906), (40, 36.3873411250109),
    (44, 35.217576663643314), (48, 33.96613536105969), (52, 32.63856452513369),
    (56, 31.241053997806176), (60, 29.780704423363996), (64, 28.266152815389898),
    (68, 26.70699035290573), (72, 25.114602234581373), (76, 23.498562234123614),
    (80, 21.870810400320693), (84, 20.230089202801466), (88, 18.59085846191256),
    (92, 16.912371686315282), (96, 15.268195474147658), (100, 13.477180596840947),
    (104, 11.887298863200714), (108, 9.793672686528922), (112, 8.449342524508063),
    (116, 5.545177444479562), (120, 4.852030263919617), (128, 0.6931471805599453),
]
exact_logdos = Dict(log_dos_beale_8x8)
domain = sort([e for (e, _) in log_dos_beale_8x8])
# TODO: make this work with arbitrary domains and not just the discrete steps
exact_logdos = BinnedLogWeight(domain[1]:4:domain[end], 0.0)
# fill the exact_logdos with the values from log_dos_beale_8x8
for (e, logdos) in log_dos_beale_8x8
    if e in domain
        exact_logdos[e] = logdos
    end
end
# normalize
exact_logdos.weights .-= exact_logdos[0]
# mask to only those values that are in the raw_domain
full_domain = collect(domain[1]:4:domain[end])
mask_rmse = full_domain .∈ Ref(domain)
#mask to only compute RMSE on the domain points that are in the exact_logdos
exact_masked = exact_logdos.weights[mask_rmse]

function rmse_exact(lw)
    MonteCarloX._assert_same_domain(lw, exact_logdos)
    est = -deepcopy(lw.weights)
    est .+= lw[0]  # shift to match exact at E=0
    # implement the mask to only compute RMSE on the domain points that are in the exact_logdos
    est_masked = est[mask_rmse]
    return sqrt(mean((est_masked .- exact_masked).^2))
end

# Problem setup
L = 8
N = L * L

# Simulation parameters
n_iter = 10
sweeps_therm = 100
sweeps_record = 10_000

pmuca = ParallelMulticanonical(MPI.COMM_WORLD, root=0)
sys = IsingLatticeOptim(L, L)
init!(sys, :random, rng=MersenneTwister(1000 + pmuca.rank))
alg = Multicanonical(MersenneTwister(1000 + pmuca.rank), zero(exact_logdos))

if is_root(pmuca)
    println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    println("MPI Multicanonical (MUCA) Ising Simulation")
    println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    println("MPI ranks: ", pmuca.size)
end

if is_root(pmuca)
    println("\nStarting MUCA simulation...")
end

# Main iteration loop
for iter in 1:n_iter
    # Thermalization
    for _ in 1:(sweeps_therm * length(sys.spins))
        spin_flip!(sys, alg)
    end
    reset!(alg)
    # each rank does 1/nprocs of the total sweeps
    for i in 1:(sweeps_record * length(sys.spins)/pmuca.size) 
        spin_flip!(sys, alg)
    end

    merge_histograms!(pmuca, alg.histogram)
    if is_root(pmuca)
        update_weight!(alg; mode=:simple)
        print("Iteration $iter/$n_iter, RMSE (exact) = $(round(rmse_exact(alg.logweight), digits=4))\r")
    end
    distribute_logweight!(pmuca, alg.logweight)
end

MPI.Barrier(pmuca.comm)

if is_root(pmuca)
    final_rmse = rmse_exact(alg.logweight)
    println("\n✓ Simulation complete!")
    println("Final RMSE (vs exact Beale): $(round(final_rmse, digits=4))")
end

MPI.Finalize()
