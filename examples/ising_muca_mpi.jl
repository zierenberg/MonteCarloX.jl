"""
ising_muca_mpi.jl

Multicanonical (MUCA) sampling for 2D Ising model using MPI for parallel replica exchange.
Run with: mpiexec -n 4 julia --project ising_muca_mpi.jl

Each MPI rank runs one independent replica and exchanges histograms via Allreduce.
"""

using Pkg
Pkg.instantiate()

# Add local dev packages to load path
push!(LOAD_PATH, joinpath(dirname(@__DIR__), "SpinSystems", "src"))

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

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs_mpi = MPI.Comm_size(comm)
mpi_thread_level = MPI.Query_thread()

if rank == 0
    println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    println("MPI Multicanonical (MUCA) Ising Simulation")
    println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    println("MPI ranks: ", nprocs_mpi)
    println("MPI thread level: ", mpi_thread_level)
end

# Problem setup
L = 8
N = L * L
bins = collect((-2N - 2):4:(2N + 2))
E_axis = Int.(bins[1:end-1] .+ 2)

# Reference exact log-DOS for validation (Beale solution for 8×8 Ising)
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

function rmse_exact(lw::TabulatedLogWeight)
    est = -Float64.(lw.table.weights)
    ref = [haskey(exact_logdos, E) ? exact_logdos[E] : NaN for E in E_axis]
    i0 = findfirst(==(0), E_axis)
    if i0 !== nothing && isfinite(ref[i0])
        est .-= est[i0]
        ref .-= ref[i0]
    end
    idx = findall(i -> isfinite(ref[i]) && isfinite(est[i]), eachindex(E_axis))
    return sqrt(mean((est[idx] .- ref[idx]).^2))
end

# Simulation parameters
n_iter = 10
sweeps_therm = 80
sweeps_record = 1000

# Per-rank initialization
sys_local = IsingLatticeOptim(L, L)
init!(sys_local, :random, rng=MersenneTwister(1000 + rank))

lw_local = TabulatedLogWeight(Histogram((collect(bins),), zeros(Float64, length(bins)-1)))
rep_local = Multicanonical(MersenneTwister(2000 + rank), lw_local)

meas_local = Measurements([:energy_hist => (s -> Int(energy(s))) => fit(Histogram, Int[], bins)], interval=1)

# Storage
mpi_hists = Histogram[]
mpi_lws = TabulatedLogWeight[]

w_local = zeros(Float64, length(bins)-1)
w_global = similar(w_local)

if rank == 0
    println("\nStarting MUCA simulation...")
end

# Main iteration loop
for iter in 1:n_iter
    # Thermalization
    for _ in 1:(sweeps_therm * length(sys_local.spins))
        spin_flip!(sys_local, rep_local)
    end

    # Measurement
    reset!(meas_local)
    for i in 1:(sweeps_record * length(sys_local.spins))
        spin_flip!(sys_local, rep_local)
        measure!(meas_local, sys_local, i)
    end

    h_local = meas_local[:energy_hist].data

    # Global histogram reduction via MPI
    w_local .= Float64.(h_local.weights)
    MPI.Allreduce!(w_local, w_global, MPI.SUM, comm)

    h_global = deepcopy(h_local)
    h_global.weights .= w_global

    # Update weights on master, then broadcast to all ranks
    update_weights!(rep_local, h_global; mode=:simple)
    MPI.Bcast!(rep_local.logweight.table.weights, 0, comm)

    # Store results
    push!(mpi_hists, deepcopy(h_global))
    push!(mpi_lws, deepcopy(rep_local.logweight))

    if rank == 0
        rmse = rmse_exact(rep_local.logweight)
        println("iter $(iter): RMSE = $(round(rmse, digits=4))")
    end
end

MPI.Barrier(comm)

if rank == 0
    final_rmse = rmse_exact(mpi_lws[end])
    println("\n✓ Simulation complete!")
    println("Final RMSE (vs exact Beale): $(round(final_rmse, digits=4))")
end

MPI.Finalize()
