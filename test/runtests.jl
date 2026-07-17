using MonteCarloX
using Test
using Random

test_verbose = false

function check(cond::Bool, message::String)
    if test_verbose
        if cond
            printstyled(message; color = :green)
        else
            printstyled(message; color = :red)
        end
    end
    return cond
end

@testset "MonteCarloX.jl" begin
    include("test_algorithm_steps.jl")
    include("test_binned_objects.jl")
    include("test_checkpointing.jl")
    include("test_ensembles.jl")
    include("test_event_handler.jl")
    include("test_event_rate_tree.jl")
    include("test_kinetic_monte_carlo.jl")
    include("test_site_events.jl")
    include("test_measurements.jl")
    include("test_message_backend.jl")
    include("test_balance.jl")
    include("test_markov_chain_monte_carlo.jl")
    include("test_step_size.jl")
    include("test_log_density_problems.jl")
    include("test_diagnostics.jl")
    include("test_multicanonical.jl")
    include("test_reweighting.jl")
    include("test_rejection_sampling.jl")
    include("test_parallel_ensembles.jl")
    include("test_rng.jl")
    include("test_utils.jl")
    include("test_wang_landau.jl")

    # Multi-rank MPI tests (opt-in): the single-process suite can only build 1-rank backends,
    # so the real cross-rank replica-exchange path is exercised here under `mpiexec -n 2`.
    # Enable with `MCX_TEST_MPI=true` (needs a working MPI runtime).
    if get(ENV, "MCX_TEST_MPI", "false") == "true"
        @testset "MPI replica exchange (2 ranks)" begin
            using MPI
            script = joinpath(@__DIR__, "mpi", "test_replica_exchange_mpi.jl")
            cmd = `$(mpiexec()) -n 2 $(Base.julia_cmd()) --project=$(dirname(@__DIR__)) $script`
            @test success(run(ignorestatus(cmd)))
        end
    end
end
