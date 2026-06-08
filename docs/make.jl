push!(LOAD_PATH,"../src/")
using Documenter, Literate, MonteCarloX

# --- Logo: copy from top-level /logo into Documenter's assets folder ---
let
    src = joinpath(@__DIR__, "..", "logo", "logo.png")
    dst = joinpath(@__DIR__, "src", "assets", "logo.png")
    isfile(src) && cp(src, dst; force=true)
end

# --- Literate: process examples ---
example_dir   = joinpath(@__DIR__, "src", "examples")
generated_dir = joinpath(@__DIR__, "src", "generated")

include(joinpath(@__DIR__, "src", "examples", "defaults.jl"))
for (root, dirs, files) in walkdir(example_dir)
    rel_path = relpath(root, example_dir)
    filter!(d -> d != "todos", dirs)
    if !contains(rel_path, "todos")
        for file in files
            if endswith(file, ".jl")            &&
               !endswith(file, "_mpi.jl")       &&
               !endswith(file, "_threads.jl")   &&
               file != "runtests.jl"            &&
               file != "defaults.jl"
                filepath = joinpath(root, file)
                Literate.markdown(filepath, generated_dir; documenter=true)
            end
        end
    end
end

# --- Documenter ---
strict_docs = get(ENV, "DOCS_STRICT", "false") == "true"

makedocs(;
    modules = [MonteCarloX],
    format = Documenter.HTML(;
        assets = ["assets/custom.css", "assets/custom.js"],
    ),
    doctest = strict_docs,
    checkdocs = strict_docs ? :exports : :none,
    warnonly = !strict_docs,
    pages = [
        "Getting Started" => "index.md",
        "Guides" => [
            "Monte Carlo Fundamentals"            => "monte_carlo_fundamentals.md",
            "Importance Sampling"                 => "importance_sampling_algorithms.md",
            "Continuous-Time Sampling"            => "continuous_time_sampling_algorithms.md",
            "Measurements"                        => "measurements.md",
            "Build Your Own System"               => "build_your_own_system.md",
            "Checkpointing"                      => "checkpointing.md",
        ],
        "Examples" => [
            "Getting Started with Examples" => [
            "Coin Flip (Bayesian inference)"          => "generated/coin_flip.md",
            "Importance sampling: Ising Model"        => "generated/importance_Ising2D.md",
            "Birth-Death Process (Gillespie)"         => "generated/gillespie_birth_death.md",
            ],
            "Spins" => [
            "Importance sampling: Ising 2D"           => "generated/importance_Ising2D.md",
            "Parallel tempering: Ising 2D"            => "generated/pt_Ising2D.md",
            "Multicanonical sampling: Ising 2D"       => "generated/muca_Ising2D.md",
            "Multicanonical sampling: Blume-Capel"    => "generated/muca_BlumeCapel.md",
            ],

            "Soft Matter" => [
            "Multicanonical sampling: LJ gas"         => "generated/muca_LJgas.md",
            ],

            "Bayesian Inference" => [
            "Importance sampling: Coin Flip"          => "generated/coin_flip.md",
            "Importance sampling: House Price Prediction" => "generated/house_price_prediction.md",
            "Hierarchical sampling: Eight Schools"    => "generated/eight_schools.md",
            ],

            "Stochastic Processes" => [
            "Gillespie: Poisson Process"              => "generated/kmc_poisson.md",
            "Gillespie: Dimerization Reaction"        => "generated/gillespie_dimerization.md",
            "Multicanonical: Ornstein-Uhlenbeck"      => "generated/muca_OU.md",
            ],

            "Infrastructure" => [
            "Checkpointing"                          => "generated/checkpointing.md",
            ],

            "Large Deviation Theory" => [
            "Multicanonical: Sum of Gaussians"        => "generated/muca_sum_gaussian.md",
            "Multicanonical: Ornstein-Uhlenbeck"      => "generated/muca_OU.md",
            ],
        ],
        "API Reference" => [
            "Systems"                             => "systems.md",
            "Weights"                             => "weights.md",
            "Helper"                              => "helper.md",
        ],
    ],
    sitename = "MonteCarloX",
    authors = "Johannes Zierenberg & Martin Weigel",
)

deploydocs(
    repo = "github.com/zierenberg/MonteCarloX.jl.git",
)
