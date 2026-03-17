push!(LOAD_PATH,"../src/")
using Documenter, Literate, MonteCarloX

# --- Literate: process examples ---

example_dir   = joinpath(@__DIR__, "src", "examples")
generated_dir = joinpath(@__DIR__, "src", "generated")

for file in readdir(example_dir)
    if endswith(file, ".jl") && !endswith(file, "_mpi.jl")
        filepath = joinpath(example_dir, file)
        Literate.markdown(filepath, generated_dir; documenter=true)
        Literate.notebook(filepath, generated_dir; execute=false)
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
        "Getting Started"                      => "index.md",
        "Monte Carlo Fundamentals"             => "monte_carlo_fundamentals.md",
        "Importance Sampling Algorithms"       => "importance_sampling_algorithms.md",
        "Continuous-Time Sampling Algorithms"  => "continuous_time_sampling_algorithms.md",
        "Build Your Own System"                => "build_your_own_system.md",
        "Measurements"                         => "measurements.md",
        "Systems"                              => "systems.md",
        "Weights"                              => "weights.md",
        "Helper"                               => "helper.md",
        "Examples" => [
            "Multicanonical Ising 2D" => "generated/muca_ising2D.md",
        ],
    ],
    sitename = "MonteCarloX",
    authors = "Johannes Zierenberg & Martin Weigel",
)

deploydocs(
    repo = "github.com/zierenberg/MonteCarloX.jl.git",
)