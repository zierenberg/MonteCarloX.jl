push!(LOAD_PATH,"../src/")

using Documenter, MonteCarloX

#DocMeta.setdocmeta!(MonteCarloX, :DocTestSetup, :(using MonteCarloX); recursive = true)

makedocs(;
    modules = [MonteCarloX],
    format = Documenter.HTML(),
    pages = [
        "Getting Started" => "index.md",
        "Monte Carlo Fundamentals" => "monte_carlo_fundamentals.md",
        "Importance Sampling Algorithms" => "importance_sampling_algorithms.md",
        "Continuous-Time Sampling Algorithms" => "continuous_time_sampling_algorithms.md",
        "Build Your Own System" => "build_your_own_system.md",
        "Measurements" => "measurements.md",
        "Systems" => "systems.md",
        "Weights" => "weights.md",
        "Helper" => "helper.md",
    ],
    sitename = "MonteCarloX",
    authors = "Johannes Zierenberg & Martin Weigel",
)

# deploydocs(;
#     deps = Deps.pip("pygments", "mkdocs", "python-markdown-math"),
#     repo = "github.com/zierenberg/MonteCarloX.jl.git",
#     target="build"
# )
deploydocs(
    repo = "github.com/zierenberg/MonteCarloX.jl.git",
)
