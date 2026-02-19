push!(LOAD_PATH,"../src/")

using Documenter, MonteCarloX

#DocMeta.setdocmeta!(MonteCarloX, :DocTestSetup, :(using MonteCarloX); recursive = true)

makedocs(;
    modules = [MonteCarloX],
    format = Documenter.HTML(),
    pages = [
        "Getting Started" => "index.md",
        "Framework" => "framework.md",
        "Core Abstractions" => "core_abstractions.md",
        "Weights" => "weights.md",
        "Importance Sampling Algorithms" => "equilibrium.md",
        "Continuous-Time Sampling Algorithms" => "non_equilibrium.md",
        "Measurements" => "measurements.md",
        "Systems" => "systems.md",
        "Worked Examples" => "examples.md",
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
