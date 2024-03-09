push!(LOAD_PATH,"../src/")

using Documenter, MonteCarloX

#DocMeta.setdocmeta!(MonteCarloX, :DocTestSetup, :(using MonteCarloX); recursive = true)

makedocs(;
    modules = [MonteCarloX],
    format = Documenter.HTML(),
    pages = [
        "Getting Started" => "index.md",
        "Equilibrium Tools" => "equilibrium.md",
        "Non-Equilibrium Tools" => "non_equilibrium.md",
        "Helper" => "helper.md",
    ],
    sitename = "MonteCarloX",
    authors = "Johannes Zierenberg",
)

# deploydocs(;
#     deps = Deps.pip("pygments", "mkdocs", "python-markdown-math"),
#     repo = "github.com/zierenberg/MonteCarloX.jl.git",
#     target="build"
# )
deploydocs(
    repo = "github.com/USER_NAME/PACKAGE_NAME.jl.git",
)
