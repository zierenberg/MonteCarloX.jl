using Documenter, MonteCarloX

DocMeta.setdocmeta!(MonteCarloX, :DocTestSetup, :(using MonteCarloX); recursive = true)

makedocs(;
    modules = [MonteCarloX],
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
    ],
    repo = "https://github.com/zierenberg/MonteCarloX.jl/blob/{commit}{path}#L{line}",
    sitename = "MonteCarloX.jl",
    authors = "Johannes Zierenberg",)

deploydocs(;
    deps = Deps.pip("pygments", "mkdocs", "python-markdown-math"),
    repo = "github.com/zierenberg/MonteCarloX.jl",
)
