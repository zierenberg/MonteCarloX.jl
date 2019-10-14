using Documenter, MonteCarloX

makedocs(;
    modules=[MonteCarloX],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/zierenberg/MonteCarloX.jl/blob/{commit}{path}#L{line}",
    sitename="MonteCarloX.jl",
    authors="Johannes Zierenberg",
    assets=String[],
)

deploydocs(;
    repo="github.com/zierenberg/MonteCarloX.jl",
)
