# Renders this folder's Sunny interop examples into markdown, with `execute = true` so
# Literate runs each script itself and bakes the output in as static text — no Sunny
# dependency needed by whatever later reads the generated markdown (e.g. Documenter).
#
# Self-instantiates: CI only ever runs `Pkg.instantiate()` on docs/, never on this folder,
# so this has to make its own environment usable on a bare checkout.
#
# Run: `julia --project=examples/external/sunny examples/external/sunny/build_docs.jl [--rerun]`
# Any arguments (e.g. `--rerun`/`--reset`) are forwarded via `ARGS` to the rendered scripts,
# which check for them directly — this script itself takes no positional arguments, so
# ARGS doesn't need to double as anything else.

import Pkg
try
    Pkg.instantiate()
catch
    Pkg.resolve()
    Pkg.instantiate()
end

using Literate

outdir = abspath(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "generated"))
mkpath(outdir)

is_source(f) = endswith(f, ".jl") && basename(f) != basename(@__FILE__)
for file in sort(filter(is_source, readdir(@__DIR__; join=true)))
    Literate.markdown(file, outdir; documenter=true, execute=true)
end
