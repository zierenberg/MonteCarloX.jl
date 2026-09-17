# Renders this folder's Sunny interop examples into markdown, with `execute = true` so
# Literate runs each script itself and bakes the output in as static text — no Sunny
# dependency needed by whatever later reads the generated markdown (e.g. Documenter).
#
# Run: `julia --project=examples/external/sunny examples/external/sunny/build_docs.jl [outdir]`
# `outdir` defaults to docs/src/generated; docs/make.jl passes it explicitly.

using Literate

outdir = isempty(ARGS) ? abspath(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "generated")) : ARGS[1]
mkpath(outdir)

is_source(f) = endswith(f, ".jl") && basename(f) != basename(@__FILE__)
for file in sort(filter(is_source, readdir(@__DIR__; join=true)))
    Literate.markdown(file, outdir; documenter=true, execute=true)
end
