# # Benchmarks
#
# There are two important benchmarks for every Monte Carlo simulation: correctness and speed. 
# In this section, we compare the MCX library against (mostly Julia) packages on prime examples and compare both the results as well as the relative speed. 
# The goal is to show that MCX is not only working correctly, but also fast.
#
# Every comparison follows one protocol: the reference code runs **its own prime example** (the example its README leads with), MCX runs the identical physics, both on the same machine in the same session. 
# Agreement is checked against an exact referee wherever one exists (Beale's 2D-Ising density of states via `reweight`); a speed number without an agreement check is meaningless.
#
# Speed is reported as the **MCX speedup**
# ```math
# \text{speedup} = \frac{t_\text{reference}}{t_\text{MCX}},
# ```
# the per-attempted-flip time of the reference over that of MCX. 
# A speedup above 1 means MCX is faster; below 1 means slower. 
# The ratio cancels the machine, so it is portable; absolute ns/flip can be found on github (`docs/src/data/benchmarks.tsv`).
#
# This landing page collects the speedups; the per-system subpages carry the physics-agreement plots, the exact protocols, and the MCX code. 
# Currently the library covers:
#
# - [Spin systems](@ref) — Ising and Heisenberg against the exact prime examples of
#   [MonteCarlo.jl](https://github.com/carstenbauer/MonteCarlo.jl/tree/master/example/ising2d),
#   [Carlo.jl](https://github.com/lukas-weber/Ising.jl/tree/main/example) and
#   [SpinMC.jl](https://github.com/fbuessen/SpinMC.jl#specific-heat-and-magnetization-of-a-cubic-lattice-heisenberg-ferromagnet),
#   [Sunny.jl](https://github.com/SunnySuite/Sunny.jl/blob/main/examples/05_MC_Ising.jl).
#

using DelimitedFiles, Printf, Markdown                                            # hide

datadir = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "docs", "src", "data")))  # hide
factors_file = joinpath(datadir, "benchmarks.tsv")                                # hide
nothing                                                                           # hide

# ## Speedup across comparisons
#
# Range of the MCX speedup over the temperatures / cases of each protocol.
# External frameworks carry scheduler, measurement, and proposal machinery, so MCX's lean local-update loop runs several times faster.
# The one exception is the compiled-C row (speedup below 1): a bare-metal kernel with no framework at all is the speed ceiling, where MCX's generality inevitably costs a constant factor. The fuller hand-optimized-Julia dissection lives in the MCXSpins top-performance benchmark.

if isfile(factors_file)                                                           # hide
tbl = readdlm(factors_file, '\t'; header=true)[1]                                 # hide
ref_ns = Float64.(tbl[:, 4]); mcx_ns = Float64.(tbl[:, 5])                        # hide
speedup = ref_ns ./ mcx_ns                                                        # hide
prime_urls = [                                                                    # hide
    "MonteCarlo.jl" => "https://github.com/carstenbauer/MonteCarlo.jl/tree/master/example/ising2d",  # hide
    "Carlo.jl"      => "https://github.com/lukas-weber/Ising.jl/tree/main/example",  # hide
    "SpinMC.jl"     => "https://github.com/fbuessen/SpinMC.jl#specific-heat-and-magnetization-of-a-cubic-lattice-heisenberg-ferromagnet",  # hide
    "Sunny.jl"      => "https://github.com/SunnySuite/Sunny.jl/blob/main/examples/05_MC_Ising.jl",  # hide
]                                                                                 # hide
linked(lab) = (i = findfirst(p -> startswith(lab, first(p)), prime_urls);         # hide
               isnothing(i) ? lab : "[$lab]($(last(prime_urls[i])))")             # hide
io = IOBuffer()                                                                   # hide
println(io, "| Reference (prime example) | MCX speedup (t_ref / t_MCX) |")   # hide
println(io, "|---|---|")                                                          # hide
for lab in unique(String.(tbl[:, 1]))                                             # hide
    s = speedup[String.(tbl[:, 1]) .== lab]                                       # hide
    r = length(s) == 1 ? @sprintf("%.2f×", s[1]) :                               # hide
        @sprintf("%.2f× – %.2f×", minimum(s), maximum(s))                         # hide
    println(io, "| ", linked(lab), " | ", r, " |")                                # hide
end                                                                               # hide
Markdown.parse(String(take!(io)))                                                 # hide
end                                                                               # hide

# Higher is faster: MCX outpaces every framework on its own prime example, and trails only the bare-metal compiled-C ceiling.
# The detail page shows that this speed buys generic, model-agnostic sampling: the same `spin_flip!` loop drives Ising and Heisenberg (and every other MCXSpins model) without specialization.
nothing                                                                           # hide
