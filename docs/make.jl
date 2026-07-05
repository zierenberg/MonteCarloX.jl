push!(LOAD_PATH,"../src/")
using Documenter, Literate, MonteCarloX

# --- Logo: copy from top-level /logo into Documenter's assets folder ---
let
    src = joinpath(@__DIR__, "..", "logo", "logo.png")
    dst = joinpath(@__DIR__, "src", "assets", "logo.png")
    isfile(src) && cp(src, dst; force=true)
end

# --- Literate: process examples (single source: the top-level examples/) ---
generated_dir = joinpath(@__DIR__, "src", "generated")

# Examples read (and, when run standalone, write) their cached data here.
ENV["MCX_EXAMPLE_DATA"] = abspath(joinpath(@__DIR__, "src", "data"))

# Plotting defaults for docs rendering (transparent backgrounds, etc.).
include(joinpath(@__DIR__, "..", "examples", "defaults.jl"))

# Every runnable example lives in the top-level examples/. Light ones execute at
# build; heavy ones cache their outcome into docs/src/data/ and only reload. Skipped:
# parallel (_mpi/_threads) reference scripts, incomplete todos/, the smoke runner, the
# plotting defaults, and reweighting (kept as a script — no docs page yet).
example_dir = joinpath(@__DIR__, "..", "examples")
skip_files  = ("reweighting.jl", "defaults.jl", "runtests.jl")
skip_dirs   = ("todos",)
for (root, dirs, files) in walkdir(example_dir)
    basename(root) in skip_dirs && continue
    for file in files
        endswith(file, ".jl")         || continue
        endswith(file, "_mpi.jl")     && continue
        endswith(file, "_threads.jl") && continue
        file in skip_files            && continue
        Literate.markdown(joinpath(root, file), generated_dir; documenter = true)
    end
end

# --- Documenter ---
strict_docs = get(ENV, "DOCS_STRICT", "false") == "true"
draft_docs  = get(ENV, "DOCS_DRAFT",  "false") == "true"

makedocs(;
    modules = [MonteCarloX],
    format = Documenter.HTML(;
        assets = ["assets/custom.css", "assets/custom.js"],
        collapselevel = 1,
        # plot-heavy example pages: inline the SVG outputs instead of the file-fallback
        # (which warns), and don't cap page size.
        example_size_threshold = nothing,
        size_threshold = nothing,
        size_threshold_warn = nothing,
    ),
    draft = draft_docs,
    doctest = strict_docs,
    checkdocs = strict_docs ? :exports : :none,
    warnonly = !strict_docs,
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "Monte Carlo Fundamentals"   => "getting_started/monte_carlo_fundamentals.md",
            "Systems"                    => "getting_started/systems.md",
            "Build Your Own System"      => "getting_started/build_your_own_system.md",
            "Weights and Ensembles"      => "getting_started/weights.md",
        ],
        "Algorithm Classes" => [
            "Basic Sampling"    => "algorithms/basic_sampling.md",
            "Markov Chain Monte Carlo" => [
                "Overview"        => "algorithms/markov_chain_monte_carlo.md",
                "Metropolis"      => "algorithms/metropolis.md",
                "Multicanonical"  => "algorithms/multicanonical.md",
            ],
            "Kinetic Monte Carlo" => [
                "Overview"        => "algorithms/kinetic_monte_carlo.md",
            ],
            "Population Monte Carlo" => [
                "Overview"        => "algorithms/population_monte_carlo.md",
            ],
        ],
        "Infrastructure" => [
            "Measurements"   => "infrastructure/measurements.md",
            "Reweighting"    => "infrastructure/reweighting.md",
            "Checkpointing"  => "infrastructure/checkpointing.md",
            "Helpers"        => "infrastructure/helper.md",
        ],
        "Examples" => [
            "Markov Chain Monte Carlo" => [
                "Ising 2D (importance sampling)"      => "generated/importance_Ising2D.md",
                "Ising 2D (parallel tempering)"       => "generated/pt_Ising2D.md",
                "Ising 2D (multicanonical)"           => "generated/muca_Ising2D.md",
                "Blume-Capel (multicanonical)"        => "generated/muca_BlumeCapel.md",
                "LJ gas (multicanonical)"             => "generated/muca_LJgas.md",
                "Ornstein-Uhlenbeck (multicanonical)" => "generated/muca_OU.md",
                "Sum of Gaussians (multicanonical)"   => "generated/muca_sum_gaussian.md",
                "Ising 2D (checkpointing)"            => "generated/checkpoint_Ising2D.md",
            ],
            "Kinetic Monte Carlo" => [
                "Birth-Death (Gillespie)"             => "generated/gillespie_birth_death.md",
                "Dimerization (Gillespie)"            => "generated/gillespie_dimerization.md",
                "Poisson Process (Gillespie)"         => "generated/kmc_poisson.md",
                "Contact Process (Gillespie)"         => "generated/contact_process.md",
                "Hawkes Process"                      => "generated/hawkes_process.md",
            ],
            "Inference" => [
                "Coin Flip"                           => "generated/coin_flip.md",
                "Conjugate Gaussian"                  => "generated/gaussian.md",
                "Eight Schools (hierarchical)"        => "generated/eight_schools.md",
                "SIR (dynamical model)"               => "generated/sir.md",
                "House Price Prediction"              => "generated/house_price_prediction.md",
            ],
            "Infrastructure" => [
                "Checkpointing"                       => "generated/checkpointing.md",
            ],
        ],
    ],
    sitename = "MonteCarloX",
    authors = "Johannes Zierenberg & Martin Weigel",
)

deploydocs(
    repo = "github.com/zierenberg/MonteCarloX.jl.git",
)
