push!(LOAD_PATH,"../src/")
using Documenter, Literate, MonteCarloX

# --- Logo: copy from top-level /logo into Documenter's assets folder ---
let
    src = joinpath(@__DIR__, "..", "logo", "logo.png")
    dst = joinpath(@__DIR__, "src", "assets", "logo.png")
    isfile(src) && cp(src, dst; force=true)
end

# --- Literate: process examples ---
example_dir   = joinpath(@__DIR__, "src", "examples")
generated_dir = joinpath(@__DIR__, "src", "generated")

# Top-level examples/ write/read their cached simulation outcomes here.
ENV["MCX_EXAMPLE_DATA"] = abspath(joinpath(@__DIR__, "src", "data"))

include(joinpath(@__DIR__, "src", "examples", "defaults.jl"))

for (root, dirs, files) in walkdir(example_dir)
    rel_path = relpath(root, example_dir)
    filter!(d -> d != "todos", dirs)
    if !contains(rel_path, "todos")
        for file in files
            if endswith(file, ".jl")            &&
               !endswith(file, "_mpi.jl")       &&
               !endswith(file, "_threads.jl")   &&
               file != "runtests.jl"            &&
               file != "defaults.jl"
                filepath = joinpath(root, file)
                Literate.markdown(filepath, generated_dir; documenter=true)
            end
        end
    end
end

# Top-level examples/ (Tier A run at build; Tier B load their cached TSV outcome).
toplevel_example_dir = joinpath(@__DIR__, "..", "examples")
skip_examples = ("reweighting.jl",)     # not yet migrated to the cached pattern
draft_dirs    = ("inference",)          # design-by-usage drafts (unbuilt API) — not executed yet
for (root, dirs, files) in walkdir(toplevel_example_dir)
    basename(root) in draft_dirs && continue
    for file in files
        endswith(file, ".jl")           || continue
        endswith(file, "_mpi.jl")       && continue
        endswith(file, "_threads.jl")   && continue
        file in skip_examples           && continue
        Literate.markdown(joinpath(root, file), generated_dir; documenter=true)
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
    ),
    draft = draft_docs,
    doctest = strict_docs,
    checkdocs = strict_docs ? :exports : :none,
    warnonly = !strict_docs,
    pages = [
        "Home" => "index.md",
        "Fundamentals" => [
            "Monte Carlo Fundamentals"   => "monte_carlo_fundamentals.md",
            "Systems"                    => "systems.md",
            "Build Your Own System"      => "build_your_own_system.md",
            "Weights and Ensembles"      => "weights.md",
        ],
        "Algorithm Classes" => [
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
            "Measurements"   => "measurements.md",
            "Checkpointing"  => "checkpointing.md",
            "Helpers"        => "helper.md",
        ],
        "Examples" => [
            "Getting Started" => [
                "Basic Sampling"                     => "generated/basic_sampling.md",
                "Coin Flip (Bayesian inference)"     => "generated/coin_flip.md",
                "Ising Model (importance sampling)"  => "generated/importance_Ising2D.md",
                "Birth-Death (Gillespie)"            => "generated/gillespie_birth_death.md",
            ],
            "Spins" => [
                "Ising 2D (importance sampling)"     => "generated/importance_Ising2D.md",
                "Ising 2D (parallel tempering)"      => "generated/pt_Ising2D.md",
                "Ising 2D (multicanonical)"          => "generated/muca_Ising2D.md",
                "Blume-Capel (multicanonical)"       => "generated/muca_BlumeCapel.md",
            ],
            "Soft Matter" => [
                "LJ gas (multicanonical)"            => "generated/muca_LJgas.md",
            ],
            "Bayesian Inference" => [
                "Coin Flip"                          => "generated/coin_flip.md",
                "Evidence (importance sampling)"     => "generated/reweighting_evidence.md",
                "House Price Prediction"             => "generated/house_price_prediction.md",
                "Eight Schools (hierarchical)"       => "generated/eight_schools.md",
            ],
            "Stochastic Processes" => [
                "Poisson Process (Gillespie)"        => "generated/kmc_poisson.md",
                "Dimerization (Gillespie)"           => "generated/gillespie_dimerization.md",
                "Ornstein-Uhlenbeck (multicanonical)" => "generated/muca_OU.md",
            ],
            "Large Deviation Theory" => [
                "Sum of Gaussians (multicanonical)"  => "generated/muca_sum_gaussian.md",
                "Ornstein-Uhlenbeck (multicanonical)" => "generated/muca_OU.md",
            ],
            "Infrastructure" => [
                "Reweighting"                        => "generated/reweighting.md",
                "Checkpointing"                      => "generated/checkpointing.md",
            ],
        ],
    ],
    sitename = "MonteCarloX",
    authors = "Johannes Zierenberg & Martin Weigel",
)

deploydocs(
    repo = "github.com/zierenberg/MonteCarloX.jl.git",
)
