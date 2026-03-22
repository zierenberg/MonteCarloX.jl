using Pkg
Pkg.activate(@__DIR__)

using JSON3

# run test of examples in smoke mode where each loop with big numbers is replaced with a loop capped at a smaller number.
const ROOT = normpath(joinpath(@__DIR__, ".."))
const SMOKE_MODE = get(ENV, "MCX_SMOKE", get(ENV, "MCX_CI", "false")) == "true"
const LOOP_CAP = tryparse(Int, get(ENV, "MCX_SMOKE_LOOP_CAP", get(ENV, "MCX_CI_LOOP_CAP", "2000")))

function cap_loops(code::AbstractString, cap::Int)
    cap <= 0 && return code
    pattern = r"for\s+([_A-Za-z][_A-Za-z0-9]*)\s+in\s+1\s*:\s*([0-9][0-9_]*)"
    replacement = SubstitutionString("for \\1 in 1:min($(cap), \\2)")
    return replace(code, pattern => replacement)
end

function sanitize_ci_code(code::AbstractString)
    lines = split(code, '\n')
    kept = String[]
    for line in lines
        stripped = strip(line)
        if occursin("Pkg.activate", stripped) || occursin("Pkg.instantiate", stripped)
            continue
        end
        push!(kept, line)
    end
    return join(kept, "\n")
end

function run_notebook(path::String; ci_mode::Bool=false, loop_cap::Int=0)
    println("\n=== Notebook: $(relpath(path, ROOT)) ===")
    nb = JSON3.read(read(path, String))
    cells = get(nb, :cells, Any[])

    mod = Module(Symbol("MCXNotebook_", hash(path)))
    Core.eval(mod, :(using MonteCarloX, Random, StatsBase, LinearAlgebra, Distributions, Plots, SpinSystems, InteractiveUtils))

    for (idx, cell) in enumerate(cells)
        celltype = String(get(cell, :cell_type, ""))
        celltype == "code" || continue

        src = join(String.(get(cell, :source, String[])), "\n")
        isempty(strip(src)) && continue

        if ci_mode
            src = sanitize_ci_code(src)
            src = cap_loops(src, loop_cap)
        end

        try
            Base.include_string(mod, src, "$(basename(path))#cell$(idx)")
        catch err
            error("Failed notebook $(relpath(path, ROOT)) at code cell #$idx: $err")
        end
    end
end

function run_script(path::String)
    println("\n=== Script: $(relpath(path, ROOT)) ===")
    include(path)
end

function discover_notebooks()
    all = String[]

    for entry in readdir(@__DIR__; join=true)
        isfile(entry) || continue
        endswith(entry, ".ipynb") || continue
        push!(all, entry)
    end

    for entry in readdir(@__DIR__; join=true)
        isdir(entry) || continue
        basename(entry) == "stash" && continue
        for (dir, _, files) in walkdir(entry)
            occursin("/stash", replace(dir, '\\' => '/')) && continue
            for f in files
                endswith(f, ".ipynb") || continue
                push!(all, joinpath(dir, f))
            end
        end
    end
    sort!(all)
    return all
end

function discover_scripts()
    all = String[]

    for entry in readdir(@__DIR__; join=true)
        isfile(entry) || continue
        endswith(entry, ".jl") || continue
        basename(entry) == "runtests.jl" && continue
        occursin("mpi", lowercase(basename(entry))) && continue
        push!(all, entry)
    end

    for entry in readdir(@__DIR__; join=true)
        isdir(entry) || continue
        basename(entry) == "stash" && continue
        for (dir, _, files) in walkdir(entry)
            occursin("/stash", replace(dir, '\\' => '/')) && continue
            for f in files
                endswith(f, ".jl") || continue
                f == "runtests.jl" && continue
                occursin("mpi", lowercase(f)) && continue
                push!(all, joinpath(dir, f))
            end
        end
    end
    sort!(all)
    return all
end

function main()
    notebooks = discover_notebooks()
    scripts = discover_scripts()

    isempty(notebooks) && isempty(scripts) && error("No maintained examples found under examples/.")

    println("Smoke test mode: ", SMOKE_MODE)
    println("Loop cap: ", isnothing(LOOP_CAP) ? 0 : LOOP_CAP)
    println("Notebooks: ", length(notebooks), " | Scripts: ", length(scripts))

    for path in notebooks
        run_notebook(path; ci_mode=SMOKE_MODE, loop_cap=something(LOOP_CAP, 0))
    end

    for path in scripts
        run_script(path)
    end

    println("\nAll maintained examples completed.")
end

main()
