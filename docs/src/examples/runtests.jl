## docs/src/examples/runtests.jl
##
## Smoke tests for all examples. Run via:
##   julia --project=docs docs/src/examples/runtests.jl
##
## Environment variables:
##   MCX_SMOKE=true          — enable smoke mode (also triggered by MCX_CI=true)
##   MCX_SMOKE_LOOP_CAP=N    — cap all numeric for-loops to N iterations (default 2000)

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))   ## activate docs/ — everything is already there
Pkg.instantiate()

const EXAMPLES_DIR = @__DIR__
const SMOKE_MODE   = get(ENV, "MCX_SMOKE", get(ENV, "MCX_CI", "false")) == "true"
const LOOP_CAP     = something(
    tryparse(Int, get(ENV, "MCX_SMOKE_LOOP_CAP", get(ENV, "MCX_CI_LOOP_CAP", ""))),
    2000,
)

## -----------------------------------------------------------------------
## Helpers
## -----------------------------------------------------------------------

function cap_loops(code::AbstractString, cap::Int)
    cap <= 0 && return code
    ## cap any for loop of the form: for x in 1:ANYTHING
    pattern     = r"(for\s+[_A-Za-z][_A-Za-z0-9]*\s+in\s+1\s*:)([^\n,\]]+)"
    replacement = SubstitutionString("\\1min($(cap), \\2)")
    return replace(code, pattern => replacement)
end

function sanitize_src_code(code::AbstractString)
    ## strip lines ending in #src (Pkg.activate, Pkg.instantiate, include defaults.jl, etc.)
    lines = filter(split(code, '\n')) do line
        !endswith(rstrip(line), "#src")
    end
    return join(lines, '\n')
end

function run_script(path::String; smoke::Bool=false, cap::Int=0)
    println("\n=== $(relpath(path, EXAMPLES_DIR)) ===")
    code = read(path, String)
    code = sanitize_src_code(code)
    smoke && (code = cap_loops(code, cap))
    mod  = Module(Symbol("MCXExample_", hash(path)))
    Base.include_string(mod, code, basename(path))
end

## -----------------------------------------------------------------------
## Discovery — skips _mpi files, defaults.jl, runtests.jl
## -----------------------------------------------------------------------

function discover_scripts(root::String)
    scripts = String[]
    for (dir, _, files) in walkdir(root)
        for f in files
            endswith(f, ".jl")             || continue
            f == "runtests.jl"             && continue
            f == "defaults.jl"             && continue
            occursin("_mpi", lowercase(f)) && continue
            push!(scripts, joinpath(dir, f))
        end
    end
    return sort!(scripts)
end

## -----------------------------------------------------------------------
## Main
## -----------------------------------------------------------------------

function main()
    scripts = discover_scripts(EXAMPLES_DIR)
    isempty(scripts) && error("No examples found under $(EXAMPLES_DIR)")

    println("Smoke mode : ", SMOKE_MODE)
    println("Loop cap   : ", LOOP_CAP)
    println("Examples   : ", length(scripts))

    n_pass = 0
    n_fail = 0

    for path in scripts
        try
            run_script(path; smoke=SMOKE_MODE, cap=LOOP_CAP)
            println("  ✓")
            n_pass += 1
        catch e
            println("  ✗")
            @warn "Failed: $(relpath(path, EXAMPLES_DIR))" exception=(e, catch_backtrace())
            n_fail += 1
        end
    end

    println("\nResults: $n_pass passed, $n_fail failed out of $(n_pass + n_fail)")
    n_fail > 0 && exit(1)
end

main()