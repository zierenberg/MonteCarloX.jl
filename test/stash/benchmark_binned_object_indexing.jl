using MonteCarloX
using Random
using Printf

# Reference implementation used previously for ND indexing.
@inline function old_getindex(bo::MonteCarloX.BinnedObject{1}, xs::NTuple{1,Real})
    idx = MonteCarloX._binindex(bo.bins[1], xs[1])
    return bo.values[idx]
end

@inline function old_getindex(bo::MonteCarloX.BinnedObject{N}, xs::NTuple{N,Real}) where {N}
    idxs = [MonteCarloX._binindex(bo.bins[i], xs[i]) for i in 1:N]
    return bo.values[idxs...]
end

# Current implementation path.
@inline function new_getindex(bo::MonteCarloX.BinnedObject{N}, xs::NTuple{N,Real}) where {N}
    return bo[xs...]
end

function benchmark_case(name::AbstractString, bo, points; reps::Int=40)
    ncalls = reps * length(points)

    # Warm-up to ensure JIT compilation is out of the timed region.
    warm = 0.0
    @inbounds for p in points
        warm += old_getindex(bo, p)
        warm += new_getindex(bo, p)
    end

    old_sum = Ref(0.0)
    old_t = @elapsed begin
        s = 0.0
        for _ in 1:reps
            @inbounds for p in points
                s += old_getindex(bo, p)
            end
        end
        old_sum[] = s
    end
    old_alloc = @allocated begin
        s = 0.0
        for _ in 1:reps
            @inbounds for p in points
                s += old_getindex(bo, p)
            end
        end
    end

    new_sum = Ref(0.0)
    new_t = @elapsed begin
        s = 0.0
        for _ in 1:reps
            @inbounds for p in points
                s += new_getindex(bo, p)
            end
        end
        new_sum[] = s
    end
    new_alloc = @allocated begin
        s = 0.0
        for _ in 1:reps
            @inbounds for p in points
                s += new_getindex(bo, p)
            end
        end
    end

    @assert isapprox(old_sum[], new_sum[]; atol=0.0, rtol=0.0)

    println("\n=== $(name) ===")
    @printf("calls                 : %d\n", ncalls)
    @printf("old time              : %.3f ms\n", old_t * 1e3)
    @printf("new time              : %.3f ms\n", new_t * 1e3)
    @printf("speedup (old/new)     : %.2fx\n", old_t / new_t)
    @printf("old alloc             : %.3f MB\n", old_alloc / 1024^2)
    @printf("new alloc             : %.3f MB\n", new_alloc / 1024^2)
    @printf("alloc reduction       : %.2fx\n", old_alloc / max(new_alloc, 1))
    @printf("old bytes/call        : %.2f\n", old_alloc / ncalls)
    @printf("new bytes/call        : %.2f\n", new_alloc / ncalls)

    return nothing
end

function build_cases(rng::AbstractRNG)
    npoints = 20_000

    bo_disc1 = BinnedObject(0:1:2000, 0.0)
    bo_disc1.values .= rand(rng, size(bo_disc1.values)...)
    pts_disc1 = [(rand(rng) * 2000.0,) for _ in 1:npoints]

    edges1 = collect(range(-8.0, 8.0; length=2001))
    bo_cont1 = BinnedObject(edges1, 0.0; interpretation=:continuous)
    bo_cont1.values .= rand(rng, size(bo_cont1.values)...)
    pts_cont1 = [(-8.0 + 16.0 * rand(rng),) for _ in 1:npoints]

    bo_disc2 = BinnedObject((0:1:200, 0:1:200), 0.0)
    bo_disc2.values .= rand(rng, size(bo_disc2.values)...)
    pts_disc2 = [(rand(rng) * 200.0, rand(rng) * 200.0) for _ in 1:npoints]

    edges = collect(range(-5.0, 5.0; length=401))
    bo_cont2 = BinnedObject((edges, edges), 0.0; interpretation=:continuous)
    bo_cont2.values .= rand(rng, size(bo_cont2.values)...)
    pts_cont2 = [(-5.0 + 10.0 * rand(rng), -5.0 + 10.0 * rand(rng)) for _ in 1:npoints]

    bo_disc3 = BinnedObject((0:1:60, 0:1:60, 0:1:60), 0.0)
    bo_disc3.values .= rand(rng, size(bo_disc3.values)...)
    pts_disc3 = [(rand(rng) * 60.0, rand(rng) * 60.0, rand(rng) * 60.0) for _ in 1:npoints]

    return [
        ("Discrete 1D", bo_disc1, pts_disc1),
        ("Continuous 1D", bo_cont1, pts_cont1),
        ("Discrete 2D", bo_disc2, pts_disc2),
        ("Continuous 2D", bo_cont2, pts_cont2),
        ("Discrete 3D", bo_disc3, pts_disc3),
    ]
end

function main()
    println("BinnedObject indexing benchmark")
    println("Comparing old ND indexing (vector indices) vs current tuple-based indexing")

    rng = MersenneTwister(42)
    for (name, bo, points) in build_cases(rng)
        benchmark_case(name, bo, points)
    end

    return nothing
end

main()
