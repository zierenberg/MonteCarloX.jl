using MonteCarloX
using Random
using Test

function test_event_handler_rate(event_handler_type::String)
    pass = true

    N = 1000
    rate0 = 0.1
    list_rate    = ones(N) * rate0
    list_event = [i for i = 1:N]
    if event_handler_type == "ListEventRateSimple"
        event_handler = ListEventRateSimple{Int}(list_event, list_rate, 0.0, 0)
    elseif event_handler_type == "ListEventRateActiveMask"
        event_handler = ListEventRateActiveMask{Int}(list_event, list_rate, 0.0, 0)
    else
        throw(UndefVarError(:event_handler_type))
    end
    pass &= check(event_handler.list_rate[1] == rate0, "rate[1] == rate0\n")
    pass &= check(event_handler.list_rate[N] == rate0, "rate[N] == rate0\n")
    pass &= check(length(event_handler) == N, "length == N\n")

    # set all rates to zero
    for i = 1:N
        event_handler[i] = 0.0
    end
    pass &= check(!(sum(event_handler.list_rate) > 1e-10), "all rates zeroed\n")
    pass &= check(length(event_handler) == 0, "length == 0 after zeroing\n")

    # set random rates and check sum tracking
    rng = MersenneTwister(1000)
    sum_before = sum(event_handler.list_rate)
    sum_changes = 0.0
    for i = 1:N
        value = rand(rng)
        sum_changes += value - event_handler[i]
        event_handler[i] = value
    end
    sum_expected = sum_before + sum_changes
    pass &= check(abs(sum(event_handler.list_rate) - sum_expected) < 1e-10, "sum tracking after random set\n")

    # stress test: many random updates, check sum remains consistent
    for round = 1:10
        for _ = 1:1e7
            event_handler[rand(rng, 1:N)] = rand(rng)
        end
        sum_rates = sum(event_handler[n] for n in 1:N)
        pass &= check(abs(sum(event_handler.list_rate) - sum_rates) < 1e-11, "sum consistent after stress round $round\n")
    end

    return pass
end

function test_event_queue()
    pass = true

    event_1 = (1.0, 1)
    event_2 = (2.0, 2)
    event_3 = (3.0, 3)

    queue = EventQueue{Int}()
    add!(queue, event_1)
    add!(queue, event_3)
    add!(queue, event_2)

    pass &= check(length(queue) == 3, "queue length == 3\n")
    pass &= check(queue[1] == event_1, "queue[1] is earliest\n")
    pass &= check(popfirst!(queue) == event_1, "popfirst! returns event_1\n")
    pass &= check(popfirst!(queue) == event_2, "popfirst! returns event_2\n")
    pass &= check(popfirst!(queue) == event_3, "popfirst! returns event_3\n")

    queue = EventQueue{Int}(nextfloat(1.0))
    pass &= check(get_time(queue) == nextfloat(1.0), "initial time set\n")
    add!(queue, event_3)
    add!(queue, event_2)
    @test_throws ErrorException add!(queue, event_1)
    pass &= check(popfirst!(queue) == event_2, "popfirst! skips past events\n")
    pass &= check(popfirst!(queue) == event_3, "popfirst! returns event_3\n")

    queue = EventQueue{Int}(0.5)
    add!(queue, (1.2, 7))
    alg = Gillespie(MersenneTwister(1))
    dt, ev = next(alg, queue)
    pass &= check(dt == 0.7, "dt from queue event\n")
    pass &= check(ev == 7, "event from queue\n")
    pass &= check(get_time(queue) == 1.2, "queue time advanced\n")

    return pass
end

@testset "Event Handler" begin
    @testset "ListEventRateSimple" begin
        @test test_event_handler_rate("ListEventRateSimple")
    end
    @testset "ListEventRateActiveMask" begin
        @test test_event_handler_rate("ListEventRateActiveMask")
    end
    @testset "EventQueue" begin
        @test test_event_queue()
    end
end
