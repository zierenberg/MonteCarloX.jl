# tests for src/event_handler.jl
using MonteCarloX
using StatsBase
using Random

function test_event_handler_rate(event_handler_type::String; verbose = false)
    pass = true

    # test construction
    N = 1000
    rate0 = 0.1
    list_rate    = ones(N) * rate0
    list_event = [i for i = 1:N]
    if event_handler_type == "ListEventRateSimple"
        event_handler =    ListEventRateSimple{Int}(list_event, list_rate, 0.0, 0)
    elseif event_handler_type == "ListEventRateActiveMask"
        event_handler = ListEventRateActiveMask{Int}(list_event, list_rate, 0.0, 0)
    else
        throw(UndefVarError(:event_handler_type))
    end
    if verbose
        println("... construction")
        println("... ... rate[1]: $(event_handler.list_rate[1]) == $(rate0)")
        println("... ... rate[N]: $(event_handler.list_rate[N]) == $(rate0)")
    end
    pass &= event_handler.list_rate[1] == rate0
    pass &= event_handler.list_rate[N] == rate0
     
    # test length
    if verbose
        println("... length: $(length(event_handler)) = $(N)")
    end
    pass &= length(event_handler) == N
     
    # test setindex! - e.g. sum_rates remains correct (random changes and keep sum of changes)
    for i = 1:N
        event_handler[i] = 0.0
    end
    if verbose
        println("... set all rates to 0.0")
        println("... ... sum(rates): $(sum(event_handler.list_rate)) !> 1e-10")
        println("... ... length: $(length(event_handler)) == 0")
    end
    pass &= !(sum(event_handler.list_rate) > 1e-10)
    pass &= length(event_handler) == 0

    # test setindex! and getindex
    rng = MersenneTwister(1000)
    sum_before = sum(event_handler.list_rate)
    sum_changes = 0.0
    for i = 1:N
        value = rand(rng)
        sum_changes += value - event_handler[i]
        event_handler[i] = value
    end
    sum_expected = sum_before + sum_changes
    if verbose
        println("... set all rates random")
        println("... ... expected sum(rates): $(sum(event_handler.list_rate)) == $(sum_expected)")
        println("... ... diff: $(abs(sum(event_handler.list_rate) - sum_expected))  < 1e-10")
    end
    pass &= abs(sum(event_handler.list_rate) - sum_expected) < 1e-10

    # test changes in rate effecting the automatic update of sum(rates)
    if verbose
        println("... automatic update of sum(rates)")
    end
    for i = 1:10
        valid_sum = true
        for j = 1:1e7
            event_handler[rand(rng, 1:N)] = rand(rng)
        end
        sum_rates = 0.0
        for n=1:N
            sum_rates += event_handler[n]
        end
        if abs(sum(event_handler.list_rate) - sum_rates) < 1e-11
            valid_sum = true
        else
            valid_sum = false
        end
        if verbose
            if valid_sum 
                println("... ... correct sum(rates): $(sum(event_handler.list_rate)) == $(sum_rates)")
            else
                println("... ... incorrect sum(rates): $(sum(event_handler.list_rate)) == $(sum_rates)")
            end
        end
        pass &= valid_sum
    end

    return pass
end

function test_event_queue(;verbose = false)
    pass = true

    event_1 = (1.0,1)
    event_2 = (2.0,2)
    event_3 = (3.0,3)

    # test sorted insert
    queue = EventQueue{Int}()
    add!(queue, event_1)
    add!(queue, event_3)
    add!(queue, event_2)
    # length
    pass &= length(queue) == 3
    # access
    pass &= queue[1] ==event_1
    # popfirst
    pass &= popfirst!(queue) == event_1
    pass &= popfirst!(queue) == event_2
    pass &= popfirst!(queue) == event_3
    if verbose
        println("... breakpoint 1: $(pass)")
    end


    # test non-default start time
    queue = EventQueue{Int}(1.0)
    pass &= MonteCarloX.get_last_time(queue)==1.0
    add!(queue, event_3)
    add!(queue, event_2)
    try
        add!(queue, event_1)
    catch
        pass &= false
    end
    pass &= popfirst!(queue) == event_1
    pass &= popfirst!(queue) == event_2
    pass &= popfirst!(queue) == event_3
    if verbose
        println("... breakpoint 2: $(pass)")
    end

    # test error upon adding event with past time
    queue = EventQueue{Int}(nextfloat(first(event_1)))
    try
        add!(queue, event_1)
    catch
        pass &= true
    end
    MonteCarloX.set_last_time!(queue, 0)
    try
        add!(queue, event_1)
    catch
        pass &= false
    end
    if verbose
        println("... breakpoint 3: $(pass)")
    end


    # test manipulation -> not clear how to deal with this in terms of API
    # queue = EventQueue{Int}()
    # add!(queue, event_1)
    # add!(queue, event_2)
    # add!(queue, event_3)
    # queue[1] = event_2
    # pass &= queue[1] == event_2
    # pass &= queue[2] == event_2
    # queue[1] = event_3


    return pass
end
