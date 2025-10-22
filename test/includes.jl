using Printf

test_verbose=false

if !@isdefined(CHECK_UTILITIES_LOADED)
    const CHECK_UTILITIES_LOADED = true

    function check(cond::Bool, message::String)
        if test_verbose
            if cond
                printstyled(message; color = :green)
            else
                printstyled(message; color = :red)
            end
        end
        return cond
    end 
end

