
# Adapted from DFTK.jl/src/common/timer.jl

# Control whether timings are enabled or not, by default no
if get(ENV, "EPW_TIMING", "NONE") == "ALL"
    timer_enabled() = :all
else
    timer_enabled() = :none
end

"""TimerOutput object used to store EPW timings."""
const timer = TimerOutput()

"""
Shortened version of the `@timeit` macro from `TimerOutputs`,
which writes to the EPW timer.
"""
macro timing(args...)
    if EPW.timer_enabled() in (:parallel, :all)
        TimerOutputs.timer_expr(__module__, false, :($(EPW.timer)), args...)
    else  # Disable taking timings
        :($(esc(last(args))))
    end
end

# """
# Similar to `@timing`, but disabled in parallel runs.
# Should be used to time threaded regions,
# since TimerOutputs is not thread-safe and breaks otherwise.
# """
# macro timing_seq(args...)
#     if EPW.timer_enabled() == :all
#         TimerOutputs.timer_expr(__module__, false, :($(EPW.timer)), args...)
#     else  # Disable taking timings
#         :($(esc(last(args))))
#     end
# end
