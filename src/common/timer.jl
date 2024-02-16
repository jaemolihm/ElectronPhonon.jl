
# Adapted from DFTK.jl/src/common/timer.jl

# Control whether timings are enabled or not, by default no
if get(ENV, "ElectronPhonon_TIMING", "NONE") == "ALL"
    timer_enabled() = :all
else
    timer_enabled() = :none
end

"""TimerOutput object used to store ElectronPhonon timings."""
const timer = TimerOutput()

"""
Shortened version of the `@timeit` macro from `TimerOutputs`,
which writes to the ElectronPhonon timer.
"""
macro timing(args...)
    if timer_enabled() in (:parallel, :all)
        TimerOutputs.timer_expr(__module__, false, :($(timer)), args...)
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
#     if timer_enabled() == :all
#         TimerOutputs.timer_expr(__module__, false, :($(timer)), args...)
#     else  # Disable taking timings
#         :($(esc(last(args))))
#     end
# end
