
# Copied from DFTK.jl/src/common/split_evenly.jl

"""
    split_count(N::Integer, n::Integer)
Return a vector of `n` integers which are approximately equally sized and sum to `N`.
"""
function split_count(N::Integer, n::Integer)
    q, r = divrem(N, n)
    return [i <= r ? q+1 : q for i = 1:n]
end

"""
    split_iterator(itr, N)
Split an iterable approximately evenly into N chunks, which will be returned.
"""
function split_iterator(itr, N)
    counts = split_count(length(itr), N)
    map(0:N-1) do i
        rng = (1+sum(counts[1:i])):sum(counts[1:i+1])
        itr[rng]
    end
end

