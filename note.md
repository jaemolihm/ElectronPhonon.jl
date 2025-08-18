# Note on implementation details

### Interpolation package
I am using `Interpolations`.

Use `BSpline(Linear())` instead of `Gridded(Linear())`. The former is ~2 times faster than the latter. An in-place version looks like `extrapolate(scale(interpolate!(data, BSpline(Linear())), rng), Flat())`. (Aug 18, 2025)

I have checked `DataInterpolations`. I did not use it because for 1d interpolation on (uniformly spaced) ranges, `Interpolations.jl` is ~3 times faster than `DataInterpolations`. The reason is that `DataInterpolations` do not specialize using the fact that ranges are uniformlt spaced. See https://github.com/SciML/DataInterpolations.jl/issues/43. (Aug 18, 2025)
