using ElectronPhonon
using Test
using Random

@testset "wannier interpolation" begin
    using ElectronPhonon: WannierObject

    irvec = [Vec3{Int}([0, 0, 0]), Vec3{Int}([1, -2, 1]),
        Vec3{Int}([1, 3, -2]), Vec3{Int}([-1, 3, -2]),
        Vec3{Int}([-1, -3, 1])]
    nr = length(irvec)
    Random.seed!(123)
    op_r = randn(ComplexF64, 6, nr)

    # Constructor should throw error if irvec is not sorted
    @test_throws ErrorException obj = WannierObject(irvec, op_r)

    ind = sortperm(irvec, by=x->reverse(x))
    irvec = irvec[ind]
    op_r = op_r[:, ind]

    obj = WannierObject(irvec, op_r)

    xk1 = Vec3([0.3, -0.4, 0.7])
    xk2 = Vec3([0.3, -0.4, -0.5])
    op_k_1d = Array{ComplexF64,1}(undef, 6)
    op_k_normal = Array{ComplexF64,2}(undef, 2, 3)
    op_k_normal_2 = Array{ComplexF64,2}(undef, 2, 3)
    op_k_gridopt = Array{ComplexF64,2}(undef, 2, 3)

    obj_normal = get_interpolator(obj, fourier_mode="normal")
    obj_gridopt = get_interpolator(obj, fourier_mode="gridopt")
    get_fourier!(op_k_normal, obj_normal, xk1)
    get_fourier!(op_k_gridopt, obj_gridopt, xk1)
    @test op_k_normal ≈ op_k_gridopt
    @test obj_gridopt.gridopt.nr_23 == 4
    @test obj_gridopt.gridopt.nr_3 == 3

    get_fourier!(op_k_1d, obj_gridopt, xk1)
    @test vec(op_k_normal) ≈ op_k_1d

    get_fourier!(op_k_normal, obj_normal, xk2)
    get_fourier!(op_k_gridopt, obj_gridopt, xk2)
    @test op_k_normal ≈ op_k_gridopt

    # Update op_r
    op_r_new = randn(ComplexF64, 6, nr)
    obj_new = WannierObject(irvec, op_r_new)
    update_op_r!(obj, op_r_new)
    obj_new_normal = get_interpolator(obj_new, fourier_mode="normal")

    get_fourier!(op_k_normal, obj_normal, xk1)
    get_fourier!(op_k_gridopt, obj_gridopt, xk1)
    get_fourier!(op_k_normal_2, obj_new_normal, xk1)
    @test op_k_normal ≈ op_k_gridopt
    @test op_k_normal ≈ op_k_normal_2
end

@testset "wannier interpolation - batched" begin
    using ElectronPhonon: WannierObject, register_kpoints!, clear_registered_kpoints!

    # Create test WannierObject
    irvec = [Vec3{Int}([0, 0, 0]), Vec3{Int}([1, -2, 1]),
        Vec3{Int}([1, 3, -2]), Vec3{Int}([-1, 3, -2]),
        Vec3{Int}([-1, -3, 1])]
    nr = length(irvec)
    Random.seed!(456)
    op_r = randn(ComplexF64, 6, nr)

    ind = sortperm(irvec, by=x->reverse(x))
    irvec = irvec[ind]
    op_r = op_r[:, ind]

    obj = WannierObject(irvec, op_r)

    # Create interpolator for comparison
    itp_normal = get_interpolator(obj, fourier_mode="normal")

    # Test both batched and batched-gridopt interpolators
    for mode in ["batched", "batched-gridopt"]
        # Create batched interpolator
        itp_batched = get_interpolator(obj, fourier_mode=mode, batch_size=3)

        # Test 1: Error when no k-points registered
        @test_throws ErrorException get_fourier!(zeros(ComplexF64, 6), itp_batched, Vec3([0.1, 0.2, 0.3]))

        # Test 2: Register k-points and sequential access
        kpoints = [Vec3(rand(3)) for _ in 1:12]
        sort!(kpoints)
        register_kpoints!(itp_batched, kpoints)

        op_k_compare = zeros(ComplexF64, 6)
        op_k_batched = zeros(ComplexF64, 6)

        # Compare results with reference interpolator for each k-point
        for xk in kpoints
            get_fourier!(op_k_compare, itp_normal, xk)
            get_fourier!(op_k_batched, itp_batched, xk)
            @test op_k_compare ≈ op_k_batched
        end

        # Test 3: Error when all k-points exhausted
        @test_throws ErrorException get_fourier!(op_k_batched, itp_batched, kpoints[1])

        # Test 4: Error when k-point doesn't match
        register_kpoints!(itp_batched, kpoints)
        @test_throws ErrorException get_fourier!(op_k_batched, itp_batched, Vec3([0.9, 0.9, 0.9]))

        # Test 5: Test clear_registered_kpoints! and re-registration
        clear_registered_kpoints!(itp_batched)
        @test_throws ErrorException get_fourier!(op_k_batched, itp_batched, kpoints[1])

        new_kpoints = [Vec3(rand(3)) for _ in 1:3]
        sort!(new_kpoints)
        register_kpoints!(itp_batched, new_kpoints)

        for xk in new_kpoints
            get_fourier!(op_k_compare, itp_normal, xk)
            get_fourier!(op_k_batched, itp_batched, xk)
            @test op_k_compare ≈ op_k_batched
        end

        # Test 6: Test with different batch sizes
        for batch_size in [1, 2, 4, 8]
            itp_test = get_interpolator(obj, fourier_mode=mode, batch_size=batch_size)
            test_kpoints = [Vec3(rand(3)) for _ in 1:7]
            sort!(test_kpoints)
            register_kpoints!(itp_test, test_kpoints)

            for xk in test_kpoints
                get_fourier!(op_k_compare, itp_normal, xk)
                get_fourier!(op_k_batched, itp_test, xk)
                @test op_k_compare ≈ op_k_batched
            end
        end

        # Test 7: Test with 2D array output
        op_k_2d_compare = Array{ComplexF64,2}(undef, 2, 3)
        op_k_2d_batched = Array{ComplexF64,2}(undef, 2, 3)

        register_kpoints!(itp_batched, kpoints)
        for xk in kpoints
            get_fourier!(op_k_2d_compare, itp_normal, xk)
            get_fourier!(op_k_2d_batched, itp_batched, xk)
            @test op_k_2d_compare ≈ op_k_2d_batched
        end
    end
end

@testset "batched modes reject a DiskWannierObject" begin
    # A disk-backed object is supported by the per-k modes only; the batched modes need an
    # in-memory op_r. The rejection is at `get_interpolator`, and its message names the two modes
    # that do work. No file is read here — construction only records the path.
    using ElectronPhonon: DiskWannierObject
    irvec = [Vec3{Int}([0, 0, 0]), Vec3{Int}([1, 0, 0])]
    obj = DiskWannierObject(Float64, "tag", length(irvec), irvec, 4, mktempdir(), "nonexistent.bin")

    for mode in ("batched", "batched-gridopt")
        err = try get_interpolator(obj; fourier_mode = mode); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("in-memory op_r", err.msg)
        @test occursin("\"normal\" or \"gridopt\"", err.msg)
    end

    # The per-k modes still accept it.
    @test get_interpolator(obj; fourier_mode = "normal") isa ElectronPhonon.NormalWannierInterpolator
    @test get_interpolator(obj; fourier_mode = "gridopt") isa ElectronPhonon.GridoptWannierInterpolator
end

@testset "backend-dependent batch_size default" begin
    using ElectronPhonon: _default_batch_size, CPUBackend
    irvec = [Vec3{Int}([0, 0, 0]), Vec3{Int}([1, 0, 0])]
    obj = WannierObject(irvec, randn(ComplexF64, 6, length(irvec)))

    # The CPU side does not move: the per-k query sites still get exactly 32, whatever the object,
    # the grid size or the number of channel buffers.
    @test _default_batch_size(CPUBackend(), 617, 9) == 32
    @test _default_batch_size(CPUBackend(), 617, 9; nk_hint = 4, nbuffers = 8) == 32
    @test get_interpolator(obj; fourier_mode = "batched").batch_size == 32
    @test get_interpolator(obj; fourier_mode = "batched-gridopt").batch_size == 32

    # An explicit `batch_size` still wins, and `get_interpolator_channel` forwards it (it used to
    # drop it silently, so every channel-built batched interpolator got the hardcoded default).
    @test get_interpolator(obj; fourier_mode = "batched", batch_size = 7).batch_size == 7
    ch = get_interpolator_channel(obj; fourier_mode = "batched", batch_size = 7, nbuffers = 2)
    @test take!(ch).batch_size == 7

    # `nk_hint` never widens past the budget, and on the CPU it cannot narrow below the fixed 32.
    @test get_interpolator(obj; fourier_mode = "batched", nk_hint = 10^9).batch_size == 32
end
