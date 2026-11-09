@testsnippet SymmWS begin
    using Random, LinearAlgebra

    # g[iw, jw, imode] = sum_{R_e, y} exp(2 pi i (k.R_e + q.(y + shift(R_e)))) M[:, :, R_e, :, y]
    function symmws_direct(M, irvec_el, y_file, xk, xq, shift)
        g = zeros(ComplexF64, size(M, 1), size(M, 2), size(M, 4))
        for ig in eachindex(y_file), ir in eachindex(irvec_el)
            phase = cispi(2 * (dot(xk, irvec_el[ir]) + dot(xq, y_file[ig] + shift(irvec_el[ir]))))
            g .+= phase .* M[:, :, ir, :, ig]
        end
        g
    end
    symmws_half(R) = fld.(R, 2)
    symmws_no_shift(R) = zero(R)

    # Sum over a WannierObject: outer R first, then the inner R (irvec_next)
    function symmws_interpolate(ep, nw, nmodes, outer, xk, xq)
        x_outer, x_inner = outer == "ph" ? (xq, xk) : (xk, xq)
        op_r = reshape(ep.op_r, nw, nw, nmodes, length(ep.irvec_next), length(ep.irvec))
        g = zeros(ComplexF64, nw, nw, nmodes)
        for (io, Ro) in enumerate(ep.irvec), (ii, Ri) in enumerate(ep.irvec_next)
            g .+= cispi(2 * (dot(x_outer, Ro) + dot(x_inner, Ri))) .* op_r[:, :, :, ii, io]
        end
        g
    end

    # cubicBN artifact copied into `dir` with a random epmatwp_symmws: crystal.fmt gets ws_method = 3, and
    # wigner.fmt gets unsorted y vectors without ndegen rows. Returns the file contents too.
    function symmws_cubicbn_folder(dir, rng)
        src = _artifact_folder("cubicBN")
        folder = joinpath(dir, "epw")
        cp(src, folder)
        chmod(folder, 0o755; recursive = true)  # artifact files are read-only
        # crystal.fmt ends at L; append lda_plus_u = F and ws_method = 3
        lines = readlines(joinpath(folder, "crystal.fmt"))
        write(joinpath(folder, "crystal.fmt"), join(vcat(lines, [" 0.0", " 0", " F", " 3"]), "\n") * "\n")

        # wigner.fmt: keep the el and ph blocks, replace the el-ph block by unsorted y vectors without ndegen rows
        wl = readlines(joinpath(folder, "wigner.fmt"))
        nr_el, nr_ph, _, nw, ndegen_ph_rows = parse.(Int, split(wl[1]))
        nmodes = parse(Int, split(readlines(joinpath(folder, "epwdata.fmt"))[2])[3])
        irvec_el = Vec3{Int}[]
        pos = 2
        for _ in 1:nr_el
            push!(irvec_el, Vec3(parse.(Int, split(wl[pos])[1:3])...))
            pos += 1 + nw
        end
        for _ in 1:nr_ph
            pos += 1 + ndegen_ph_rows
        end
        y_file = unique(Vec3{Int}[Vec3(rand(rng, -2:2, 3)...) for _ in 1:9])
        nr_g = length(y_file)
        ywl = [string(y[1], " ", y[2], " ", y[3], " 1.0") for y in y_file]
        write(joinpath(folder, "wigner.fmt"),
              join(vcat(["$nr_el $nr_ph $nr_g $nw $ndegen_ph_rows"], wl[2:pos-1], ywl), "\n") * "\n")

        M = randn(rng, ComplexF64, nw, nw, nr_el, nmodes, nr_g)
        write(joinpath(folder, "temp", "cubicBN.epmatwp_symmws"), M)
        (; folder, M, y_file, irvec_el, nw, nmodes)
    end
end

@testitem "epw_parse_structure: ws_method tail of crystal.fmt" setup = [Common] begin
    # crystal.fmt of the cubicBN artifact up to L (line 14), then the tail under test.
    lines = readlines(joinpath(_artifact_folder("cubicBN"), "crystal.fmt"))[1:14]
    function parse_ws(tail)
        mktempdir() do dir
            write(joinpath(dir, "crystal.fmt"), join(vcat(lines, tail), "\n") * "\n")
            (_, _, _, ws_method, ndegen_pre_divided) = ElectronPhonon.epw_parse_structure(dir)
            (; ws_method, ndegen_pre_divided)
        end
    end
    # lda_plus_u, then ws_method alone (EPW use_ws = 'symmetric')
    @test parse_ws([" 0.0", " 0", " F", " 3"]) == (ws_method = 3, ndegen_pre_divided = false)
    # ws_method idiv (EPW v6.2)
    @test parse_ws([" 0.0", " 0", " 2 1"]) == (ws_method = 2, ndegen_pre_divided = true)
    # lda_plus_u = T, ws_method = 1
    @test parse_ws([" 0.0", " 0", " T", " 1"]) == (ws_method = 1, ndegen_pre_divided = false)
    # a lone integer after ngauss, with no lda_plus_u line, is ambiguous
    @test_throws ArgumentError parse_ws([" 0.0", " 0", " 3"])
    # no ws_method line
    @test parse_ws([" 0.0", " 0", " F"]) == (ws_method = -1, ndegen_pre_divided = false)
end

@testitem "read_epmat_symmws_y" setup = [Common, SymmWS] begin
    rng = MersenneTwister(1234)
    nw, nmodes = 2, 3
    # R_e in file order, with odd and negative components
    irvec_el = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(-1, 2, 0), Vec3(3, -1, 1), Vec3(-2, -3, 1)]
    y_file = [Vec3(0, 0, 0), Vec3(1, -1, 0), Vec3(-2, 1, 3), Vec3(2, 2, -1), Vec3(0, 1, 1), Vec3(-1, 0, -2)]
    nr_el, nr_g = length(irvec_el), length(y_file)
    M = randn(rng, ComplexF64, nw, nw, nr_el, nmodes, nr_g)
    ind_el = sortperm(irvec_el, by = reverse)
    irvec_el_sorted = irvec_el[ind_el]

    mktempdir() do dir
        filename = joinpath(dir, "test.epmatwp_symmws")
        write(filename, M)

        for outer in ("el", "ph")
            ep = ElectronPhonon.read_epmat_symmws_y(filename, nw, nmodes, irvec_el_sorted, ind_el, y_file, outer)
            @test issorted(ep.irvec, by = reverse)
            err = 0.0
            err_wrong = 0.0
            for _ in 1:4
                xk, xq = Vec3(rand(rng, 3) .- 0.5), Vec3(rand(rng, 3) .- 0.5)
                g = symmws_interpolate(ep, nw, nmodes, outer, xk, xq)
                err = max(err, norm(g - symmws_direct(M, irvec_el, y_file, xk, xq, symmws_half)) / norm(g))
                err_wrong = max(err_wrong, norm(g - symmws_direct(M, irvec_el, y_file, xk, xq, symmws_no_shift)) / norm(g))
            end
            @test err < 1e-12
            # The reference without the fld(R_e, 2) shift must disagree: the test resolves the shift.
            @test err_wrong > 1e-2
        end

        # A file longer than nw² nr_el nmodes nr_g throws
        write(joinpath(dir, "long"), vcat(vec(M), ComplexF64[1]))
        @test_throws ErrorException ElectronPhonon.read_epmat_symmws_y(joinpath(dir, "long"), nw, nmodes, irvec_el_sorted, ind_el, y_file, "ph")
    end
end

@testitem "load_model_from_epw_new: epmatwp_symmws" setup = [Common, SymmWS] begin
    rng = MersenneTwister(99)
    mktempdir() do dir
        (; folder, M, y_file, irvec_el, nw, nmodes) = symmws_cubicbn_folder(dir, rng)
        @test !issorted(y_file, by = reverse)
        xks = [Vec3(rand(rng, 3) .- 0.5) for _ in 1:3]
        xqs = [Vec3(rand(rng, 3) .- 0.5) for _ in 1:3]
        for outer in ("el", "ph"), on_disk in (false, true)
            tmpdir = mktempdir(dir)
            model = load_model_from_epw_new(folder, "temp", "cubicBN"; epmat_outer_momentum = outer,
                                            epmat_on_disk = on_disk, tmpdir)
            @test model.epmat_outer_momentum == outer
            err = maximum(zip(xks, xqs)) do (xk, xq)
                g = symmws_interpolate(model.epmat, nw, nmodes, outer, xk, xq)
                norm(g - symmws_direct(M, irvec_el, y_file, xk, xq, symmws_half)) / norm(g)
            end
            @test err < 1e-12
        end

        # Negative control: reading with the sorted y order (as the sorted irvec_ep would give)
        # disagrees with the reference built on the file order.
        ind_el = sortperm(irvec_el, by = reverse)
        ep_sorted_y = ElectronPhonon.read_epmat_symmws_y(joinpath(folder, "temp", "cubicBN.epmatwp_symmws"),
                          nw, nmodes, irvec_el[ind_el], ind_el, sort(y_file, by = reverse), "ph")
        xk, xq = xks[1], xqs[1]
        g = symmws_interpolate(ep_sorted_y, nw, nmodes, "ph", xk, xq)
        @test norm(g - symmws_direct(M, irvec_el, y_file, xk, xq, symmws_half)) / norm(g) > 1e-2
    end
end

# Stage 1 (R_e transform, g(k; R_p)) and the full g(k, k+q) of the y storage against the expanded storage.
@testsnippet SymmWSYStorage begin
    using Random, LinearAlgebra

    # Both storages of the cubicBN fixture, loaded with epmat_outer_momentum = "el"
    function symmws_load_both(dir, rng)
        (; folder, M, y_file, irvec_el, nw, nmodes) = symmws_cubicbn_folder(dir, rng)
        model_x = load_model_from_epw_new(folder, "temp", "cubicBN"; epmat_outer_momentum = "el")
        model_y = load_model_from_epw_new(folder, "temp", "cubicBN"; epmat_outer_momentum = "el",
                                          epmat_symmws_storage = "y")
        (; model_x, model_y, M, y_file, irvec_el, nw, nmodes)
    end

    # g(k; R_p) of a WannierObject by the batched engine on `backend`, returned on the host as (ndata, nk)
    function symmws_stage1(epmat, xks, backend)
        obj = ElectronPhonon.to_device(backend, epmat)
        itp = ElectronPhonon.BatchedWannierInterpolator(obj; backend, batch_size = 2)
        out = ElectronPhonon.alloc(backend, ComplexF64, obj.ndata, length(xks))
        ElectronPhonon.get_fourier_batched!(out, itp, xks)
        Array(out)
    end

    # Second step: sum over R_p at q, for each column of g_kR (ndata = nw^2 nmodes nr_p, nk)
    function symmws_stage2(g_kR, irvec_p, xqs)
        mapreduce(+, enumerate(irvec_p)) do (ip, Rp)
            nwm = div(size(g_kR, 1), length(irvec_p))
            hcat([cispi(2 * dot(xq, Rp)) .* g_kR[(ip-1)*nwm+1:ip*nwm, j] for (j, xq) in enumerate(xqs)]...)
        end
    end
end

@testitem "epmat_symmws_storage = y: g(k; R_p) and g(k, k+q) match the expanded storage (CPU)" setup = [Common, SymmWS, SymmWSYStorage] begin
    rng = MersenneTwister(7)
    mktempdir() do dir
        (; model_x, model_y, M, y_file, irvec_el, nw, nmodes) = symmws_load_both(dir, rng)
        ex, ey = model_x.epmat, model_y.epmat
        @test ey.symmws !== nothing && ex.symmws === nothing
        @test ey.irvec_next == ex.irvec_next
        @test ey.ndata == ex.ndata
        @test length(ey.op_r) < length(ex.op_r)   # y storage is smaller than the R_p union
        @test sort(ey.irvec, by = reverse) == ex.irvec

        xks = [Vec3(rand(rng, 3) .- 0.5) for _ in 1:5]
        xqs = [Vec3(rand(rng, 3) .- 0.5) for _ in 1:5]
        gx = symmws_stage1(ex, xks, ElectronPhonon.CPUBackend())
        gy = symmws_stage1(ey, xks, ElectronPhonon.CPUBackend())
        @test norm(gy - gx) / norm(gx) < 1e-12

        # g(k, k+q) of the y storage against the direct (R_e, y) sum
        gkq = symmws_stage2(gy, ey.irvec_next, xqs)
        err = maximum(eachindex(xks)) do j
            ref = symmws_direct(M, irvec_el, y_file, xks[j], xqs[j], symmws_half)
            norm(reshape(gkq[:, j], nw, nw, nmodes) - ref) / norm(ref)
        end
        @test err < 1e-12

        # Negative control: swapping the R_p indices of two groups (a wrong shift) must fail
        lay = ey.symmws
        ng = length(lay.group_start) - 1
        @test ng >= 2
        wrong = ElectronPhonon.SymmWSLayout(lay.nr_p, lay.nr_y, lay.group_start, lay.ip_of_y[:, circshift(1:ng, 1)])
        ey_wrong = ElectronPhonon.WannierObject(ey.irvec, ey.op_r; ey.irvec_next, ey.ndata, symmws = wrong)
        gw = symmws_stage1(ey_wrong, xks, ElectronPhonon.CPUBackend())
        @test norm(gw - gx) / norm(gx) > 1e-2

        # Options: other transforms refuse the y layout; the y storage needs outer "el" and a symmws file
        @test_throws ArgumentError get_interpolator(ey; fourier_mode = "normal")
        @test_throws ArgumentError ElectronPhonon.wannier_object_multiply_R(ey, model_y.lattice)
        @test_throws ArgumentError load_model_from_epw_new(joinpath(dir, "epw"), "temp", "cubicBN";
            epmat_outer_momentum = "ph", epmat_symmws_storage = "y")
        @test_throws ArgumentError load_model_from_epw_new(joinpath(dir, "epw"), "temp", "cubicBN";
            epmat_symmws_storage = "bad")
    end
end

@testitem "epmat_symmws_storage = y: g(k; R_p) on the GPU matches the CPU expanded storage" tags = [:gpu] setup = [Common, SymmWS, SymmWSYStorage] begin
    CUDA_OK = try
        @eval using CUDA
        CUDA.functional()
    catch
        false
    end
    if CUDA_OK
        rng = MersenneTwister(8)
        mktempdir() do dir
            (; model_x, model_y) = symmws_load_both(dir, rng)
            xks = [Vec3(rand(rng, 3) .- 0.5) for _ in 1:5]
            gx = symmws_stage1(model_x.epmat, xks, ElectronPhonon.CPUBackend())
            gy = symmws_stage1(model_y.epmat, xks, ElectronPhonon.gpu_backend())
            @test norm(gy - gx) / norm(gx) < 1e-12
        end
    else
        @info "CUDA not functional, skipping the GPU arm"
    end
end

# The engine path (OuterKEngine: device epmat, stage 1, stage 2) with the y storage against the expanded
# storage of the same random epmat, on the same backend (g2 of random-coupling cubicBN is gauge-dependent
# across backends in degenerate multiplets, so a CPU/GPU comparison would not isolate the storage).
@testitem "epmat_symmws_storage = y: run_eph_over_k_and_kq g2 matches the expanded storage" setup = [Common, SymmWS, SymmWSYStorage, EphReferenceLoop] begin
    CUDA_OK = try
        @eval using CUDA
        CUDA.functional()
    catch
        false
    end
    CUDA_OK && CUDA.allowscalar(false)
    mktempdir() do dir
        (; model_x, model_y) = symmws_load_both(dir, MersenneTwister(5))
        function record_g2(model, backend)
            rec = _PairRecorder()
            ElectronPhonon.run_eph_over_k_and_kq(model, (3, 3, 3), (3, 3, 3); calculators = [rec], symmetry = nothing,
                verbosity = 0, progress_print_step = 10^9, window_k = (-Inf, Inf), window_kq = (-Inf, Inf),
                n_outer_batch = 5, n_inner_tile = 40, backend)
            rec.g2abs
        end
        for backend in (CUDA_OK ? Any[ElectronPhonon.CPUBackend(), ElectronPhonon.gpu_backend()] : Any[ElectronPhonon.CPUBackend()])
            gx = record_g2(model_x, backend)
            gy = record_g2(model_y, backend)
            @test !isempty(gx) && keys(gy) == keys(gx)
            @test maximum(norm(gy[k] - gx[k]) / norm(gx[k]) for k in keys(gx)) < 1e-12
        end
    end
end
