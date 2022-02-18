using Test
using EPW
using LinearAlgebra

# Test symmetry operator that acts on the electron Wannier functions

@testset "symmetry operator" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model(folder, load_symmetry_operators=true)

    # Read symmetry operators from file
    @test model.el_sym.symmetry.nsym == 24
    @test all(model.el_sym.symmetry.is_tr .== false)
    @test length(model.el_sym.operators) == model.el_sym.symmetry.nsym

    for el_velocity_mode in [:Direct, :BerryConnection]
        model.el_velocity_mode = el_velocity_mode

        # Test symmetry of Hamiltonian and eigenstates
        kpts = generate_kvec_grid(6, 6, 6);
        kpts = GridKpoints(kpts);
        els = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity"]; fourier_mode="gridopt");

        nw = model.nw
        sym_k = zeros(ComplexF64, nw, nw)
        hk = zeros(ComplexF64, nw, nw)
        hsk = zeros(ComplexF64, nw, nw)

        for ik in 1:kpts.n
            xk = kpts.vectors[ik]
            get_fourier!(hk, model.el_ham, xk)
            for (isym, symop) in enumerate(model.el_sym.symmetry)
                sxk = symop.is_tr ? -symop.S * xk : symop.S * xk
                isk = xk_to_ik(sxk, kpts)
                get_fourier!(sym_k, model.el_sym.operators[isym], xk)
                @test norm(sym_k' * sym_k - I(nw)) < 3e-6

                # Test symmetry of Hamiltonian in Wannier basis
                get_fourier!(hsk, model.el_ham, sxk)
                @test norm(hsk - sym_k * hk * sym_k') < 1e-2

                # Test symmetry of eigenstates
                sym_H = els[isk].u_full' * sym_k * els[ik].u_full
                @test norm(sym_H' * sym_H - I(nw)) < 3e-6
                e = els[ik].e_full
                # sym_H[i, j] must be zero if the energies of bands i and j are different.
                for j in 1:nw, i in 1:nw
                    if abs(e[i] - e[j]) > 1e-2
                        @test abs(sym_H[i, j]) < 3e-6
                    end
                end

                # Test symmetry of velocity matrices
                v_rotated = Ref(symop.Scart) .* (sym_H * els[ik].v * sym_H')
                if symop.is_tr
                    v_rotated .*= -1
                end
                if model.el_velocity_mode === :BerryConnection
                    for j in 1:nw, i in 1:nw
                        if abs(e[i] - e[j]) < EPW.electron_degen_cutoff
                            @test els[isk].v[i, j] ≈ v_rotated[i, j] atol=1e-2
                        end
                    end
                    # For the BerryConnection case, the error in the off-diagonal element
                    # between non-degenerate states is large. So we use a loose tolerance.
                    @test all(norm.(els[isk].v - v_rotated) .< 0.8)
                else
                    @test els[isk].v ≈ v_rotated atol=2e-5
                end
            end
        end
    end
end
