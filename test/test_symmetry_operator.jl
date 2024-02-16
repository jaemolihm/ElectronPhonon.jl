using Test
using ElectronPhonon
using LinearAlgebra
using ElectronPhonon.WanToBloch: get_symmetry_representation_wannier!

# Test symmetry operator that acts on the electron Wannier functions

@testset "symmetry operator" begin
    BASE_FOLDER = dirname(dirname(pathof(ElectronPhonon)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model(folder, load_symmetry_operators=true)

    # Read symmetry operators from file
    @test model.el_sym.symmetry.nsym == 48
    @test all(model.el_sym.symmetry.is_tr[1:24] .== false)
    @test all(model.el_sym.symmetry.is_tr[25:48] .== true)
    @test length(model.el_sym.operators) == model.el_sym.symmetry.nsym

    for el_velocity_mode in [:Direct, :BerryConnection]
        model.el_velocity_mode = el_velocity_mode

        # Test symmetry of Hamiltonian and eigenstates
        kpts = kpoints_grid((6, 6, 6));
        kpts = GridKpoints(kpts);
        els = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity"]; fourier_mode="gridopt");

        nw = model.nw
        sym_W = zeros(ComplexF64, nw, nw)
        sym_H = zeros(ComplexF64, nw, nw)
        hk = zeros(ComplexF64, nw, nw)
        hsk = zeros(ComplexF64, nw, nw)

        for ik in 1:kpts.n
            xk = kpts.vectors[ik]
            get_fourier!(hk, model.el_ham, xk)
            for (isym, symop) in enumerate(model.el_sym.symmetry)
                sxk = symop.is_tr ? -symop.S * xk : symop.S * xk
                isk = xk_to_ik(sxk, kpts)

                get_symmetry_representation_wannier!(sym_W, model.el_sym.operators[isym], xk, symop.is_tr)
                compute_symmetry_representation!(sym_H, els[ik], els[isk], xk, model.el_sym.operators[isym], symop.is_tr)

                @test norm(sym_W' * sym_W - I(nw)) < 3e-6

                # Test symmetry of Hamiltonian in Wannier basis
                get_fourier!(hsk, model.el_ham, sxk)
                if symop.is_tr
                    @test norm(hsk - sym_W * conj.(hk) * sym_W') < 1e-2
                else
                    @test norm(hsk - sym_W * hk * sym_W') < 1e-2
                end

                # Test symmetry of eigenstates
                @test norm(sym_H' * sym_H - I(nw)) < 3e-6
                e = els[ik].e_full
                # sym_H[i, j] must be zero if the energies of bands i and j are different.
                for j in 1:nw, i in 1:nw
                    if abs(e[i] - e[j]) > 1e-2
                        @test abs(sym_H[i, j]) < 3e-6
                    end
                end

                # Test sym_W * u(k) gives eigenvectors of u(Sk)
                if symop.is_tr
                    usk = sym_W * conj.(els[ik].u_full)
                else
                    usk = sym_W * els[ik].u_full
                end
                @test usk' * hsk * usk ≈ Diagonal(els[isk].e_full) atol=1e-6

                # Test symmetry of velocity matrices
                if symop.is_tr
                    v_rotated = .- Ref(symop.Scart) .* (sym_H * conj.(els[ik].v) * sym_H')
                else
                    v_rotated = Ref(symop.Scart) .* (sym_H * els[ik].v * sym_H')
                end
                if model.el_velocity_mode === :BerryConnection
                    for j in 1:nw, i in 1:nw
                        if abs(e[i] - e[j]) < ElectronPhonon.electron_degen_cutoff
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
