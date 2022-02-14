using Test
using EPW

@testset "unfold" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    model = load_model(folder)

    # Unfolding of irreducible BZ to full BZ
    nk = 30
    kpts = GridKpoints(bzmesh_ir_wedge((nk, nk, nk), model.symmetry));
    kpts_unfold_ref = GridKpoints(generate_kvec_grid(nk, nk, nk));
    kpts_unfold = EPW.unfold_kpoints(kpts, model.symmetry);
    @test kpts_unfold == kpts_unfold_ref

    # Unfolding of irreducible BZ inside given energy window to full BZ
    nk = 53
    window_k = (0.7, 0.8)
    kpts = GridKpoints(filter_kpoints((nk, nk, nk), model.nw, model.el_ham, window_k, nothing, symmetry=model.symmetry)[1])
    kpts_unfold_ref = GridKpoints(filter_kpoints((nk, nk, nk), model.nw, model.el_ham, window_k, nothing, symmetry=nothing)[1])
    kpts_unfold = EPW.unfold_kpoints(kpts, model.symmetry);
    @test kpts_unfold == kpts_unfold_ref
end
