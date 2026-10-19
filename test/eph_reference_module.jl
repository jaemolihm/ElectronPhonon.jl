# A @testmodule, not a @testsnippet: evaluated once per process, so the recorder types and the
# ElectronPhonon methods specialised on them compile once instead of in every test item.
@testmodule EphReferenceLoop begin
    using Test, LinearAlgebra, ElectronPhonon

    include("eph_reference_loop.jl")

    export _DgRecorder, _PairRecorder, _pair_key, compare_with_reference, contract_multiplets,
        eph_reference, eph_reference_dg, eph_reference_k_and_q
end
