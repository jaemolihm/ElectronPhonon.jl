# A @testmodule, not a @testsnippet: evaluated once per process, so the harness and the
# ElectronPhonon methods specialised on its types compile once instead of in every test item.
# It reuses the reference loop of `EphReferenceLoop` (`using ..EphReferenceLoop`, so the harness's
# `isdefined` guard skips its own include), hence an item lists `EphReferenceLoop` before this module.
@testmodule CalculatorContractHarness begin
    using Test, LinearAlgebra, ElectronPhonon
    using ..EphReferenceLoop

    include("calculator_contract_harness.jl")

    export check_calculator_contract, contract_fixtures, contract_foreach_reference_pair,
        contract_group_sum, contract_multiplet_ids, contract_pair_sum, contract_physical_ω,
        contract_reference, contract_settings, pair_ω, run_contract
end
