using Test
using EPW

@testset "check model symmetry" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    model = load_model(folder, load_symmetry_operators=true)
    model.el_velocity_mode = :Direct

    res = EPW.check_electron_symmetry_of_model(model, (5, 5, 5))
    @test model.el_velocity_mode == :Direct # Check model.el_velocity_mode is not changed

    for data in (res.rms_errors, res.max_errors)
        @test Set(keys(data)) == Set([:Energy, :SymMatrixUnitarity, :Hamiltonian, :Velocity_Direct,
                                      :Velocity_BerryConnection])
        @test all(values(data) .> 0)
        for key in keys(data)
            if key === :Velocity_BerryConnection
                # Error of Velocity_BerryConnection is large because of finite-difference error
                @test data[key] < 3.0
            else
                @test data[key] < 1e-2
            end
        end
    end
end
