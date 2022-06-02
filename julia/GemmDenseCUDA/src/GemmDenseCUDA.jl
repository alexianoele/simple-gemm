module GemmDenseCUDA

import CUDA

function gemm!(A, B, C, A_rows, A_cols, B_cols)

    row = CUDA.blockIdx().x * CUDA.blockDim().x + CUDA.threadIdx().x
    col = CUDA.blockIdx().y * CUDA.blockDim().y + CUDA.threadIdx().y

    if row <= A_rows && col <= B_cols
        tmp = 0.0
        for k = 1:A_cols
            tmp += A[row, k] * B[k, col]
        end
        C[row, col] = tmp
    end
    return
end


function main(args::Array{String,1})::Int32

    # must initialize scalars
    A_rows::Int32 = -1
    A_cols::Int32 = -1
    B_rows::Int32 = -1
    B_cols::Int32 = -1

    @show args

    # args don't include Julia executable and program
    if size(args)[1] != 3
        throw(
            ArgumentError(
                "Usage: 3 arguments: matrix A rows, matrix A cols and matrix B cols",
            ),
        )
    else
        A_rows = parse(Int32, args[1])
        A_cols = parse(Int32, args[2])
        B_rows = parse(Int32, args[2])
        B_cols = parse(Int32, args[3])
    end

    # Julia is column-based (like Fortran)
    A = CUDA.CuArray{Float32,2}(undef, A_rows, A_cols)
    B = CUDA.CuArray{Float32,2}(undef, B_rows, B_cols)
    C = CUDA.zeros(Float32, A_rows, B_cols)

    CUDA.rand!(A)
    CUDA.rand!(B)

    CUDA.@cuda threads = 1024 gemm!(A, B, C, A_rows, A_cols, B_cols)

    CUDA.synchronize()

    return 0

end


end
