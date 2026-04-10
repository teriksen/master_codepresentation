using Plots
using BenchmarkTools
using Random
using GZip
using DelimitedFiles
using DataFrames
using Arrow
using Dates
using Statistics
using Base.Threads
using StaticArrays


function main()
    a = ones(4,4)
    for i in 1:4, j in 1:4
        a[i,j] = i + (j-1)*4
    end

    b = @view a[2:3, 2:3]
    c = a[2:3, 2:3]

    b .= 0

    for i in 1:4
        println(a[i,:])
    end
    for i in 1:2
        println(c[i,:])
    end
    println("b:")
    for i in 1:2
        println(b[i,:])
    end
    return
end

main()