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
using LinearAlgebra

# Arguments:
# array_dim: the dimension of the rotor array (array_dim x array_dim)
# angvel_scale: the scale of the angular velocities; small values lead to more synchronized behavior, larger values lead to more chaotic behavior
# time_steps: the number of time steps to simulate
# dt: the time step size for the RK4 integrator
# save_rate: for animation and analysis; e.g. save_rate=1000 means that the phases will be saved every 1000 time steps
# animate: creates a gif animation of the rotor phases over time if true
# phase_type: "random" gives random initial phases, "preset" uses the preset_phases argument, "presynced" gives all rotors the same initial phase
# init_angle: the initial phase angle for the "presynced" phase type, 0.9 is a randomly chosen value
# angvel_type: "random" chooses random angular velocities from a uniform distribution, "preset" uses the preset_angvels argument, "perturbed" gives angular velocities in the range [angvel_scale-epsilon, angvel_scale+epsilon]
# epsilon: the perturbation range for the "perturbed" angvel_type
# path: the path where the animation and data will be saved
# preset_angvels: a 2D array of size array_dim x array_dim of Float64 values, only used if angvel_type is "preset"
# preset_phases: a 2D array of size array_dim x array_dim of Float64 values, only used if phase_type is "preset". The additional boundary points are for periodic boundary conditions.
# filename: the name of the gif file to save the animation to, only used if animate is true
# benchmark: For benchmarking purposes; if true, will run the RK4 integrator multiple times
function kuramoto_model(array_dim; 
    angvel_scale=0.5,
    time_steps=1000000, dt=0.001, save_rate=1000,
    animate=true, phase_type="random", init_angle=0.9,
    angvel_type="random", epsilon=0.1,
    path = "data/", preset_angvels=nothing,
    preset_phases=nothing,
    filename="kuramoto_model.gif",
    small_angle=false, benchmark=false
    )
    
    # we define phases as a 2D array of size dim_x+2  x  dim_y+2 to simplify
    # the calculation of the sine sums, exchanging some memory usage for speed
    N = array_dim + 2
    
    if preset_phases === nothing
        preset_phases = zeros(Float64, array_dim, array_dim)
    end
    if phase_type == "presynced"
        println("Pre-synchronized initial phases")
        init_phases = ones(N,N) .* init_angle # gives pre-synchronized initial phase angles
    elseif phase_type == "random"
        init_phases = 2pi .* rand(N, N) # gives initial phase angles from 0 to 2pi
    elseif phase_type == "preset"
        init_phases = preset_phases
    else
        error("Invalid initial angle type. Choose 'random', 'preset' or 'presynced'.")
    end

    # periodic boundary conditions, corners are set to zero
    init_phases[1,1] = 0
    init_phases[1,end] = 0
    init_phases[end,1] = 0
    init_phases[end,end] = 0
    init_phases[1,2:end-1] = init_phases[end-1,2:end-1]
    init_phases[end,2:end-1] = init_phases[2,2:end-1]
    init_phases[2:end-1,1] = init_phases[2:end-1,end-1]
    init_phases[2:end-1,end] = init_phases[2:end-1,2]
    
    innate_angvels = zeros(Float64, array_dim, array_dim)
    if preset_angvels === nothing
        # Defines preset_angvels to maintain type stability
        preset_angvels = zeros(Float64, array_dim, array_dim)
    end
    if angvel_type == "random"
        # Sets individual angular velocities from a uniform distribution for each rotor
        innate_angvels .= angvel_scale .* rand(array_dim,array_dim)
    elseif angvel_type == "preset"
        if size(preset_angvels, 1) != array_dim || size(preset_angvels, 2) != array_dim
            error("Preset angular velocities must be of size $(array_dim) x $(array_dim).")
        elseif preset_angvels == zeros(Float64, array_dim, array_dim)
            @warn "Preset angular velocities are all zeros."
        end
        innate_angvels .= preset_angvels
    elseif angvel_type == "perturbed"
        # Perturbs the initial angular velocities: w in [angvel_scale-epsilon, angvel_scale+epsilon]
        innate_angvels .= epsilon .* 2 .*(rand(array_dim,array_dim) .- 0.5) .+ angvel_scale
    else
        error("Invalid angular velocity type. Choose 'random', 'preset' or 'perturbed'.")
    end

    # Preallocate the array used for the sine sums
    sine_sums = zeros(Float64, N, N)
    # Calculates the time derivative of the phases and applies periodic boundary conditions
    # Mutates in place to avoid unnecessary allocations, for performance
    function theta_dot!(out, subout,  
        t_idx, phases)
        sine_sums .= 0      # Reset the sine sums in place
        @inbounds for i in 2:N-1, j in 2:N-1 # Loop over "real" rotor positions
            s_u = sin(phases[i-1,j] - phases[i,j])
            s_l = sin(phases[i, j-1] - phases[i,j])
            sine_sums[i,   j]   = s_u + s_l
            sine_sums[i-1, j]  -= s_u
            sine_sums[i,   j-1] -= s_l
        end
        # After extensive testing, running a nested for loop to calculate the sine sums was faster
        # than using array operations by a small margin.
        #subout = @view out[2:end-1, 2:end-1]
        subout .= innate_angvels
        axpy!(1.0, view(sine_sums, 2:N-1, 2:N-1), subout)
        # Periodic boundary conditions
        #@inbounds out[1,2:end-1]   .= out[end-1,2:end-1]
        #@inbounds out[end,2:end-1] .= out[2,2:end-1]
        #@inbounds out[2:end-1,1]   .= out[2:end-1,end-1]
        #@inbounds out[2:end-1,end] .= out[2:end-1,2]
        @inbounds for k in 2:N-1
            out[1, k]   = out[end-1, k]
            out[end, k] = out[2, k]
            out[k, 1]   = out[k, end-1]
            out[k, end] = out[k, 2]
        end
    end
    function theta_dot_old!(out, subout, t_idx, phases)
        sine_sums .= 0 # Reset the sine sums in place
        for j in 2:N-1 # Loop over "real" rotor positions
            for i in 2:N-1
                s_u = sin(phases[i-1,j] - phases[i,j])
                s_l = sin(phases[i,j-1] - phases[i,j])
                sine_sums[i  ,   j] = s_u + s_l
                sine_sums[i-1,   j] -= s_u
                sine_sums[i  , j-1] -= s_l
            end
        end
        # After extensive testing, running a nested for loop to calculate the sine sums was faster
        # than using array operations by a small margin.
        out[2:end-1, 2:end-1] .= innate_angvels .+ sine_sums[2:end-1,2:end-1]
        # Periodic boundary conditions
        out[1,2:end-1]   .= out[end-1,2:end-1]
        out[end,2:end-1] .= out[2,2:end-1]
        out[2:end-1,1]   .= out[2:end-1,end-1]
        out[2:end-1,end] .= out[2:end-1,2]
    end

    function theta_dot_small_angle!(out, subout, t_idx, phases)
        sine_sums .= 0 # Reset the sine sums in place
        # Using column major order for closer memory access
        @inbounds for j in 2:N-1, i in 2:N-1 # Loop over "real" rotor positions
            s_u = phases[i-1,j] - phases[i,j]
            s_l = phases[i, j-1] - phases[i,j]
            sine_sums[i,   j]   = s_u + s_l
            sine_sums[i-1, j]  -= s_u
            sine_sums[i,   j-1] -= s_l
        end
        # After extensive testing, running a nested for loop to calculate the sine sums
        # was faster than using array operations by a small margin.
        #subout = @view out[2:end-1, 2:end-1]
        subout .= innate_angvels
        @inbounds for j in 2:N-1, i in 2:N-1
            subout[i-1, j-1] += sine_sums[i, j]
        end
        #@inbounds out[2:end-1, 2:end-1] .= innate_angvels .+ sine_sums[2:end-1,2:end-1]
        # Periodic boundary conditions
        @inbounds for k in 2:N-1
            out[1, k]   = out[end-1, k]
            out[end, k] = out[2, k]
            out[k, 1]   = out[k, end-1]
            out[k, end] = out[k, 2]
        end
    end

    function rk4(f, phases_0, time_steps, step_size)
        t_idx = 1
        phases = phases_0
        # Define the history array to save the phases at each time step
        hist_pbc = zeros(Float64, ceil(Int32, time_steps/save_rate)+1, N, N)
        hist_pbc[1, :, :] .= phases  # Save the initial phases
        step_size_half  = step_size / 2
        step_size_sixth = step_size / 6
        
        # Allocate the k arrays, their "real rotor" views, and temp array
        k1 = zeros(Float64, N, N)
        k1_sub = @view k1[2:end-1, 2:end-1]
        k2 = zeros(Float64, N, N)
        k2_sub = @view k2[2:end-1, 2:end-1]
        k3 = zeros(Float64, N, N)
        k3_sub = @view k3[2:end-1, 2:end-1]
        k4 = zeros(Float64, N, N)
        k4_sub = @view k4[2:end-1, 2:end-1]
        temp = zeros(Float64, N, N)

        for step in 1:time_steps
            # f() (theta_dot!) mutates the k arrays in place, reducing allocations.
            f(k1, k1_sub, t_idx, phases)
            temp .= phases; axpy!(step_size_half, k1, temp)
            f(k2, k2_sub, t_idx, temp)
            temp .= phases; axpy!(step_size_half, k2, temp)
            f(k3, k3_sub, t_idx, temp)
            temp .= phases; axpy!(step_size, k3, temp)
            f(k4, k4_sub, t_idx, temp)
            
            # 300 allocations over the entire simulation
            if step % save_rate == 0
                hist_pbc[ ( step ÷ save_rate ) + 1, :, :] .= phases
            end
            # update the phases
            #@. phases = phases + step_size / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            temp .= k1
            axpy!(2, k2, temp)
            axpy!(2, k3, temp)
            axpy!(1, k4, temp)
            axpy!(step_size_sixth, temp, phases)

            t_idx += 1
        end
        return hist_pbc
    end

    function plot_history(rotors_hist, time_steps, save_rate, filename)
        println("Creating animation...")
        @time anim = @animate for i in 1:ceil(Int64, time_steps/save_rate)
            heatmap((rotors_hist[i, :, :]),   # Data
                c=:hsv,                     # Colormap
                aspect_ratio=1,             # Aspect ratio
                colorbar=true,              # Show colorbar
                title="Kuramoto Model, t=$(round(i*dt*save_rate))",     # Title
                clims=(0, 2pi))
        end
        Plots.gif(anim, filename) # Save the animation as a gif
        println("Animation saved to $filename.")
    end



    # Integrator block
    if benchmark
        println("Benchmarking RK4 integrator with old f()...")
        @btime $rk4($theta_dot_old!, $init_phases, $time_steps, $dt)
        println("")
        println("Benchmarking RK4 integrator with @inbounds in f()...")
        @btime $rk4($theta_dot!, $init_phases, $time_steps, $dt)
        println("")
        println("Benchmarking RK4 integrator with @inbounds and small angle approximation in f()...")
        @btime $rk4($theta_dot_small_angle!, $init_phases, $time_steps, $dt)
        println("")
        println("Benchmarking complete.")
        return
    else
        println("---------------------------------------")
        println("--------Running RK4 integrator:--------")
        if small_angle
            hist_pbc = @time rk4(theta_dot_small_angle!, init_phases, time_steps, dt)
        else
            hist_pbc= @time rk4(theta_dot!, init_phases, time_steps, dt)
        end
        println("--------Finished RK4 integrator--------")
        println("---------------------------------------")
    end
    
    # Get rid of the extra boundary points, and cast the phases to the interval [0, 2pi]
    rotors_hist = hist_pbc[:,2:end-1,2:end-1] .% 2pi
    rotors_hist[rotors_hist .< 0] .+= 2pi
    
    if animate
        plot_history(rotors_hist, time_steps, save_rate, filename)
    end
    return rotors_hist, innate_angvels, init_phases
end

### - MAIN KURAMOTO MODEL FUNCTION ENDS - ###
### ------------------------------------- ###

# Defines a number of bins to cast the continuous phase values into, and encodes the 
# binned values as integers for more compact storage. This is used for approximating
# the Shannon entropy of the system, and used in the GZip compression of the rotor history.
function cast_and_encode(rotors_hist; bits=8)
    rotors_hist_binned = floor.(Int16, 2^bits .* rotors_hist ./ (2pi))
    # ensure that no entries are less than 0 (modulo adjustment)
    @. rotors_hist_binned[rotors_hist_binned < 0] += 2^bits
    
    # ensure that no entries exceed the bin size
    if any(rotors_hist_binned .>= 2^bits)
        println("[cast_and_encode] Warning: Some entries are greater than the bin size.")
    end
    if any(rotors_hist_binned .< 0)
        println("[cast_and_encode] Warning: Some entries are less than 0.")
    end

    return rotors_hist_binned
end

# Calculates the order parameters r and the mean phase theta for
# each time step, given the history of the rotor phases.
# r*exp(im*theta_bar) = (1/N*M) * sum( exp(im*theta_j) )_{j=1}^{N*M}
function get_order_params(rotors_hist)
    N = size(rotors_hist, 2)
    M = size(rotors_hist, 3)
    t_series = size(rotors_hist, 1)
    order_param_series = zeros(Float64, t_series, 2)
    for i in 1:t_series
        order_param_series[i,1] = abs( sum( exp.( im .* rotors_hist[i, :, :] ) ) ) / (N*M) # r
        order_param_series[i,2] = mean( rotors_hist[i, :, :] ) # mean of the phases
    end
    return order_param_series
end

# Old plotting function. New plotting is done in python.
function plot_order_params(order_param_series, filename, plotname; path="data/")
    Plots.plot(order_param_series, 
        xlabel="Time", 
        ylabel=plotname, 
        title="Time evolution of $plotname"
        )
    Plots.savefig(path * filename)
end

# Calculates the standard deviation of the order parameters in a given time window,
# which can be used as a measure of the fluctuations in the system.
function calculate_standard_deviation_of_order_parameters(order_params; start_index=1, end_indexoffset=0)
    rs = order_params[:, 1]
    thetas = order_params[:, 2]
    t_idx_len = size(order_params, 1)
    end_index = t_idx_len - end_indexoffset
    @assert start_index <= end_index "Start index must be less than or equal to end index."
    @assert start_index >= 1 && end_index <= t_idx_len "Start and end indices must be within the bounds of the data."
    var_r, var_theta = 0, 0
    r_mean = mean( rs[start_index:end_index] )
    theta_mean = mean( thetas[start_index:end_index] )
    for i in start_index:end_index
        var_r += (rs[i] - r_mean)^2
        var_theta += (thetas[i] - theta_mean)^2
    end
    var_r /= end_index - start_index + 1
    var_theta /= end_index - start_index + 1

    std_r = sqrt(var_r)
    std_theta = sqrt(var_theta)

    return std_r, std_theta
end

# This function runs a single simulation of the Kuramoto model with the specified parameters,
# and saves the rotor history, innate angular velocities, and initial phases to feather files for later analysis.
function single_sol(rotor_array_dim, angvel_scale, integrator, 
    t_steps, dt, n_saved_points; path="animations/", pre_synced=false, 
    angvel_type="random", animate=true, save_fig=true)
    save_rate = t_steps ÷ n_saved_points
    println("================Running Kuramoto Model Simulation================")
    println("Running Kuramoto model with $rotor_array_dim x $rotor_array_dim rotors:")
    rotors_hist, innate_angvels, init_phases = @time kuramoto_model(rotor_array_dim,
        angvel_scale = angvel_scale,
        time_steps = t_steps,
        dt = dt,
        save_rate = save_rate,
        animate = animate,
        phase_type = "random",
        filename = path * "kuramoto_$(rotor_array_dim)_t$(t_steps)_angvels$(angvel_scale)_$(integrator).gif",
        path = path,
        angvel_type = angvel_type,
        )
    println("=============================Finished=============================")
    return rotors_hist, innate_angvels, init_phases
end

# Makes a histogram of the binned rotor phases for each time step.
# Returns a #t_saved x 2^bits, where column t_saved ([t_saved, :]) gives the number of rotors
# in each bin, at the t_saved time step.
function get_binned_histogram(rotors_hist_binned; bits=8)
    number_of_steps = size(rotors_hist_binned, 1)
    number_of_rotors = size(rotors_hist_binned, 2) * size(rotors_hist_binned, 3)
    step_counts = zeros(Int, 2^bits)
    step_flat = zeros(Int, number_of_rotors)
    hist = zeros(Int, number_of_steps, 2^bits)
    for i in 1:number_of_steps
        step_flat = reshape(rotors_hist_binned[i, :, :], number_of_rotors)
        step_counts .= 0
        for j in 1:number_of_rotors
            step_counts[step_flat[j]+1] += 1
        end
        hist[i, :] = step_counts
    end
    return hist
end

### ENTROPY VARIANTS BEGIN ###
function shannon_individual(rotors_hist_binned; bits=8)
    number_of_steps = size(rotors_hist_binned, 1)
    number_of_rotors = size(rotors_hist_binned, 2) * size(rotors_hist_binned, 3)
    shannon_entropy = zeros(Float64, number_of_steps)

    hist = get_binned_histogram(rotors_hist_binned, bits=bits)
    hist_probs = hist ./ number_of_rotors
    loop_length = 2^bits
    for i in 1:number_of_steps
        for j in 1:2^loop_length
            prob = hist_probs[i, j]
            if prob != 0
                shannon_entropy[i] -=  prob * log2(prob)
            end
        end
    end

    return shannon_entropy
end

function shannon_individual_time_averaged(rotors_hist_binned; bits=8)
    number_of_steps = size(rotors_hist_binned, 1)
    number_of_rotors = size(rotors_hist_binned, 2) * size(rotors_hist_binned, 3)
    shannon_entropy = zeros(Float64, number_of_steps)
    flattened_hist = reshape(rotors_hist_binned, number_of_steps, number_of_rotors)

    hist = get_binned_histogram(rotors_hist_binned, bits=bits)
    averaged_probs = zeros(Float64, 2^bits)
    
    for i in 1:number_of_steps
        averaged_probs += hist[i, :]/(number_of_rotors * number_of_steps)
    end
    
    for i in 1:number_of_steps
        for j in 1:number_of_rotors
            prob = averaged_probs[flattened_hist[i,j]+1]
            if prob != 0
                shannon_entropy[i] -=  prob * log2(prob)
            end
        end
    end
    
    return shannon_entropy
end

function zip_and_save(rotors_hist_binned, number_of_files)
    file_sizes = zeros(number_of_files+1)
    number_of_steps = size(rotors_hist_binned, 1)
    step = 1
    path = "./data/gzipped"
        
    
    zipfile = GZip.open(path, "w")
    write(zipfile, rotors_hist_binned[step, :, :])
    close(zipfile)
    file_sizes[1] = filesize(path)
    for i in 1:number_of_files
        step = floor(Int64, i*number_of_steps/number_of_files)
        if step % floor(number_of_steps/5) == 0
            println("Step: $(step)")
        end
        zipfile = GZip.open(path, "w")
        write(zipfile, rotors_hist_binned[step, :, :])
        close(zipfile)
        file_sizes[i+1] = filesize(path)
    end
    return file_sizes
end

### ENTROPY VARIANTS END ###

# Calculates the cosine sums for each rotor with their neighbors, used to calculate
# the total energy of the system.
function calc_cos_sums!(cos_sums, phases)
    N = size(phases, 1)
    M = size(phases, 2)
    pbc_phases = zeros(Float64, N+2, M+2)
    pbc_phases[2:end-1, 2:end-1] .= phases
    pbc_phases[1, 2:end-1] .= phases[end, :]
    pbc_phases[end, 2:end-1] .= phases[1, :]
    pbc_phases[2:end-1, 1] .= phases[:, end]
    pbc_phases[2:end-1, end] .= phases[:, 1]

    temp = zeros(Float64, N+2, M+2) # reset the sine sums array
    for j in 2:N-1 # here we loop over the actual rotor positions
        for i in 2:M-1
            @inbounds s_u = cos(pbc_phases[i-1,j] - pbc_phases[i,j])
            @inbounds s_l = cos(pbc_phases[i,j-1] - pbc_phases[i,j])
            @inbounds temp[i, j] = s_u + s_l
            @inbounds temp[i-1, j] += s_u
            @inbounds temp[i, j-1] += s_l
        end
    end
    cos_sums .= temp[2:end-1, 2:end-1]
    return
end

function calculate_total_energy(phasehist)
    N = size(phasehist, 2)
    M = size(phasehist, 3)
    t_length = size(phasehist, 1)
    pbc_phasehist = zeros(Float64, t_length, N+2, M+2)
    pbc_phasehist[:, 2:end-1, 2:end-1] .= phasehist
    pbc_phasehist[:, 1, 2:end-1] .= phasehist[:, end, :]
    pbc_phasehist[:, end, 2:end-1] .= phasehist[:, 1, :]
    pbc_phasehist[:, 2:end-1, 1] .= phasehist[:, :, end]
    pbc_phasehist[:, 2:end-1, end] .= phasehist[:, :, 1]
    total_energy_series = zeros(Float64, t_length)
    cos_phasehist = cos.(pbc_phasehist)
    sin_phasehist = sin.(pbc_phasehist)
    prods_u = zeros(Float64, N, M)
    prods_l = zeros(Float64, N, M)
    prods_r = zeros(Float64, N, M)
    prods_d = zeros(Float64, N, M)
    for t in 1:t_length
        @inbounds prods_u .= cos_phasehist[t, 2:end-1, 2:end-1] .* cos_phasehist[t, 1:end-2, 2:end-1] .+ sin_phasehist[t, 2:end-1, 2:end-1] .* sin_phasehist[t, 1:end-2, 2:end-1]
        @inbounds prods_l .= cos_phasehist[t, 2:end-1, 2:end-1] .* cos_phasehist[t, 2:end-1, 1:end-2] .+ sin_phasehist[t, 2:end-1, 2:end-1] .* sin_phasehist[t, 2:end-1, 1:end-2]
        @inbounds prods_r .= cos_phasehist[t, 2:end-1, 2:end-1] .* cos_phasehist[t, 2:end-1, 3:end] .+ sin_phasehist[t, 2:end-1, 2:end-1] .* sin_phasehist[t, 2:end-1, 3:end]
        @inbounds prods_d .= cos_phasehist[t, 2:end-1, 2:end-1] .* cos_phasehist[t, 3:end, 2:end-1] .+ sin_phasehist[t, 2:end-1, 2:end-1] .* sin_phasehist[t, 3:end, 2:end-1]
        total_energy_series[t] = - sum(prods_u) - sum(prods_l) - sum(prods_r) - sum(prods_d)
        
    end
    return total_energy_series
end

# Save data from 3D files (for example rotor histories, t x N x M) to feather format,
# a very compact binary format that can be quickly loaded for analysis and plotting.
function save_3D_to_feather(data, path, filename)
    timedim = size(data, 1)
    xdim = size(data, 2)
    ydim = size(data, 3)
    flattened_data = reshape(data, timedim, xdim*ydim)
    df = DataFrame(flattened_data, :auto)
    Arrow.write(path*filename, df)
    #also save the dimensions of the original data as a separate txt file for later loading
    open(path*filename*"_x_y_dims.txt", "w") do f
        write(f, string(xdim) * "," * string(ydim))
    end
    return
end

# Save data from 2D files (for example binned rotor-histogram, t x 2^bits) to feather format.
function save_2D_to_feather(data, path, filename)
    timedim = size(data, 1)
    xdim = size(data, 2)
    flattened_data = reshape(data, timedim, xdim)
    df = DataFrame(flattened_data, :auto)
    Arrow.write(path*filename, df)
    open(path*filename*"_x_dims.txt", "w") do f
        write(f, string(xdim))
    end
    return
end

# Save data from 1D files (for example total energy history) to feather format.
function save_1D_to_feather(data, path, filename)
    timedim = size(data, 1)
    # check if data is a vector, in which case we use the latter option
    if ndims(data) == 1
        df = DataFrame(:value => data)
    else
        df = DataFrame(data, :auto)
    end
    #df = DataFrame(data, :auto)
    Arrow.write(path*filename, df)
    return
end

# Load the final phases from a feather file. Used to continue simulations from a given final state.
function get_final_phases_bcd(filename, nrows, dim)
    # load the final angles from the file, unpacking the data like a 3D array
    read_hist = DataFrame(Arrow.Table(filename))
    warmup = zeros(Float64, nrows, dim, dim)
    for k in 1:dim
        for j in 1:dim
            for i in 1:nrows
                @inbounds warmup[i, j, k] = read_hist[i, j + (k-1)*dim]
            end
        end
    end
    final_phases = zeros(Float64, dim+2, dim+2)
    final_phases[2:end-1, 2:end-1] = warmup[end, :, :]
    final_phases[1, 2:end-1] = final_phases[end, 2:end-1]
    final_phases[end, 2:end-1] = final_phases[2, 2:end-1]
    final_phases[2:end-1, 1] = final_phases[2:end-1, end-1]
    final_phases[2:end-1, end] = final_phases[2:end-1, 2]

    return final_phases
end

# Either gets existing preset angular velocities or generates and saves a new preset.
# This allows for reproducible simulations with the same preset angular velocities,
# which is important for comparing different runs and analyzing the results.
function get_preset_angvels(array_dim, angvel_scale; v="")
    # check if there is a file called preset_$array_dim.csv in the data folder
    # if not, create it with random values and save it
    filename = "data/presets/preset_$(array_dim)$(v).csv"
    mkpath("data/presets")
    if !isfile(filename)
        println("Generating preset angular velocities for $array_dim x $array_dim rotors.")
        preset_angvels = 2 .* (rand(array_dim, array_dim) .- 0.5)
        # save the preset angular velocities to a csv file
        writedlm(filename, preset_angvels, ',')
        println("Saved preset angular velocities to $filename.")
    else
        println("Loading preset angular velocities from $filename.")
        preset_angvels = readdlm(filename, ',')
        if size(preset_angvels, 1) != array_dim || size(preset_angvels, 2) != array_dim
            error("Preset angular velocities must be of size $(array_dim) x $(array_dim).")
        end
    end
    preset_angvels = preset_angvels .* angvel_scale # scale the preset angular velocities
    if preset_angvels == zeros(Float64, array_dim, array_dim)
        warn("Preset angular velocities are all zeros.")
    end
    return preset_angvels
end





### MAIN SIMULATION AND DATA GENERATION FUNCTIONS START ###
### These functions run the simulations for specified parameters, and save the results to the highly compact feather format,
### which can be quickly loaded for analysis and plotting. The total data generated in in my master's thesis is around 110 GB,
### which if saved as CSV files would be around 5 times larger.
### The parameters were carefully chosen around the critical points of the system, where interesting phenomena such as phase
### transitions and hysteresis occur.

# Generating datasets for comparing measures of complexity and order, 
# and for plotting the phase diagrams.
function generate_solution_orderparams_and_entropy_save_less(array_dim, angvel_scale; 
    time=1000, dt=0.001, n_saved_files=1000,
    anim=false, v="", preset_phase=false, save_hist=true,
    init_phases=zeros(Float64, array_dim+2, array_dim+2),
    overwrite=false, use_version_specific_angvels=true,
    path=nothing
    )
    if path === nothing
        path = "data/angvel_series/$(array_dim)/"
    end
    mkpath(path)
    # check if there is a file called "$(angvel_scale)w_rotorhist.feather" in the data folder
    # if there is, return
    filename = path * "feather/$(angvel_scale)w_orderparams.feather"
    if isfile(filename) && !overwrite
        println("Simulation for array_dim=$array_dim and angvel_scale=$angvel_scale already exists at $filename. Skipping simulation.")
        return
    end

    timesteps = Int(round(time/dt))
    saverate = Int(round(timesteps ÷ time))
    if use_version_specific_angvels
        preset_angvels = get_preset_angvels(array_dim, angvel_scale, v=v)
    else
        preset_angvels = get_preset_angvels(array_dim, angvel_scale, v="")
    end
    
    if preset_phase
        rotors_hist, innate_angvelss, init_phases = kuramoto_model(array_dim,
            angvel_scale = angvel_scale,
            time_steps = timesteps,
            dt = dt,
            save_rate = saverate,
            animate = anim,
            phase_type = "preset",
            preset_phases = init_phases,
            filename = path * "kuramoto_$(array_dim)_angvels$(angvel_scale).gif",
            path = path,
            angvel_type = "preset",
            preset_angvels = preset_angvels
        )
    else
        rotors_hist, innate_angvelss, init_phases = kuramoto_model(array_dim,
            angvel_scale = angvel_scale,
            time_steps = timesteps,
            dt = dt,
            save_rate = saverate,
            animate = anim,
            phase_type = "random",
            filename = path * "kuramoto_$(array_dim)_angvels$(angvel_scale).gif",
            path = path,
            angvel_type = "preset",
            preset_angvels = preset_angvels
        )
    end
    feather_path = path * "feather/"
    mkpath(feather_path)

    if save_hist
        println("Saving rotor history to feather file...")
        save_3D_to_feather(rotors_hist, feather_path, "$(angvel_scale)w_rotorhist.feather")
    end

    encoded_rotors_hist = cast_and_encode(rotors_hist, bits=8)
    shannon_entropy = shannon_individual(encoded_rotors_hist, bits=8)
    save_1D_to_feather(shannon_entropy, feather_path, "$(angvel_scale)w_shannon8.feather")

    file_sizes = zip_and_save(encoded_rotors_hist, n_saved_files)
    save_1D_to_feather(file_sizes, feather_path, "$(angvel_scale)w_file_sizes.feather")

    encoded_rotors_hist6bit = cast_and_encode(rotors_hist, bits=6)
    shannon_entropy = shannon_individual(encoded_rotors_hist6bit, bits=6)
    save_1D_to_feather(shannon_entropy, feather_path, "$(angvel_scale)w_shannon6.feather")

    file_sizes6bit = zip_and_save(encoded_rotors_hist6bit, n_saved_files)
    save_1D_to_feather(file_sizes6bit, feather_path, "$(angvel_scale)w_file_sizes_6bit.feather")

    total_energy = calculate_total_energy(rotors_hist)
    save_1D_to_feather(total_energy, feather_path, "$(angvel_scale)w_energy.feather")

    order_param_series = get_order_params(rotors_hist)
    save_2D_to_feather(order_param_series, feather_path, "$(angvel_scale)w_orderparams.feather")

    return
end

# This function is similar to the previous one, but it only saves the history and parameters
# that are useful for checking for hysteresis.
function generate_solution_orderparams_hysteresis(array_dim, angvel_scale; 
    time=1000, dt=0.001, n_saved_files=1000,
    anim=false, v="", preset_phase=false, save_hist=true,
    init_phases=zeros(Float64, array_dim+2, array_dim+2),
    overwrite=false, use_version_specific_angvels=true,
    path=nothing
    )
    if path === nothing
        path = "data/angvel_series/$(array_dim)/"
    end
    mkpath(path)
    # check if there is a file called "$(angvel_scale)w_rotorhist.feather" in the data folder
    # if there is, return
    filename = path * "feather/$(angvel_scale)w_orderparams.feather"
    if isfile(filename) && !overwrite
        println("Simulation for array_dim=$array_dim and angvel_scale=$angvel_scale already exists at $filename. Skipping simulation.")
        return
    end

    timesteps = Int(round(time/dt))
    saverate = Int(round(timesteps ÷ time))
    if use_version_specific_angvels
        preset_angvels = get_preset_angvels(array_dim, angvel_scale, v=v)
    else
        preset_angvels = get_preset_angvels(array_dim, angvel_scale, v="")
    end
    
    if preset_phase
        rotors_hist, innate_angvelss, init_phases = kuramoto_model(array_dim,
            angvel_scale = angvel_scale,
            time_steps = timesteps,
            dt = dt,
            save_rate = saverate,
            animate = anim,
            phase_type = "preset",
            preset_phases = init_phases,
            filename = path * "kuramoto_$(array_dim)_angvels$(angvel_scale).gif",
            path = path,
            angvel_type = "preset",
            preset_angvels = preset_angvels
        )
    else
        rotors_hist, innate_angvelss, init_phases = kuramoto_model(array_dim,
            angvel_scale = angvel_scale,
            time_steps = timesteps,
            dt = dt,
            save_rate = saverate,
            animate = anim,
            phase_type = "random",
            filename = path * "kuramoto_$(array_dim)_angvels$(angvel_scale).gif",
            path = path,
            angvel_type = "preset",
            preset_angvels = preset_angvels
        )
    end
    feather_path = path * "feather/"
    mkpath(feather_path)

    if save_hist
        println("Saving rotor history to feather file...")
        save_3D_to_feather(rotors_hist, feather_path, "$(angvel_scale)w_rotorhist.feather")
    end

    order_param_series = get_order_params(rotors_hist)
    save_2D_to_feather(order_param_series, feather_path, "$(angvel_scale)w_orderparams.feather")

    encoded_rotors_hist = cast_and_encode(rotors_hist, bits=8)
    shannon_entropy = shannon_individual(encoded_rotors_hist, bits=8)
    save_1D_to_feather(shannon_entropy, feather_path, "$(angvel_scale)w_shannon8.feather")

    file_sizes = zip_and_save(encoded_rotors_hist, n_saved_files)
    save_1D_to_feather(file_sizes, feather_path, "$(angvel_scale)w_file_sizes.feather")

    encoded_rotors_hist6bit = cast_and_encode(rotors_hist, bits=6)
    shannon_entropy = shannon_individual(encoded_rotors_hist6bit, bits=6)
    save_1D_to_feather(shannon_entropy, feather_path, "$(angvel_scale)w_shannon6.feather")

    file_sizes6bit = zip_and_save(encoded_rotors_hist6bit, n_saved_files)
    save_1D_to_feather(file_sizes6bit, feather_path, "$(angvel_scale)w_file_sizes_6bit.feather")

    return
end

# Runs multiple instances of the function above for a range of angular velocities
function hysteresis_64x64()
    #angvels_hysteresis_32 = [
    #    "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0",
    #    "1.1", "1.15", "1.2", "1.25", "1.3", "1.32", "1.34", "1.36", "1.38",
    #    "1.4", "1.41", "1.42", "1.43", "1.44", "1.46", "1.48",
    #    "1.5", "1.52", "1.54"
    #]
    angvels_hysteresis_64 = [
        "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0",
        #"1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "2.0",
        ]

    system_size = 64#32
    versions = [
        "below", 
        "above"]
    sim_time = 1500#1000
    stepsize = 0.001
    ow = true
    for ver in versions
        if ver == "below"
                init_state = get_final_phases_bcd("data/angvel_series/64/feather/0.1w_rotorhist.feather", 2500, system_size)
                #init_state = get_final_phases_bcd("data/angvel_series/32/feather/0.1w_rotorhist.feather", 2000, system_size)
                #ow = true
                angs = angvels_hysteresis_64
            else
                init_state = get_final_phases_bcd("data/angvel_series/hysteresis/64above/feather/0.7w_rotorhist.feather", 1000, system_size)
                #init_state = get_final_phases_bcd("data/angvel_series/hysteresis/32above/feather/1.1w_rotorhist.feather", 1000, system_size)
                angs = reverse(angvels_hysteresis_64)
            end
        for angvel in angs
            println("----- Running for angvel $angvel, version $ver -----")
            vel = parse(Float64, angvel)
            @time generate_solution_orderparams_hysteresis(system_size, vel, time=sim_time, dt=stepsize, 
                anim=false, v=ver, n_saved_files=round(Int, sim_time/2),
                preset_phase=true, init_phases=init_state,
                use_version_specific_angvels=false,
                overwrite=ow, path="data/angvel_series/hysteresis/$(system_size)$(ver)/",
            )
            println("Finished for angvel $angvel.")
        end
    end
    return
end

# Runs a large number of simulations for a range of angular velocities and system sizes, to
# generate data for the purpose of comparing the effects of changing the system size.
function l_scales()
    angvels8  = ["0.2", "0.4", "0.6", "0.8"]#, "1.0", "1.2", "1.4", "1.6", "1.7", "1.8", "1.9", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.9", "3.0"]
    angvels12 = ["0.2", "0.4", "0.6", "0.8"]#, "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.7", "2.9"]
    angvels16 = ["0.2", "0.4", "0.6", "0.8"]#, "0.9", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "2.0", "2.2", "2.4"]
    angvels20 = ["0.2", "0.4", "0.6", "0.8"]#, "0.9", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "2.0", "2.2", "2.4"]
    angvels24 = ["0.2", "0.4", "0.6", "0.7"]#, "0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.8", "2.0", "2.2", "2.4"]
    angvels32 = ["0.2", "0.4", "0.6", "0.7"]#, "0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "2.0", "2.2", "2.4"]
    angvels40 = ["0.2", "0.4", "0.6", "0.7"]#, "0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "2.0", "2.2", "2.4"]
    angvels48 = ["0.2", "0.4", "0.6", "0.7"]#, "0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "2.0", "2.2", "2.4"]
    angvels = [angvels8, angvels12, angvels16, angvels20, angvels24, angvels32, angvels40, angvels48]
    system_sizes = [8, 12, 16, 20, 24, 32, 40, 48
    ]
    versions = ["", "_lscale1", "_lscale2", "_lscale3", "_lscale4",
    "l_scale5", "l_scale6", "l_scale7", "l_scale8", "l_scale9"
    ]
    sim_times = [1000, 1000, 1000, 1500, 1500, 2000, 2000, 2000]
    stepsize = 0.005
    path = "data/angvel_series/l_scales_sync/"
    for (i, sys_size) in enumerate(system_sizes)
        sim_time = sim_times[i]
        for ver in versions
            for angvel in angvels[i]
                init_state = zeros(Float64, sys_size+2, sys_size+2)
                println("---------- w = $angvel, L = $sys_size, v = $ver ----------")
                vel = parse(Float64, angvel)
                @time generate_solution_orderparams_and_entropy_save_less(sys_size, vel, time=sim_time, dt=stepsize, 
                    anim=false, v=ver, n_saved_files=1000,
                    preset_phase=true, init_phases=init_state,
                    use_version_specific_angvels=true,
                    overwrite=false, path=path * "$(sys_size)$(ver)/",
                )
                println("--- Finished for w = $angvel, L = $sys_size, v = $ver ---")
            end
        end
    end
    return
end

# As above but for a single system size, to obtain more data points around
# critical points (phase transitions).
function l_scale()
    system_size = 56
    versions = ["", "_lscale1", "_lscale2", "_lscale3", "_lscale4", "l_scale5", "l_scale6", "l_scale7", "l_scale8", "l_scale9"]
    sim_time = 2500
    stepsize = 0.005
    angvels = ["0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.4", "1.6", "1.8", "2.0", "2.2", "2.4",
    #"1.9", "2.1", "2.3", "2.5", "2.6", "2.7", "2.8"# L=8
    #"1.3", "1.5", "1.7", "1.9", "2.1"#L=12
    #"0.9", "1.1", "1.3", "1.5", "1.7", "1.9"# L=16
    #"0.9", "1.1", "1.3", "1.5", "1.7"# L=20
    #"0.7", "0.9", "1.1", "1.3", "1.5"# L=24
    #"0.7", "0.9", "1.1", "1.3", "1.5", "1.7"# L=32
    #"0.9", "1.1", "1.3", "1.5", "1.7"# L=40
    "0.7", "0.9", "1.1", "1.3", "1.5", "1.7"# L=48
    ]
    path = "data/angvel_series/l_scales_sync/"
    for ver in versions
        for angvel in angvels
            init_state = zeros(Float64, system_size+2, system_size+2)
            println("---------- w = $angvel, L = $system_size, v = $ver ----------")
            vel = parse(Float64, angvel)
            @time generate_solution_orderparams_and_entropy_save_less(system_size, vel, time=sim_time, dt=stepsize, 
                anim=false, v=ver, n_saved_files=1000,
                preset_phase=true, init_phases=init_state,
                use_version_specific_angvels=true,
                overwrite=false, path=path * "$(system_size)$(ver)/",
            )
            println("--- Finished for w = $angvel, L = $system_size, v = $ver ---")
        end
    end
    return
end

# Checking the stability of the simulations for different time steps. Even with highly optimized
# code, the simulations take a long time to run for large systems and long simulation times, so
# finding the largest time step dt that still produces stable and accurate results is important.
function time_step_stability()
    array_dim = 32
    angvel_scales = [0.3, 1.0, 1.42, 1.6, 2.0]
    sim_time = 2000
    dts = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
    path = "data/angvel_series/time_step_stability/"
    for angvel_scale in angvel_scales
        for dt in dts
            init_state = get_final_phases_bcd("data/angvel_series/32/feather/8.0w_rotorhist.feather", 2000, 32)
            println("Running with dt = $dt")
            @time generate_solution_orderparams_and_entropy_save_less(array_dim, angvel_scale, 
                time=sim_time, dt=dt, n_saved_files=500,
                anim=true, v="$dt", preset_phase=true, save_hist=true,
                init_phases=init_state,
                overwrite=true, use_version_specific_angvels=false,
                path=path * "$(array_dim)_dt_$(dt)/"
            )
        end
    end
    return
end

# Computation time increases quadratically with the system size. The specific angular velocity
# at which the phase transitions occurs also changes with the system size. To save time, only
# a few angular velocities are simulated for the largest system size, around the values where
# the phase transitions is expected to occur.
function large_system()
    array_dim = 100
    angvel_scales = [
        "0.55", "0.6", "0.65", 
        "0.685", "0.785", "0.885", "0.9", 
        #"0.95"
    ]
    sim_time = 4000
    dt = 0.005
    path = "data/angvel_series/100/"
    init_state = get_final_phases_bcd("data/angvel_series/100/feather/0.95w_rotorhist.feather", 4000, array_dim)
    for angvel_scale in reverse(angvel_scales)
        println("Running with angvel_scale = $angvel_scale")
        vel = parse(Float64, angvel_scale)
        @time generate_solution_orderparams_and_entropy_save_less(array_dim, vel, 
            time=sim_time, dt=dt, n_saved_files=1000,
            anim=false, v="$angvel_scale", preset_phase=true, save_hist=true,
            init_phases=init_state,
            overwrite=true, use_version_specific_angvels=false,
            path=path
        )
    end
    return
end

function benchmark_RK4()
    array_dim = 32
    angvel_scale = 1.1
    sim_time = 1500
    dt = 0.005
    path = "data/benchmark/"
    init_state = zeros(Float64, array_dim+2, array_dim+2)
    t_steps = Int(round(sim_time/dt))
    angvel_preset = get_preset_angvels(array_dim, angvel_scale)
    kuramoto_model(array_dim; 
        angvel_scale=angvel_scale, angvel_type="preset", preset_angvels=angvel_preset,
        time_steps=t_steps, dt=dt, phase_type="preset", preset_phases=init_state,
        animate=false, benchmark=true, path=path
    )
    return
end

function animate(;v="")
    array_dim = 32
    angvel_scale = 0.3
    sim_time = 1500
    dt = 0.005
    path = "data/animations/"
    t_steps = Int(round(sim_time/dt))
    angvel_preset = get_preset_angvels(array_dim, angvel_scale, v=v)
    kuramoto_model(array_dim; 
        angvel_scale=angvel_scale, angvel_type="preset", preset_angvels=angvel_preset,
        time_steps=t_steps, dt=dt, phase_type="random",
        animate=true, filename=path * "kuramoto_$(array_dim)_angvels$(angvel_scale)_$(v).gif"
    )
    return
end


function main()
    #hysteresis()
    #l_scale()
    #l_scales()
    #time_step_stability()
    #large_system()
    #benchmark_RK4()
    return
end    



println("Timing full run of main")
@time main()
println("Finished running kuramoto_model.jl")
