module FluoroDist
using Images
using Random
using StatsBase
using LinearAlgebra
using StaticArrays
using Interpolations
using ArgParse

export shape_from_points, ellipse, to_image_stack

function get_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--disable-naturalization"
            help = "do not reparameterize curve"
            action = :store_false
        "shape"
            help = "shape"
            required = true
            arg_type = String
    end
    parse_args(s)
end

curry_get(ind) = x -> x[ind]
components(svec) = (map(curry_get(1), svec), map(curry_get(2), svec))

function simulate_fluoresence(postprocess, t, cx, cy, f, df, nsamp, d, psf_sigma)
    ts = sample(t, nsamp)
    x, y = components(f.(ts))
    dx, dy = components(@. normalize(df(ts)))
    v = psf_sigma*randn(nsamp, 2) .+ postprocess(hcat(x .+ cx .- dy.*d, y .+ cy .+ dx.*d))
    @view(v[:,1]), @view(v[:,2])
end

function create_patch_transformer(spaces, space_factor)
    function patch_transformer(t)
        spaces = length(t)*spaces./(space_factor*sum(spaces))
        new_t = Float64[]
        ind = 1
        for s in round.(Int, spaces)
            append!(new_t, t[ind:ind+s])
            ind += space_factor*s
        end
        new_t
    end
end

function patchy(n_patches, space_factor)
    spaces = rand(n_patches)
    create_patch_transformer(spaces, space_factor)
end

function dotty(n, space_factor)
    create_patch_transformer(ones(n), space_factor)
end

function naturalize(theta, magdf, n_t)
    new_theta = Float64[]
    cur = theta[1]
    while cur < theta[end]
        push!(new_theta, cur)
        cur += 1/(n_t * magdf(cur) * 2pi)
    end
    new_theta
end

function ellipse(a, b; angoff=0, d=2.0, nsamp=100000, n_stack=20, p=110)
    fx(t) = a*sin(t)
    dfx(t) = a*cos(t)
    fy(t) = b*cos(t)
    dfy(t) = -b*sin(t)
    rot = SMatrix{2,2}(cos(angoff), sin(angoff), -sin(angoff), cos(angoff))
    f(theta) = rot*SVector(fx(theta), fy(theta))
    df(theta) = rot*SVector(dfx(theta), dfy(theta))
    magdf(theta) = hypot(df(theta)...)
    t(s) = 2π*s
    
    f ∘ t, df ∘ t
end

function shape_from_points(points_x, points_y)
    points_x = points_x .- mean(points_x)
    points_y = points_y .- mean(points_y)
    
    itp_y = interpolate(points_x, BSpline(Quadratic(Periodic())), OnCell())
    itp_x = interpolate(points_y, BSpline(Quadratic(Periodic())), OnCell())
    grad(itp) = x -> Interpolations.gradient(itp, x)[1]
    
    gx = grad(itp_x)
    gy = grad(itp_y)

    df(t) = SVector(gx(t), gy(t))
    f(t) = SVector(itp_x(t), itp_y(t))
    t(s) = 0.5+length(points_x)*s
    
    f ∘ t, df ∘ t
end

function sample_circle(n)
    ang = rand(n,2)*2pi
    hcat(cos.(ang[:,1]).*sin.(ang[:,2]), sin.(ang[:,1]).*sin.(ang[:,2]))
end

function to_image_stack(shape; d=2.0, nsamp=100000, t_transform=identity, naturalize_t=true, n_stack=20, psf_sigma=4, n_t=1000, width=0, primary_length=0, secondary_length=0, pos_jiggle = 40)
    f, df = shape
    p = 40
    
    magdf(theta) = hypot(df(theta)...)
    
    t = LinRange(0, 1, n_t+1)[1:end-1]
    t = naturalize_t ? naturalize(t, magdf, n_t) : t

    w = ceil(Int, maximum(abs.(extrema([v[d] for v in f.(t) for d in 1:2]))) + pos_jiggle + psf_sigma*4 + d)
    cx, cy = 1 .+ p .+ w .+ pos_jiggle*rand(2) .- pos_jiggle/2
    
    mapreduce((x,y) -> cat(x, y, dims=3), 1:n_stack) do i
        xs, ys = simulate_fluoresence(identity, t, cx, cy, f, df, nsamp, (rand(nsamp).-0.5)*width, psf_sigma)
        xds, yds = simulate_fluoresence(t_transform(t), cx, cy, f, df, nsamp, d, psf_sigma) do mat
            mat .+= sample_circle(nsamp)*primary_length
            mat .+= sample_circle(nsamp)*secondary_length
            mat
        end
        
        img = zeros(2p+2w+1,2p+2w+1,2)
        for (x,y,xd,yd) in zip(xs,ys,xds,yds)
            img[round(Int,x), round(Int,y), 1] += 1.0
            img[round(Int,xd), round(Int,yd), 2] += 1.0
        end
        img[:,:,1] ./= maximum(img[:,:,1])
        img[:,:,2] ./= maximum(img[:,:,2])
        img
    end
end

end # module
