module FluoroDistGUI
include(joinpath(@__DIR__, "..", "..", "src", "FluoroDist.jl"))
using .FluoroDist
import Gtk
using Gtk.ShortNames, GtkObservables, Graphics, Images
using Random

transform_dict = Dict(
    "Full" => identity,
    "Dotty" => FluoroDist.dotty(7,2),
    "Patchy" => FluoroDist.patchy(7,2)
)

function gui()
    win = Window("Testing", 800, 1130)
    bx = Box(:v)
    win |> bx
    function push_labeled!(w, l)
        hbx = Box(:h)
        push!(hbx, Label(l))
        push!(hbx, w)
        push!(bx, hbx)
        set_gtk_property!(hbx,:expand,widget(w),true)
        observable(w)
    end
    a = push_labeled!(slider(1:100), "First axis")
    b = push_labeled!(slider(1:100), "Second axis")
    angle = push_labeled!(slider(0.0:0.01:2π), "Angle")
    psf_sigma = push_labeled!(slider(0.0:0.1:8.0), "PSF sigma")
    distance = push_labeled!(slider(0.0:0.1:16.0), "Distance")
    width = push_labeled!(slider(0.0:0.1:20.0), "Membrane Width")
    primary_length = push_labeled!(slider(0.0:0.1:20.0), "Ligand Length")
    secondary_length = push_labeled!(slider(0.0:0.1:20.0), "Ligand Length")
    transform_dd = dropdown(keys(transform_dict))
    transform = map(transform_dd) do t
        x -> transform_dict[t](x)
    end
    nsamp = push_labeled!(slider(1:20), "Exposure")
    nsamp_full = map(nsamp) do n
        2^n
    end
    
    img = map(a, b, nsamp_full, transform, angle, psf_sigma, distance, width, primary_length, secondary_length) do a, b, nsamp, t_transform, angoff, psf_sigma, d, width, primary_length, secondary_length
        Random.seed!(1)
        to_image_stack(ellipse(a,b; angoff); n_stack=1, nsamp, t_transform, psf_sigma, d, width, primary_length, secondary_length, pos_jiggle=0)
    end

    c = canvas(800,800)
    redraw = draw(c, img) do cnvs, img
        memb = @view img[:,:,1]
        target = @view img[:,:,2]
        copy!(cnvs, colorview(RGB, memb, target, target))
    end
    push!(bx, transform_dd)
    push!(bx, c)
    Gtk.showall(win)
    win
end


end # module
