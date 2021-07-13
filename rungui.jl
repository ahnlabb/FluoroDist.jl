import Pkg
Pkg.activate("gui")
Pkg.instantiate()
import FluoroDistGUI, Gtk
win = FluoroDistGUI.gui()
@async Gtk.gtk_main()
Gtk.waitforsignal(win,:destroy)
