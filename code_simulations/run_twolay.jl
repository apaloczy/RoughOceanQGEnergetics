all_F₁s = [25, 75, 400]#[2500, 1600, 900, 625, 400, 100, 25]#400:-25:25
all_hts = [1, 5, 10]#1:15#0#7#1:10

#ttype = "ridg"
ttype = "cosi"
#ttype = "rand"; fname_hrand = "hrand256Km2tk10filtnx32.mat"

SAVE_OUTPUT = true
PSI_HOVMOLLER = true

#-------------------
include("twolay.jl")
