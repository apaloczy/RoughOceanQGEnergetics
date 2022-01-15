all_F₁s = [400, 100, 25]#400#400:-25:25
all_hts = 7#0:10

ttype = "cosi"
# ttype = "rand"; fname_hrand = "hrand256Km2tk10filtnx32.mat"

SAVE_OUTPUT = true
PSI_HOVMOLLER = true

#-------------------
include("twolay.jl")
