using FourierFlows: CPU, TwoDGrid
using FFTW
using Random: seed!, rand

dev = CPU()           # Device.
N = 256               # Number of grid points.
L = 2.0               # Length of the square domain.
stepper = "RK4"       # Type of time stepper (e.g., RK4, AB3).
dt = 2e-3             # Size of timestep.
effort = FFTW.PATIENT # FFTW effort flag (e.g., FFTW.MEASURE, FFTW.PATIENT).
nlog = 100            # Display log and save Hovmöller slices every `nlog` timesteps.
δ = 1/4               # Upper- to lower-layer rest thickness ratio.
kt = 10π              # Wavenumber of monochromatic topography (1D or 2D).
α = 0.0               # Linear slope component of topographic background potential vorticity gradient.
β = 0.0               # Planetary background potential vorticity gradient.
U₀ = 1.0              # Amplitude of basic jet.
Lj = L/10             # Half-width of basic jet.
U₂fac = 0.0           # Amplitude of lower-layer jet as fraction of upper-layer jet's amplitude.
qmag = 0.004          # Amplitude of the random initial condition.
Rthresh = 0.01        # Threshold energy ratio for renormalization.
cycles = 3            # Total number of renormalization cycles.

cyclesm = cycles - 1
gr = TwoDGrid(N, L, N, L, x0=0.0, aliased_fraction=0.0)
U, Uyy = NondimLinearTwoLayerQG.get_bickley(gr, U₀, Lj; U₂fac=U₂fac)
seed!(1234) # Fix the seed of the random number generator for reproducibility.
q₀ = qmag * gr.nx * gr.ny * rand(gr.nx, gr.ny, 2)
q₀[:, :, 2] .= 0
