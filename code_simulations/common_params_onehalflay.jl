using FourierFlows: CPU, TwoDGrid
using FFTW
using Random: seed!, rand

dev = CPU()           # Device.
N = 256               # Number of grid points.
L = 2.0               # Length of the square domain.
stepper = "RK4"       # Type of time stepper (e.g., RK4, AB3).
dt = 2e-3             # Size of timestep.
effort = FFTW.PATIENT # FFTW effort flag (e.g., FFTW.MEASURE, FFTW.PATIENT).
nlog = 500            # Display log and save Hovmöller slices every `nlog` timesteps.
β = 0.0               # Planetary background potential vorticity gradient.
U₀ = 1.0              # Amplitude of basic jet.
Lj = L/10             # Half-width of basic jet.
qmag = 0.004          # Amplitude of the random initial condition.
Rthresh = 0.01        # Threshold energy ratio for renormalization.
cycles = 3            # Total number of renormalization cycles.

cyclesm = cycles - 1
gr = TwoDGrid(N, L, N, L, x0=0.0, aliased_fraction=0.0)
U, Uyy = NondimLinearOneHalfLayerQG.get_bickley(gr, U₀, Lj)
seed!(1234) # Fix the seed of the random number generator for reproducibility.
q₀ = qmag * gr.nx * gr.ny * rand(gr.nx, gr.ny)
