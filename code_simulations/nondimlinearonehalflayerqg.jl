module NondimLinearOneHalfLayerQG

export
  Problem,
  set_q!,
  updatevars!,
  get_bickley,
  get_σ,
  streamfunctionfrompv!,
  energies

using
  FFTW,
  Reexport,
  DocStringExtensions

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!
using FourierFlows: parsevalsum


"""
    Problem(dev::Device=CPU();
                nx = 256,
                ny = nx,
                Lx = 2.0,
                Ly = Lx,
                x0 = 0.0,
                 F = 25.0,
                 U = nothing,
               Uyy = nothing,
                 β = 0.0,
                dt = 2e-3,
           stepper = "RK4",
            effort = FFTW.MEASURE,
  aliased_fraction = 0,
                 T = Float64)
Construct a two-dimensional nondimensional linear two-layer QG `problem` on device `dev`.
Keyword arguments
=================
  - `dev`: (required) `CPU()` or `GPU()`; computer architecture used to time-step `problem`.
  - `nx`: Number of grid points in ``x``-domain.
  - `ny`: Number of grid points in ``y``-domain.
  - `Lx`: Extent of the ``x``-domain.
  - `Ly`: Extent of the ``y``-domain.
  - `x0`: Origin of the ``x``-domain.
  - `F`: Inverse Burger number.
  - `U`: Imposed zonal flow U(y).
  - `Uyy`: Meridional relative vorticity PV gradient.
  - `β`: Meridional linear planetary PV gradient.
  - `dt`: Time-step.
  - `stepper`: Time-stepping method.
  - `effort`: FFTW effort.
  - `aliased_fraction`: the fraction of high-wavenumbers that are zero-ed out by `dealias!()`.
  - `T`: `Float32` or `Float64`; floating point type used for `problem` data.
"""
function Problem(dev = CPU();
    # Numerical parameters
                  nx = 256,
                  ny = nx,
                  Lx = 2.0,
                  Ly = Lx,
                  x0 = 0.0,
    # Physical parameters
                   F = 25.0,     # inverse Burger number
                   U = nothing,  # imposed zonal flow U(y)
                 Uyy = nothing,  # meridional relative vorticity PV gradient
                   β = 0.0,      # meridional linear planetary PV gradient
    # Timestepper and equation options
                  dt = 2e-3,
             stepper = "RK4",
              effort = FFTW.MEASURE,
  # Float type and dealiasing
    aliased_fraction = 0.0,
                   T = Float64)

    # The grid.
    grid = TwoDGrid(dev, nx, Lx, ny, Ly; x0=x0, aliased_fraction=aliased_fraction, T=T, effort=effort)

    # No mean flow case.
    U === nothing && (U = zeros(dev, T, (1)); Uyy = zeros(dev, T, (1)))

    params = Params(F, U, Uyy, β, grid, dev=dev)
    vars = Vars(dev, grid)
    equation = Equation(dev, params, grid)

    FourierFlows.Problem(equation, stepper, dt, grid, vars, params, dev)
end


abstract type NondimLinearOneHalfLayerQGParams <: AbstractParams end

"""
    Params{T, Aphys2D}(F, U, Uyy, β, grid, Qy)
A struct containing the parameters for the NondimLinearOneHalfLayerQG problem. Included are:
$(TYPEDFIELDS)
"""
struct Params{T, Aphys2D} <: NondimLinearOneHalfLayerQGParams
    # prescribed params
    "inverse Burger number"
         F :: T
    "background zonal flow U(y)"
         U :: Aphys2D
    "background PV gradient due to zonal flow's lateral shear"
       Uyy :: Aphys2D
     "planetary vorticity gradient (in the y direction)"
         β :: T

    # derived param
    "background potential vorticity gradient in y-direction due to β and U"
        Qy :: Aphys2D
end


function convert_U_to_U3D(dev, grid, U::AbstractArray{TU, 1}) where TU
  T = eltype(grid)
  if length(U) == 1
    U_2D = zeros(dev, T, (1, 2))
    U_2D[:] = U
    U_2D = repeat(U_2D, outer=(grid.ny, 1))
  else
    U_2D = zeros(dev, T, (grid.ny, 1))
    U_2D[:] = U
  end
  U_3D = zeros(dev, T, (1, grid.ny))
  @views U_3D[1, :, :] = U_2D
  return U_3D
end


function convert_U_to_U3D(dev, grid, U::AbstractArray{TU, 2}) where TU
  T = eltype(grid)
  U_3D = zeros(dev, T, (1, grid.ny))
  @views U_3D[1, :, :] = U
  return U_3D
end


function Params(F, U, Uyy, β, grid; dev::Device=CPU()) where TU
    T = eltype(grid)
    A = ArrayType(dev)

    U = convert_U_to_U3D(dev, grid, U)
    Uyy = convert_U_to_U3D(dev, grid, Uyy)

    ny, nx = grid.ny, grid.nx
    Krsq = grid.Krsq

    Qy = zeros(dev, T, (nx, ny))
    @. Qy = F*U - Uyy + β

    return Params(F, U, Uyy, β, Qy)
end


"""
    Equation(params::NondimLinearOneHalfLayerQGParams, grid)
Return the `equation` for a nondimensional one and a half-layer QG problem with `params` and `grid`. Linear operator
``L`` is empty in this case (no hypo-/hyer-viscosity, and β terms are absorbed in the background PV gradient term.)
Nonlinear term is computed via `calcN!` function.
"""
function Equation(dev, params, grid)
  T = eltype(grid)
  L = zeros(dev, T, (grid.nkr, grid.nl))

  return FourierFlows.Equation(L, calcN!, grid)
end


abstract type NondimLinearOneHalfLayerQGVars <: AbstractVars end

"""
    Vars{Aphys, Atrans, F, P}(q, ψ, u, v, qh, ψh, uh, vh, prevsol)
The variables for NondimLinearOneHalfLayer QG:
$(FIELDS)
"""
struct Vars{Aphys, Atrans, P} <: NondimLinearOneHalfLayerQGVars
    "potential vorticity"
        q :: Aphys
    "streamfunction"
        ψ :: Aphys
    "x-component of velocity"
        u :: Aphys
    "y-component of velocity"
        v :: Aphys
    "Fourier transform of potential vorticity"
       qh :: Atrans
    "Fourier transform of streamfunction"
       ψh :: Atrans
    "Fourier transform of x-component of velocity"
       uh :: Atrans
    "Fourier transform of y-component of velocity"
       vh :: Atrans
    "`sol` at previous time-step"
  prevsol :: P
end


"""
    Vars(dev, grid)
Return the `vars` for for the NondimLinearOneHalfLayerQG problem on device `dev` and
with `grid`.
"""
function Vars(::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny) q ψ u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) qh ψh uh vh prevsol

  return Vars(q, ψ, u, v, qh, ψh, uh, vh, prevsol)
end


"""
    calcN!(N, sol, t, clock, vars, params, grid)
Calculate the nonlinear term, that is the advection term,
```math
N = - \\widehat{𝖩(ψ, q+η)}.
```
"""
function calcN!(N, sol, t, clock, vars, params, grid)
    @. vars.qh = sol
    streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
    @. vars.uh = -im * grid.l  * vars.ψh
    @. vars.vh =  im * grid.kr * vars.ψh

    qx, qxh = deepcopy(vars.q), deepcopy(vars.qh)
    @. qxh = im * grid.kr * vars.qh

    ldiv!(vars.q, grid.rfftplan, deepcopy(vars.qh))
    ldiv!(vars.u, grid.rfftplan, deepcopy(vars.uh))
    ldiv!(vars.v, grid.rfftplan, deepcopy(vars.vh))
    ldiv!(qx, grid.rfftplan, deepcopy(qxh))

    # use vars.u, vars.v, vars.uh, vars.vh as scratch variables
    rhs1, rhs1h = deepcopy(vars.u), deepcopy(vars.uh)
    rhs2, rhs2h = deepcopy(vars.v), deepcopy(vars.vh)
    rhs1 = @. params.U*qx
    rhs2 = @. vars.v*params.Qy
    mul!(rhs1h, grid.rfftplan, deepcopy(rhs1)) # \hat{U*qx}
    mul!(rhs2h, grid.rfftplan, deepcopy(rhs2)) # \hat{v*Qy}

    @. N = - (rhs1h + rhs2h)
    return nothing
end


"""
    streamfunctionfrompv!(ψh, qh, params::NondimLinearOneHalfLayerQGParams, grid)
Invert the PV to obtain the Fourier transform of the streamfunction `ψh`.
"""
function streamfunctionfrompv!(ψh, qh, params::NondimLinearOneHalfLayerQGParams, grid)
  @. ψh  = - qh / (grid.Krsq + params.F)

  return nothing
end


# ----------------
# Helper functions
# ----------------


"""
    updatevars!(vars, params, grid, sol)
    updatevars!(prob)
Update all problem variables using `sol`.
"""
function updatevars!(vars, params, grid, sol)
  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  ldiv!(vars.q, grid.rfftplan, deepcopy(vars.qh))
  ldiv!(vars.ψ, grid.rfftplan, deepcopy(vars.ψh))
  ldiv!(vars.u, grid.rfftplan, deepcopy(vars.uh))
  ldiv!(vars.v, grid.rfftplan, deepcopy(vars.vh))

  return nothing
end

updatevars!(prob) = updatevars!(prob.vars, prob.params, prob.grid, prob.sol)


"""
    set_q!(sol, params, vars, grid, q)
    set_q!(prob, q)
Set the solution `prob.sol` as the transform of `q` and update variables.
"""
function set_q!(sol, params, vars, grid, q)
  A = typeof(vars.q)
  mul!(vars.qh, grid.rfftplan, A(q))

  vars.qh[1, 1] = 0
  @. sol = vars.qh

  updatevars!(vars, params, grid, sol)

  return nothing
end

set_q!(prob, q) = set_q!(prob.sol, prob.params, prob.vars, prob.grid, q)


"""
    get_bickley(grid, U₀, Lj)
Get a Bickley jet of half-width `Lj` and amplitude `U₀` on grid `grid`.
"""
function get_bickley(grid, U₀, Lj)
    T = eltype(grid)
    y, ny = grid.y, grid.ny
    U, Uyy = zeros(T, ny), zeros(T, ny)

    @. U = U₀*sech(y/Lj)^2
    @. Uyy = (U₀*2/Lj^2)*U*(3*tanh(y/Lj)^2 - 1)

    return U, Uyy
end


"""
    get_σ(t, KE, PE)
Estimate the growth rate of instabilities from a least-squares fit to energy
histories. `KE` is the kinetic energy and `PE` is the potential energy.
"""
function get_σ(t, KE, PE)
  f = findlast(diff(KE + PE).<-0.5) + 1 # Get first time index from the last renormalization cycle.
  t, KE = t[f:end], KE[f:end]
  n = size(t)[1]
  d = Matrix(reshape(log.(KE), (1, n)))
  gm = Matrix(reshape(t, (1, n)))
  Gm = Matrix([ones(n, 1) gm'])
  GmT = Gm'
  mv = inv(GmT*Gm)*(GmT*d')
  σ = mv[2]

  return σ
end


"""
    energies(vars, params, grid, sol)
    energies(prob)
Return the kinetic energy KE, and the potential energy PE.
The kinetic energy is
```math
𝖪𝖤 = \\int \\frac{1}{2} |{\\bf ∇} ψ|^2 dx dy = \\frac{1}{2} \\sum_{𝐤} |𝐤|² |ψ̂|²,
```
while the potential energy that corresponds to the interface (i.e., the interface between the active and inert layers) is
```math
𝖯𝖤 = \\int \\frac{1}{2} F ψ^2 dx dy = \\frac{1}{2} F \\sum_{𝐤} |ψ|².
```
"""
function energies(vars, params::NondimLinearOneHalfLayerQGParams, grid, sol)
  KE, PE = zeros(1), zeros(1)

  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

  abs²∇𝐮h = vars.uh        # use vars.uh as scratch variable
  @. abs²∇𝐮h = grid.Krsq * abs2(vars.ψh)

  KE[1] = @views 0.5 * parsevalsum(abs²∇𝐮h, grid)
  PE[1] = @views 0.5 * params.F * parsevalsum(abs2.(vars.ψh), grid)

  return KE, PE
end


energies(prob) = energies(prob.vars, prob.params, prob.grid, prob.sol)


end # module
