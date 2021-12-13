module NondimLinearTwoLayerQG

export
  Problem,
  set_q!,
  updatevars!,
  get_topo,
  get_bickley,
  get_σ,
  streamfunctionfrompv!,
  energies,
  fluxes,
  specfluxes

using
  FFTW,
  Reexport,
  DocStringExtensions

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!
using FourierFlows: parsevalsum, plan_flows_rfft
import FFTW: rfft


"""
    Problem(dev::Device=CPU();
                nx = 256,
                ny = nx,
                Lx = 2.0,
                Ly = Lx,
                x0 = 0.0,
                F₁ = 25.0,
                 δ = 0.25,
                 U = nothing,
               Uyy = nothing,
                 h = nothing,
                hx = nothing,
                hy = nothing,
                 α = 0.0,
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
  - `F₁`: Upper-layer inverse Burger number.
  - `δ`: Upper/lower layer rest thickness ratio.
  - `U`: Imposed zonal flow U(y) in each layer.
  - `Uyy`: Meridional relative vorticity PV gradient in each layer.
  - `h`: Nondimensional topographic PV.
  - `hx`: Zonal small-scale (roughness) topographic PV gradient.
  - `hy`: Meridional small-scale (roughness) topographic PV gradient.
  - `α`: Meridional linear topographic PV gradient.
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
                  F₁ = 25.0,     # upper-layer inverse Burger number
                   δ = 0.25,     # upper/lower layer rest thickness ratio
                   U = nothing,  # imposed zonal flow U(y) in each layer
                 Uyy = nothing,  # meridional relative vorticity PV gradient in each layer
                   h = nothing,  # nondimensional topographic PV
                  hx = nothing,  # zonal small-scale (roughness) topographic PV gradient
                  hy = nothing,  # meridional small-scale (roughness) topographic PV gradient
                   α = 0.0,      # meridional linear topographic PV gradient
                   β = 0.0,      # meridional linear planetary PV gradient
    # Timestepper and equation options
                  dt = 2e-3,
             stepper = "RK4",
              effort = FFTW.MEASURE,
  # Float type and dealiasing
    aliased_fraction = 0.0,
                   T = Float64)

    # The grid.
    grid = TwoDGrid(dev, nx, Lx, ny, Ly; x0=x0, aliased_fraction=aliased_fraction, T=T)

    # No mean flow case.
    U === nothing && (U = zeros(dev, T, (2)); Uyy = zeros(dev, T, (2)))

    # Flat bottom case.
    h === nothing && (h = zeros(dev, T, (nx, ny)); hx = zeros(dev, T, (nx, ny)); hy = zeros(dev, T, (nx, ny)))

    params = Params(F₁, δ, U, Uyy, h, hx, hy, α, β, grid, effort=effort, dev=dev)
    vars = Vars(dev, grid)
    equation = Equation(dev, params, grid)

    FourierFlows.Problem(equation, stepper, dt, grid, vars, params, dev)
end


abstract type NondimLinearTwoLayerQGParams <: AbstractParams end

"""
    Params{T, Aphys3D, Aphys2D}(F₁, δ, U, Uyy, h, hx, hy, α, β, grid, δ₁, δ₂, F₂, F, Qx, Qy, a, b, c, d, Δ)
A struct containing the parameters for the NondimLinearTwoLayerQG problem. Included are:
$(TYPEDFIELDS)
"""
struct Params{T, Aphys3D, Aphys2D, Atrans2D, Trfft} <: NondimLinearTwoLayerQGParams
    # prescribed params
    "upper layer inverse Burger number"
        F₁ :: T
    "ratio between rest thicknesses of upper and lower layers"
         δ :: T
    "background zonal flows U(y) in each layer"
         U :: Aphys3D
    "background PV gradients due to zonal flow's lateral shear in each layer"
       Uyy :: Aphys3D
    "background topographic PV (bottom roughness, i.e., without large-scale linear slopes)"
         h :: Aphys2D
    "background potential vorticity gradient in x-direction due to hx in lower layer"
        hx :: Aphys2D
     "background potential vorticity gradient in y-direction due to hy in lower layer"
        hy :: Aphys2D
     "topographic vorticity y-gradient (in the y direction, due to a linear bottom slope)"
         α :: T
     "planetary vorticity gradient (in the y direction)"
         β :: T

    # derived params
    "upper layer fractional resting thickness with respect to total depth"
        δ₁ :: T
    "lower layer fractional resting thickness with respect to total depth"
        δ₂ :: T
    "lower layer inverse Burger number"
        F₂ :: T
    "full-depth inverse Burger number"
         F :: T
    "background potential vorticity gradient in x-direction due to hx in lower layer"
        Qx :: Aphys3D
    "background potential vorticity gradient in y-direction due to hy, α, β, U, in each layer"
        Qy :: Aphys3D
    "elements of the streamfunction-PV inversion matrix"
         a :: Atrans2D
         b :: Atrans2D
         c :: Atrans2D
         d :: Atrans2D
    "determinant of streamfunction-PV inversion matrix"
         Δ :: Atrans2D
    "rfft plan for FFTs"
  rfftplan :: Trfft
end


function convert_U_to_U3D(dev, grid, U::AbstractArray{TU, 1}) where TU
  T = eltype(grid)
  if length(U) == 2
    U_2D = zeros(dev, T, (1, 2))
    U_2D[:] = U
    U_2D = repeat(U_2D, outer=(grid.ny, 1))
  else
    U_2D = zeros(dev, T, (grid.ny, 1))
    U_2D[:] = U
  end
  U_3D = zeros(dev, T, (1, grid.ny, 2))
  @views U_3D[1, :, :] = U_2D
  return U_3D
end


function convert_U_to_U3D(dev, grid, U::AbstractArray{TU, 2}) where TU
  T = eltype(grid)
  U_3D = zeros(dev, T, (1, grid.ny, 2))
  @views U_3D[1, :, :] = U
  return U_3D
end


function Params(F₁, δ, U, Uyy, h, hx, hy, α, β, grid;
                effort=FFTW.MEASURE, dev::Device=CPU()) where TU
    T = eltype(grid)
    A = ArrayType(dev)

    δ₁ = δ/(1 + δ)
    δ₂ = 1/(1 + δ)
    F₂ = δ*F₁
    F = δ₁*F₁

    U = convert_U_to_U3D(dev, grid, U)
    Uyy = convert_U_to_U3D(dev, grid, Uyy)

    ny, nx = grid.ny, grid.nx
    Krsq = grid.Krsq

    Qx = zeros(dev, T, (nx, ny, 2))
    Qx[:, :, 2] = hx

    Qy = zeros(dev, T, (nx, ny, 2))
    @. Qy[:, :, 1] = F₁*(U[:, :, 1] - U[:, :, 2]) - Uyy[:, :, 1] + β
    @. Qy[:, :, 2] = F₂*(U[:, :, 2] - U[:, :, 1]) - Uyy[:, :, 2] + β + α + hy

    a = @. - Krsq - F₁
    b = @. Krsq*0 + F₁
    c = @. Krsq*0 + F₂
    d = @. - Krsq - F₂
    Δ = @. a*d - b*c
    Δ[1, 1] = 1e10

    rfftplanlayered = plan_flows_rfft(A{T, 3}(undef, grid.nx, grid.ny, 2), [1, 2]; flags=effort)

    return Params(F₁, δ, U, Uyy, h, hx, hy, α, β, δ₁, δ₂, F₂, F, Qx, Qy, a, b, c, d, Δ, rfftplanlayered)
end


"""
    Equation(params::NondimLinearTwoLayerQGParams, grid)
Return the `equation` for a nondimensional two-layer QG problem with `params` and `grid`. Linear operator
``L`` is empty in this case (no hypo-/hyer-viscosity, and β terms are absorbed in the background PV gradient terms.)
Nonlinear term is computed via `calcN!` function.
"""
function Equation(dev, params, grid)
  T = eltype(grid)
  L = zeros(dev, T, (grid.nkr, grid.nl, 2))

  return FourierFlows.Equation(L, calcN!, grid)
end


abstract type NondimLinearTwoLayerQGVars <: AbstractVars end

"""
    Vars{Aphys, Atrans, F, P}(q, ψ, u, v, qh, ψh, uh, vh, prevsol)
The variables for NondimLinearTwoLayer QG:
$(FIELDS)
"""
struct Vars{Aphys, Atrans, P} <: NondimLinearTwoLayerQGVars
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
Return the `vars` for for the NondimLinearTwoLayerQG problem on device `dev` and
with `grid`.
"""
function Vars(::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny, 2) q ψ u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, 2) qh ψh uh vh prevsol

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
    invtransform!(vars.q, deepcopy(vars.qh), params)
    invtransform!(vars.u, deepcopy(vars.uh), params)
    invtransform!(vars.v, deepcopy(vars.vh), params)
    invtransform!(qx, qxh, params)

    # use vars.u, vars.v, vars.q, vars.uh, vars.vh, vars.qh as scratch variables
    rhs1, rhs1h = deepcopy(vars.u), deepcopy(vars.uh)
    rhs2, rhs2h = deepcopy(vars.v), deepcopy(vars.vh)
    rhs3, rhs3h = deepcopy(vars.q), deepcopy(vars.qh)
    rhs1 = @. params.U*qx
    rhs2 = @. vars.v*params.Qy
    rhs3 = @. vars.u*params.Qx
    fwdtransform!(rhs1h, rhs1, params) # \hat{U*qx}
    fwdtransform!(rhs2h, rhs2, params) # \hat{v*Qy}
    fwdtransform!(rhs3h, rhs3, params) # \hat{u*hx}

    @. N = - (rhs1h + rhs2h + rhs3h)
    return nothing
end


"""
    streamfunctionfrompv!(ψh, qh, params::NondimLinearTwoLayerQGParams, grid)
Invert the PV to obtain the Fourier transform of the streamfunction `ψh`.
"""
function streamfunctionfrompv!(ψh, qh, params::NondimLinearTwoLayerQGParams, grid)
  a, b, c, d, Δ = params.a, params.b, params.c, params.d, params.Δ
  q1h, q2h = view(qh, :, :, 1), view(qh, :, :, 2)

  @views @. ψh[:, :, 1] = (  d * q1h - b * q2h) / Δ
  @views @. ψh[:, :, 2] = (- c * q1h + a * q2h) / Δ

  return nothing
end


# ----------------
# Helper functions
# ----------------


"""
    fwdtransform!(varh, var, params)
Compute the Fourier transform of `var` and store it in `varh`.
"""
fwdtransform!(varh, var, params::AbstractParams) = mul!(varh, params.rfftplan, var)


"""
    invtransform!(var, varh, params)
Compute the inverse Fourier transform of `varh` and store it in `var`.
"""
invtransform!(var, varh, params::AbstractParams) = ldiv!(var, params.rfftplan, varh)


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

  invtransform!(vars.q, deepcopy(vars.qh), params)
  invtransform!(vars.ψ, deepcopy(vars.ψh), params)
  invtransform!(vars.u, deepcopy(vars.uh), params)
  invtransform!(vars.v, deepcopy(vars.vh), params)

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
  fwdtransform!(vars.qh, A(q), params)
  @. vars.qh[1, 1, :] = 0
  @. sol = vars.qh
  updatevars!(vars, params, grid, sol)

  return nothing
end

set_q!(prob, q) = set_q!(prob.sol, prob.params, prob.vars, prob.grid, q)


"""
    get_topo(grid, ht, kt, type="cosi", α=0.0)
Get a monchromatic bottom topography of type `type`, nondimensional height `ht`
and wavenumber `kt` on grid `grid`.
"""
function get_topo(grid, ht, kt; type="cosi", α=0.0)
    T = eltype(grid)
    nx, ny = grid.nx, grid.ny
    h, hx, hy = zeros(T, nx, ny), zeros(T, nx, ny), zeros(T, nx, ny)
    x, y = gridpoints(grid)

    if type=="cosi"
        @. h = ht*cos(kt*x)*cos(kt*y)
        @. hx = - kt*ht*sin(kt*x)*cos(kt*y)
        @. hy = - kt*ht*cos(kt*x)*sin(kt*y)
    elseif type=="ridg"
        @. h = ht*cos(kt*y)
        @. hx = 0.0
        @. hy = - kt*ht*sin(kt*y)
    elseif type=="slop"
        @. h = α*y
        @. hx = 0.0
        @. hy = α
    end

    return h, hx, hy
end


"""
    get_bickley(grid, U₀, Lj; U₂fac=0.0)
Get a Bickley jet of half-width `Lj`, upper-layer amplitude `U₀` and lower-layer
amplitude `U₀*U₂fac` on grid `grid`.
"""
function get_bickley(grid, U₀, Lj; U₂fac=0.0)
    T = eltype(grid)
    y, ny = grid.y, grid.ny
    U, Uyy = zeros(T, ny, 2), zeros(T, ny, 2)

    @. U[:, 1] = U₀*sech(y/Lj)^2
    @. U[:, 2] = U[:, 1]*U₂fac
    @. Uyy[:, 1] = (U₀*2/Lj^2)*U[:, 1]*(3*tanh(y/Lj)^2 - 1)
    @. Uyy[:, 2] = Uyy[:, 1]*U₂fac

    return U, Uyy
end


"""
    get_σ(t, KE1, KE2, PE)
Estimate the growth rate of instabilities from a least-squares fit to energy
histories. `KE1`, `KE2` are respectively the upper- and lower-layer
kinetic energies and `PE` is the potential energy.
"""
function get_σ(t, KE1, KE2, PE)
  f = findlast(diff(KE1 + KE2 + PE).<-0.5) + 1 # Get first time index from the last renormalization cycle.
  t, KE1 = t[f:end], KE1[f:end]
  n = size(t)[1]
  d = Matrix(reshape(log.(KE1), (1, n)))
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
Return the kinetic energy of each fluid layer KE``_1, `` KE``_2``, and the
potential energy of the fluid interface PE.
The kinetic energy at the ``j``-th fluid layer is
```math
𝖪𝖤_j = δ_j \\int \\frac{1}{2} |{\\bf ∇} ψ_j|^2 dx dy = \\frac{1}{2} \\frac{H_j}{H} \\sum_{𝐤} |𝐤|² |ψ̂_j|², \\ j = 1, ..., n ,
```
while the potential energy that corresponds to the interface (i.e., the interface between the lower and upper layers) is
```math
𝖯𝖤 = \\int \\frac{1}{2} F (ψ_1 - ψ_2)^2 dx dy = \\frac{1}{2} F \\sum_{𝐤} |ψ_1 - ψ_2|².
```
"""
function energies(vars, params::NondimLinearTwoLayerQGParams, grid, sol)
  KE, PE = zeros(2), zeros(1)

  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

  abs²∇𝐮h = vars.uh        # use vars.uh as scratch variable
  @. abs²∇𝐮h = grid.Krsq * abs2(vars.ψh)

  ψ1h, ψ2h = view(vars.ψh, :, :, 1), view(vars.ψh, :, :, 2)

  KE[1] = @views 0.5 * parsevalsum(abs²∇𝐮h[:, :, 1], grid) * params.δ₁
  KE[2] = @views 0.5 * parsevalsum(abs²∇𝐮h[:, :, 2], grid) * params.δ₂
  PE[1] = @views 0.5 * params.F * parsevalsum(abs2.(ψ2h .- ψ1h), grid)

  return KE, PE
end


energies(prob) = energies(prob.vars, prob.params, prob.grid, prob.sol)


"""
    fluxes(vars, params, grid, sol)
    fluxes(prob)
Return the lateral eddy fluxes within each fluid layer, lateralfluxes``_1,2``
and also the vertical eddy flux at the fluid interface, verticalflux.

The lateral eddy fluxes within the ``j``-th fluid layer are
```math
\\textrm{lateralfluxes}_j = δ_j \\int U_j v_j ∂_y u_j
dx dy , \\  j = 1, 2,
```
while the vertical eddy flux at the fluid interface (i.e., interface between the upper and lower layers) is
```math
\\textrm{verticalflux} = \\int F (U_1 - U_2) \\,
v_2 ψ_1 dx dy.
```
"""
function fluxes(vars, params::NondimLinearTwoLayerQGParams, grid, sol)
  lateralfluxes, verticalflux = zeros(2), zeros(1)

  updatevars!(vars, params, grid, sol)

  ∂u∂yh = vars.uh           # use vars.uh as scratch variable
  ∂u∂y  = vars.u            # use vars.u  as scratch variable

  @. ∂u∂yh = im * grid.l * vars.uh
  invtransform!(∂u∂y, ∂u∂yh, params)

  lateralfluxes = (sum(@. params.U * vars.v * ∂u∂y; dims=(1, 2)))[1, 1, :]
  lateralfluxes[1] *= params.δ₁
  lateralfluxes[2] *= params.δ₂
  lateralfluxes *= grid.dx * grid.dy

  U₁, U₂ = view(params.U, :, :, 1), view(params.U, :, :, 2)
  ψ₁ = view(vars.ψ, :, :, 1)
  v₂ = view(vars.v, :, :, 2)

  verticalflux = sum(@views @. params.F * (U₁ - U₂) * v₂ * ψ₁; dims=(1, 2))
  verticalflux *= grid.dx * grid.dy

  return lateralfluxes, verticalflux
end


fluxes(prob) = fluxes(prob.vars, prob.params, prob.grid, prob.sol)


"""
    specfluxes(vars, params, grid, sol)
    specfluxes(prob)
Return the lateral, vertical and topographic spectral fluxes. The lateral energy
fluxes are within each fluid layer, lateralspecfluxes``_1,2``. The vertical eddy
flux at the fluid interface is verticalspecflux, and the spectral topographic
energy transfer term in the lower layer is topographicspecflux. Hats indicate
Fourier transforms in the ``x``-direction, and stars in dicate complex
conjugation.

The lateral eddy fluxes within the ``j``-th fluid layer are
```math
\\textrm{lateralfluxes}_j = i k δ_j \\int U_j (\\hat{ψ_j}^\\star \\hat{∂_y u_j} - \\hat{ψ_j} \\hat{∂_y u_j}^\\star)
dk, \\  j = 1, 2,
```
while the vertical eddy flux at the fluid interface (i.e., interface between the upper and lower layers) is
```math
\\textrm{verticalflux} = i k \\int F (U_1 - U_2) \\,
(\\hat{ψ_1} \\hat{ψ_2}^\\star - \\hat{ψ_1}^\\star \\hat{ψ_2}) dk.
```
and the topographic flux in the lower layer is
```math
\\textrm{topograpicflux} = \\int \\hat{u_2}^\\star \\hat{v_2 h} + \\hat{u_2} \\hat{v_2 h}^\\star \\,
+ i k (\\hat{ψ_2}^\\star \\hat{u_2 h} - \\hat{ψ_2} \\hat{u_2 h}^\\star) dk.
```
"""
function specfluxes(vars, params::NondimLinearTwoLayerQGParams, grid, sol)
  lateralspecfluxes = zeros(grid.nkr, 2)
  verticalspecflux, topographicspecflux = zeros(grid.nkr), zeros(grid.nkr)

  updatevars!(vars, params, grid, sol)
  U₁, U₂ = view(params.U, :, :, 1), view(params.U, :, :, 2)
  u₂, v₂ = view(vars.u, :, :, 2), view(vars.v, :, :, 2)

  ∂u∂yh = vars.uh           # use vars.uh as scratch variable
  ∂u∂y  = vars.u            # use vars.u  as scratch variable
  @. ∂u∂yh = im * grid.l * vars.uh
  invtransform!(∂u∂y, ∂u∂yh, params)

  ∂u∂yhx = rfft(∂u∂y, 1)       # FFT{∂u∂y} in x-direction.
  ψhx = rfft(vars.ψ, 1)        # FFT{ψ} in x-direction.
  u₂hx = rfft(u₂, 1)           # FFT{u₂} in x-direction.
  u₂hhx = rfft(u₂*params.h, 1) # FFT{u₂h} in x-direction.
  v₂hhx = rfft(v₂*params.h, 1) # FFT{v₂h} in x-direction.
  ψ₁hx, ψ₂hx = view(ψhx, :, :, 1), view(ψhx, :, :, 2)

  # Lateral (barotropic) energy fluxes.
  auxCMh = @. im * grid.kr * params.U * (conj(ψhx)*∂u∂yhx - ψhx*conj(∂u∂yhx))
  auxCMh[:, :, 1] *= params.δ₁
  auxCMh[:, :, 2] *= params.δ₂
  # Vertical (baroclinic) energy flux.
  auxCTh = @. im * grid.kr * params.F * (U₁ - U₂) * (ψ₁hx*conj(ψ₂hx) - conj(ψ₁hx)*ψ₂hx)
  # Topographic energy flux in the lower layer (already on the RHS).
  auxCtopoh = @. conj(u₂hx)*v₂hhx + u₂hx*conj(v₂hhx) + im * grid.kr * (conj(ψ₂hx)*u₂hhx - ψ₂hx*conj(u₂hhx))

  # Integrate in y-direction and move to RHS.
  lateralspecfluxes = - real((sum(auxCMh, dims=2))[:, 1, :]) * grid.dy
  verticalspecflux = - real(sum(auxCTh, dims=2)) * grid.dy
  topographicspecflux = real(sum(auxCtopoh, dims=2)) * grid.dy # already on the RHS.

  return lateralspecfluxes, verticalspecflux, topographicspecflux
end


specfluxes(prob) = specfluxes(prob.vars, prob.params, prob.grid, prob.sol)


end # module
