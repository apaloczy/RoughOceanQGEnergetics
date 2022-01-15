using FourierFlows: stepforward!
using Printf
using NPZ, MAT

include("nondimlineartwolayerqg.jl")
set_q! = NondimLinearTwoLayerQG.set_q!
updatevars! = NondimLinearTwoLayerQG.updatevars!
energies = NondimLinearTwoLayerQG.energies
fluxes = NondimLinearTwoLayerQG.fluxes
get_bickley = NondimLinearTwoLayerQG.get_bickley
get_topo = NondimLinearTwoLayerQG.get_topo
get_σ = NondimLinearTwoLayerQG.get_σ

include("common_params_twolay.jl")

for F₁ in all_F₁s, ht in all_hts

println("")
println("F₁ = ", string(F₁), ", ht = ", string(ht), ", topography = ", ttype)
println("")
F₁, ht = float(F₁), float(ht)

if ttype!="rand"
    h, hx, hy = get_topo(gr, ht, kt, type=ttype)
else
    T = eltype(gr)
    nx, ny = gr.nx, gr.ny
    h, hx, hy = zeros(T, nx, ny), zeros(T, nx, ny), zeros(T, nx, ny)
    mfile = matopen(fname_hrand)
    h = read(mfile, "h")
    close(mfile)
    @. h = h*ht*0.5 # ht is rms in random topography. The factor 1/2 is the rms of sin(x)sin(y).
    hh = rfft(h)
    hxh = @. im * gr.kr * hh
    hyh = @. im * gr.l * hh
    hx = irfft(hxh, nx)
    hy = irfft(hyh, nx)
end


prob = NondimLinearTwoLayerQG.Problem(dev;
                                      nx = N,
                                      ny = N,
                                      Lx = L,
                                      Ly = L,
                                      x0 = 0.0,
                                      F₁ = F₁,
                                       δ = δ,
                                       U = U,
                                     Uyy = Uyy,
                                       h = h,
                                      hx = hx,
                                      hy = hy,
                                       α = α,
                                       β = β,
                                      dt = dt,
                                 stepper = stepper,
                                  effort = effort)

sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
set_q!(prob, q₀)

if PSI_HOVMOLLER thovm, ψ₁hovm, ψ₂hovm = nothing, nothing, nothing; fm = Int(round(N/2)) end
startwalltime = time()
j = 0
R = 1.0
cyc = 0
Eo = energies(prob) # Reference upper layer KE level for renormalization.
ke1o, ke2o, peo = Eo[1][1], Eo[1][2], Eo[2][1]
t = Array([clock.t])
KE1 = Array([ke1o])
KE2 = Array([ke2o])
PE = Array([peo])

while cyc<cycles
  stepforward!(prob)
  updatevars!(prob)
  E = energies(prob)
  ke1, ke2, pe = E[1][1], E[1][2], E[2][1]

  if j % nlog == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])
    log = @sprintf("step: %05d, t: %2.1f, R: %1.2f, cfl: %1.2f, KE₁/KE₂: %1.2f, walltime: %1.2f min", clock.step, clock.t, R, cfl, ke1/ke2, (time()-startwalltime)/60)
    println(log)

    if cyc==cyclesm && PSI_HOVMOLLER
      if isnothing(thovm)
        ψ₁hovm = vars.ψ[:, fm, 1]
        ψ₂hovm = vars.ψ[:, fm, 2]
        thovm = Array([clock.t])
      else
        ψ₁hovm = cat(ψ₁hovm, vars.ψ[:, fm, 1], dims=2)
        ψ₂hovm = cat(ψ₂hovm, vars.ψ[:, fm, 2], dims=2)
        append!(thovm, clock.t)
      end
    end
  end

  append!(t, clock.t)
  append!(KE1, ke1)
  append!(KE2, ke2)
  append!(PE, pe)

  j += 1
  R = ke1o/ke1 # renormalize (KE_{ref}/KE).
  if R<Rthresh
    set_q!(prob, vars.q*R)
    cyc += 1
    println("")
    println("Completed renormalization cycle", " ", cyc, "/", cycles)
    println("")
  end
end

# Calculate bulk metrics after final renormalization.
E, F = energies(prob), fluxes(prob)
ke1, ke2, pe = E[1][1], E[1][2], E[2][1]
keratio = ke1/ke2
mom1, mom2, thick = F[1][1], F[1][2], F[2][1]

# Get growth rate from exponential fit to upper-layer KE time series.
σ = get_σ(t, KE1, KE2, PE)

if SAVE_OUTPUT
  println("Saving output data.")
  tk = Int(kt/π)
  Nstr = string(N)
  F1str = string(Int(F₁))
  htstr = string(Int(ht))

  if ttype=="rand"
      npzname = "lin_N" * Nstr * "_ht" * htstr * "_F1" * F1str * "_" * split(fname_hrand, "/")[end][1:end-4]
  elseif ttype=="slop"
      npzname = "lin_N" * Nstr * "_al" * string(α) * "_F1" * F1str * "_" * ttype
  else
      npzname = "lin_N" * Nstr * "_ht" * htstr * "_F1" * F1str * "_" * ttype * string(tk)
  end

  npzname = "../simulations/" * npzname * ".npz"
  ψ₁, ψ₂ = vars.ψ[:, :, 1], vars.ψ[:, :, 2]
  npzdata = Dict("p1" => ψ₁', "p2" => ψ₂', "thick" => thick, "mom1" => mom1, "mom2" => mom2, "N" => N, "L" => L, "dt" => dt, "F1" => F₁, "U2fac" => U₂fac, "al" => α, "bet" => β, "d12" => δ, "ht" => ht, "tk" => tk, "sigma" => σ, "keratio" => keratio)

  if PSI_HOVMOLLER
    npzdata = merge(npzdata, Dict("p1hovm" => ψ₁hovm', "p2hovm" => ψ₂hovm', "thovm" => thovm))
  end

  npzwrite(npzname, npzdata)
end

end # all_F₁s, all_hts
