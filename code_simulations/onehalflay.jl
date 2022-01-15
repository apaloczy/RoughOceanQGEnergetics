using FourierFlows: stepforward!
using Printf
using NPZ

include("nondimlinearonehalflayerqg.jl")
set_q! = NondimLinearOneHalfLayerQG.set_q!
updatevars! = NondimLinearOneHalfLayerQG.updatevars!
energies = NondimLinearOneHalfLayerQG.energies
get_bickley = NondimLinearOneHalfLayerQG.get_bickley
get_σ = NondimLinearOneHalfLayerQG.get_σ

include("common_params_onehalflay.jl")

for F in all_Fs

println("")
println("F = ", string(F))
println("")
F = float(F)


prob = NondimLinearOneHalfLayerQG.Problem(dev;
                                          nx = N,
                                          ny = N,
                                          Lx = L,
                                          Ly = L,
                                          x0 = 0.0,
                                           F = F,
                                           U = U,
                                         Uyy = Uyy,
                                           β = β,
                                          dt = dt,
                                     stepper = stepper,
                                      effort = effort)

sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
set_q!(prob, q₀)

if PSI_HOVMOLLER thovm, ψhovm = nothing, nothing; fm = Int(round(N/2)) end
startwalltime = time()
j = 0
R = 1.0
cyc = 0
Eo = energies(prob) # Reference KE level for renormalization.
keo, peo = Eo[1][1], Eo[2][1]
t = Array([clock.t])
KE = Array([keo])
PE = Array([peo])

while cyc<cycles
  stepforward!(prob)
  updatevars!(prob)
  E = energies(prob)
  ke, pe = E[1][1], E[2][1]

  if j % nlog == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])
    log = @sprintf("step: %06d, t: %2.1f, R: %1.2f, cfl: %1.2f, KE/PE: %1.2f, walltime: %1.2f min", clock.step, clock.t, R, cfl, ke/pe, (time()-startwalltime)/60)
    println(log)

    if cyc==cyclesm && PSI_HOVMOLLER
      if isnothing(thovm)
        ψhovm = vars.ψ[:, fm]
        thovm = Array([clock.t])
      else
        ψhovm = cat(ψhovm, vars.ψ[:, fm], dims=2)
        append!(thovm, clock.t)
      end
    end
  end

  append!(t, clock.t)
  append!(KE, ke)
  append!(PE, pe)

  j += 1
  R = keo/ke # renormalize (KE_{ref}/KE).
  if R<Rthresh
    set_q!(prob, vars.q*R)
    cyc += 1
    println("")
    println("Completed renormalization cycle", " ", cyc, "/", cycles)
    println("")
  end
end

# Get growth rate from exponential fit to KE time series.
σ = get_σ(t, KE, PE)

if SAVE_OUTPUT
  println("Saving output data.")

  npzname = "lin_onehalflay_N" * string(N) * "_F" * string(Int(F))
  npzname = "../simulations/" * npzname * ".npz"
  npzdata = Dict("p" => vars.ψ', "N" => N, "L" => L, "dt" => dt, "bet" => β, "sigma" => σ, "F" => F)

  if PSI_HOVMOLLER
    npzdata = merge(npzdata, Dict("phovm" => ψhovm', "thovm" => thovm))
  end

  npzwrite(npzname, npzdata)
end

end # all_Fs
