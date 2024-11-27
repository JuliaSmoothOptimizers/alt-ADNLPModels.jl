module ADNLPModelsZygoteExt

using Zygote, ADNLPModels

function gradient(::ZygoteADGradient, f, x)
  g = Zygote.gradient(f, x)[1]
  return g === nothing ? zero(x) : g
end
function gradient!(::ZygoteADGradient, g, f, x)
  _g = Zygote.gradient(f, x)[1]
  g .= _g === nothing ? 0 : _g
end

function Jprod!(::ZygoteADJprod, Jv, f, x, v, ::Val)
  Jv .= vec(Zygote.jacobian(t -> f(x + t * v), 0)[1])
  return Jv
end

function Jtprod!(::ZygoteADJtprod, Jtv, f, x, v, ::Val)
  g = Zygote.gradient(x -> dot(f(x), v), x)[1]
  if g === nothing
    Jtv .= zero(x)
  else
    Jtv .= g
  end
  return Jtv
end

function jacobian(::ZygoteADJacobian, f, x)
  return Zygote.jacobian(f, x)[1]
end

function hessian(b::ZygoteADHessian, f, x)
  return jacobian(
    ForwardDiffADJacobian(length(x), f, x0 = x),
    x -> gradient(ZygoteADGradient(), f, x),
    x,
  )
end

end
