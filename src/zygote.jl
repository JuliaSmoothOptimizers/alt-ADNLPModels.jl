struct ZygoteADGradient <: ADBackend end
struct ZygoteADJprod <: ImmutableADbackend end
struct ZygoteADJtprod <: ImmutableADbackend end
struct ZygoteADJacobian <: ImmutableADbackend
  nnzj::Int
end
struct ZygoteADHessian <: ImmutableADbackend
  nnzh::Int
end

# See https://fluxml.ai/Zygote.jl/latest/limitations/
function get_immutable_c(nlp::ADModel)
  function c(x; nnln = nlp.meta.nnln)
    c = Zygote.Buffer(x, nnln)
    nlp.c!(c, x)
    return copy(c)
  end
  return c
end
get_c(nlp::ADModel, ::ImmutableADbackend) = get_immutable_c(nlp)

function get_immutable_F(nls::AbstractADNLSModel)
  function F(x; nequ = nls.nls_meta.nequ)
    Fx = Zygote.Buffer(x, nequ)
    nls.F!(Fx, x)
    return copy(Fx)
  end
  return F
end
get_F(nls::AbstractADNLSModel, ::ImmutableADbackend) = get_immutable_F(nls)

function ZygoteADGradient(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return ZygoteADGradient()
end

function ZygoteADJprod(
  nvar::Integer,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return ZygoteADJprod()
end

function ZygoteADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return ZygoteADJtprod()
end

function ZygoteADJacobian(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  @assert nvar > 0
  nnzj = nvar * ncon
  return ZygoteADJacobian(nnzj)
end

function ZygoteADHessian(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  @assert nvar > 0
  nnzh = nvar * (nvar + 1) / 2
  return ZygoteADHessian(nnzh)
end
