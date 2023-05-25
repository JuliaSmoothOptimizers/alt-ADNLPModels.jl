struct GenericForwardDiffADGradient <: ADBackend end
GenericForwardDiffADGradient(args...; kwargs...) = GenericForwardDiffADGradient()
function gradient!(::GenericForwardDiffADGradient, g, f, x)
  return ForwardDiff.gradient!(g, f, x)
end

struct ForwardDiffADGradient <: ADBackend
  cfg
end
function ForwardDiffADGradient(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  kwargs...,
)
  @assert nvar > 0
  @lencheck nvar x0
  cfg = ForwardDiff.GradientConfig(f, x0)
  return ForwardDiffADGradient(cfg)
end
function gradient!(adbackend::ForwardDiffADGradient, g, f, x)
  return ForwardDiff.gradient!(g, f, x, adbackend.cfg)
end

struct ForwardDiffADJacobian <: ADBackend
  nnzj::Int
end
function ForwardDiffADJacobian(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  @assert nvar > 0
  nnzj = nvar * ncon
  return ForwardDiffADJacobian(nnzj)
end
jacobian(::ForwardDiffADJacobian, f, x) = ForwardDiff.jacobian(f, x)

struct ForwardDiffADHessian <: ADBackend
  nnzh::Int
end
function ForwardDiffADHessian(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  @assert nvar > 0
  nnzh = nvar * (nvar + 1) / 2
  return ForwardDiffADHessian(nnzh)
end
hessian(::ForwardDiffADHessian, f, x) = ForwardDiff.hessian(f, x)

struct GenericForwardDiffADJprod <: ADBackend end
function GenericForwardDiffADJprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return GenericForwardDiffADJprod()
end
function Jprod!(::GenericForwardDiffADJprod, Jv, f, x, v)
  Jv .= ForwardDiff.derivative(t -> f(x + t * v), 0)
  return Jv
end

struct ForwardDiffADJprod{T} <: InPlaceADbackend
  tmp_in::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
  tmp_out::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
end

function ForwardDiffADJprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  tmp_in =
    Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, nvar)
  tmp_out =
    Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, ncon)
  return ForwardDiffADJprod(tmp_in, tmp_out)
end

function Jprod!(b::ForwardDiffADJprod, Jv, c!, x, v)
  SparseDiffTools.auto_jacvec!(Jv, c!, x, v, b.tmp_in, b.tmp_out)
  return Jv
end

struct GenericForwardDiffADJtprod <: ADBackend end
function GeneriForwardDiffADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return GenericForwardDiffADJtprod()
end
function Jtprod!(::GenericForwardDiffADJtprod, Jtv, f, x, v)
  Jtv .= ForwardDiff.gradient(x -> dot(f(x), v), x)
  return Jtv
end

struct ForwardDiffADJtprod{Tag, GT, S} <: ADNLPModels.InPlaceADbackend
  cfg::ForwardDiff.GradientConfig{Tag}
  ψ::GT
  temp::S
  sol::S
end

function ForwardDiffADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}

  temp = similar(x0, nvar + 2 * ncon)
  sol = similar(x0, nvar + 2 * ncon)

  function ψ(z; nvar = nvar, ncon = ncon)
    cx, x, u = view(z, 1:ncon), view(z, (ncon + 1):(nvar + ncon)), view(z, (nvar + ncon + 1):(nvar + ncon + ncon))
    c!(cx, x)
    dot(cx, u)
  end
  tagψ = ForwardDiff.Tag(ψ, T)
  cfg = ForwardDiff.GradientConfig(ψ, temp, ForwardDiff.Chunk(temp), tagψ)

  return ForwardDiffADJtprod(cfg, ψ, temp, sol)
end

function ADNLPModels.Jtprod!(b::ForwardDiffADJtprod{Tag, GT, S}, Jtv, c!, x, v) where {Tag, GT, S}
  ncon = length(v)
  nvar = length(x)

  b.sol[1:ncon] .= 0
  b.sol[(ncon + 1):(ncon + nvar)] .= x
  b.sol[(ncon + nvar + 1):(2 * ncon + nvar)] .= v
  ForwardDiff.gradient!(b.temp, b.ψ, b.sol, b.cfg)
  Jtv .= view(b.temp, (ncon + 1):(nvar + ncon))
  return Jtv
end

struct GenericForwardDiffADHvprod <: ADBackend end
function GenericForwardDiffADHvprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return GenericForwardDiffADHvprod()
end
function Hvprod!(::GenericForwardDiffADHvprod, Hv, f, x, v)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(f, x + t * v), 0)
  return Hv
end

struct ForwardDiffADHvprod{T} <: ADBackend
  tmp_in::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
  tmp_out::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
end
function ForwardDiffADHvprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  tmp_in =
    Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, nvar)
  tmp_out =
    Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, nvar)
  return ForwardDiffADHvprod(tmp_in, tmp_out)
end

function Hvprod!(b::ForwardDiffADHvprod, Hv, f, x, v)
  ϕ!(dy, x; f = f) = ForwardDiff.gradient!(dy, f, x)
  SparseDiffTools.auto_hesvecgrad!(Hv, (dy, x) -> ϕ!(dy, x), x, v, b.tmp_in, b.tmp_out)
  return Hv
end

struct ForwardDiffADGHjvprod <: ADBackend end
function ForwardDiffADGHjvprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return ForwardDiffADGHjvprod()
end
function directional_second_derivative(::ForwardDiffADGHjvprod, f, x, v, w)
  return ForwardDiff.derivative(t -> ForwardDiff.derivative(s -> f(x + s * w + t * v), 0), 0)
end
