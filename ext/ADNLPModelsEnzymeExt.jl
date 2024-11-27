module ADNLPModelsEnzymeExt

using Enzyme, ADNLPModels

function ADNLPModels.gradient!(::ADNLPModels.EnzymeADGradient, g, f, x)
  Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Duplicated(x, g)) # gradient!(Reverse, g, f, x)
  return g
end

end
