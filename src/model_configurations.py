from function_transformer_attention import ODEFuncTransformerAtt
from function_GAT_attention import ODEFuncAtt
from function_laplacian_diffusion import LaplacianODEFunc
from block_transformer_attention import AttODEblock
from block_constant import ConstantODEblock

from function_beltrami_trans import ODEFuncBektramiAtt

from function_beltrami_fa import ODEFuncBelFA

from function_laplacian_convection import ODEFuncLapCONV
from function_beltrami_convection import ODEFuncBeltramiCONV

from function_GAT_convection import ODEFuncAttConv
from function_beltrami_gat import ODEFuncBeltramiGAT

from function_transformer_convection import ODEFuncTransConv

from function_beltramitrans_convection import ODEFuncBeltramiTRANSCONV


class BlockNotDefined(Exception):
  pass

class FunctionNotDefined(Exception):
  pass


def set_block(opt):
  ode_str = opt['block']
  if ode_str == 'attention':
    block = AttODEblock


  elif ode_str == 'constant':
    block = ConstantODEblock

  else:
    raise BlockNotDefined
  return block


def set_function(opt):
  ode_str = opt['function']
  if ode_str == 'laplacian':
    f = LaplacianODEFunc
  elif ode_str == 'GAT':
    f = ODEFuncAtt
  elif ode_str == 'transformer':
    f = ODEFuncTransformerAtt
  elif ode_str == 'beltrami':
    f = ODEFuncBektramiAtt
 

  elif ode_str == 'lapconv':
    f = ODEFuncLapCONV
  elif ode_str == 'belconv':
    f = ODEFuncBeltramiCONV

  elif ode_str == 'gatconv':
    f = ODEFuncAttConv
  elif ode_str == 'belgat':
    f = ODEFuncBeltramiGAT
  elif ode_str == 'transconv':
    f = ODEFuncTransConv
  elif ode_str == 'beltransconv':
    f = ODEFuncBeltramiTRANSCONV
 


  else:
    raise FunctionNotDefined
  return f
