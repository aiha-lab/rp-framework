import torch
import torch.nn as nn
from mx import Linear
from mx.specs import MxSpecs, get_mx_specs
from dataclasses import dataclass, field

@dataclass
class QuantizationArguments:
    w_format: str | None = field(default=None, metadata={"help":"MXFP4 supports"})
    a_format: str | None = field(default=None, metadata={"help":"MXFP4 supports"})
    g_format: str | None = field(default=None, metadata={"help":"MXFP4 supports"})

class TempMxSpecs:
    def __init__(self, w_format, a_format, g_format):
        self.w_elem_format=w_format
        self.a_elem_format=a_format
        self.block_size=32
        self.custom_cuda=True

def get_mx_model(model, w_format, a_format, g_format):
    wrapped_modules={}
    module_dict={}

    mx_specs = get_mx_specs(TempMxSpecs(w_format, a_format, g_format))
    # Backward pass
    mx_specs['w_elem_format_bp'] = g_format
    mx_specs['a_elem_format_bp_ex'] = g_format # activation,gradient in bwd pass
    mx_specs['a_elem_format_bp_os'] = g_format # weight*gradient in bwd pass
    mx_specs['quantize_backprop'] = False
    print(mx_specs)
    
    it=[(name,m) for name,m in model.named_modules()]
    head_name = it[-1][0]

    for name,m in it:
        module_dict[name]=m
        idx=name.rfind('.')
        if idx==-1:
            idx=0
        father_name=name[:idx]
        if father_name in module_dict:
            father_module=module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")
        if isinstance(m,nn.Linear) and 'head' not in name:
            idx = idx+1 if idx != 0 else idx
            new_m = Linear(m.in_features,m.out_features,m.bias is not None,mx_specs=mx_specs)
            new_m.weight.data=m.weight.data
            new_m.bias=m.bias
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx:],replace_m)

    print(f'[MX Specs] Completed MX spec adaptation')
    print(model)
