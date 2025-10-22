import torch
import torch.nn as nn
from mx import Linear, LayerNorm, matmul
from mx import gelu, simd_split, simd_add
from mx import add_mx_args, get_mx_specs
from mx.specs import MxSpecs
from mx.specs import mx_assert_test
import logging
import transformers

SILENT=True

def parse_mx_specs(args,mtype):
    mx_specs = MxSpecs()
    keys = ['scale_bits','block_size','w_elem_format','a_elem_format','A_elem_format','B_elem_format']
    for key in keys:
        try:
            val = vars(args)[f'{key}_{mtype}']
            mx_specs[key] = val if val!='none' else None
        except:
            logging.info(f'[{mtype}] Set {key} to None')
            mx_specs[key] = None
            continue
    mx_specs['custom_cuda'] = args.custom_cuda
    mx_specs['a_scale_mode'] = args.a_scale_mode
    mx_specs['w_scale_mode'] = args.w_scale_mode
    mx_specs['A_scale_mode'] = args.A_scale_mode
    mx_specs['B_scale_mode'] = args.B_scale_mode
    mx_specs['per_tensor'] = args.per_tensor
    return mx_specs

class MXMatMul(nn.Module):
    def __init__(
        self,
        mx_specs=None,
        axes=[-1,-2],
        args=None,
    ):
        super().__init__()
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None
        self.mx_specs = mx_specs
        self.axes = axes
        self.args = args
        self.custom_bins = False

    def forward(self, A, B):
        out = matmul(A, B, mx_specs=self.mx_specs, mode_config='aa', args=self.args, axes=self.axes)
        return out

class matmul_module(nn.Module):
    def forward(self, A, B):
        return torch.matmul(A,B)

def get_mx_model(model,
        mx_specs_linear=None,
        mx_specs_matmul=None,
        mx_specs_ln=None,
        mx_specs_head=None,
        args=None,
    ):
    model_with_matmul(model, args)
    wrapped_modules={}
    module_dict={}
    
    it=[(name,m) for name,m in model.named_modules()]
    head_name = it[-1][0]
    if not SILENT:
        logging.info(f'[MX Specs] Linear specs: {mx_specs_linear}')
        logging.info(f'[MX Specs] MatMul specs: {mx_specs_matmul}')
        logging.info(f'[MX Specs] LayerNorm specs: {mx_specs_ln}')
        logging.info(f'[MX Specs] Head specs: {mx_specs_head}')
        logging.info(f'[MX Specs] Detected head name: {head_name}')

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
        # Head
        if name==head_name and isinstance(m,nn.Linear):
            idx = idx+1 if idx != 0 else idx
            new_m = Linear(m.in_features,m.out_features,m.bias is not None,mx_specs=mx_specs_head, args=args)
            new_m.weight.data=m.weight.data
            new_m.bias=m.bias
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx:],replace_m)
        # Linear (Excluding LM Head and High-precision LoRA)
        elif isinstance(m,nn.Linear) and 'head' not in name and 'lora' not in name:
            idx = idx+1 if idx != 0 else idx
            new_m = Linear(m.in_features,m.out_features,m.bias is not None,mx_specs=mx_specs_linear, args=args)
            new_m.weight.data=m.weight.data
            new_m.bias=m.bias
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx:],replace_m)
        elif isinstance(m,matmul_module):
            axes = [-1,-2]
            if args.kv_tokenwise:
                if 'matmul2' in name:
                    axes = [-1,-1]
            idx = idx+1 if idx != 0 else idx
            new_m = MXMatMul(mx_specs=mx_specs_matmul, args=args, axes=axes)
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx:],replace_m)

    logging.info(f'[MX Specs] Completed MX spec adaptation')
    for name, module in model.named_modules():
        if hasattr(module, 'mx_specs'):
            logging.info(f"==> {name:50s}: [Wgt] {module.mx_specs['w_elem_format']} / [Act] {module.mx_specs['a_elem_format']} / [A] {module.mx_specs['A_elem_format']} / [B] {module.mx_specs['B_elem_format']} /[Block] {module.mx_specs['block_size']} / [Scale-bits] {module.mx_specs['scale_bits']} ")

    logging.info("[MX Specs] Completed model wrap with MX modules")

from types import MethodType
def model_with_matmul(model, args=None):
    from utils.attention import llama_forward
    from transformers.models.llama.modeling_llama import LlamaAttention
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            setattr(module, "matmul1", matmul_module())
            setattr(module, "matmul2", matmul_module())
            module.forward = MethodType(llama_forward, module)
    logging.info(f'[MX Specs] Replace torch.matmul into custom module')
