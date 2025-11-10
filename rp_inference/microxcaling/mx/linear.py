"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import torch
import torch.nn.functional as F

from .mx_ops import quantize_mx_op
from .elemwise_ops import quantize_elemwise_op
from .specs import apply_mx_specs, get_backwards_mx_specs
from .specs import mx_assert_test

try:
    from scale_utils import hadamard_utils
    import fast_hadamard_transform
except:
    print('hadamard_utils is not imported')
import math

f_linear = F.linear
torch_matmul = torch.matmul

class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        mx_specs=None,
        name=None,
        args=None,
        apply_stats=None,
    ):
        dtype = input.dtype
        # element-wise quantize for input
        bf_in = quantize_elemwise_op(
            input.float(), mx_specs=mx_specs, round=mx_specs["round_output"]
        )

        # element-wise quantize for weight and bias
        bf_weight = quantize_elemwise_op(
            weight.float(), mx_specs=mx_specs, round=mx_specs["round_weight"]
        )

        if bias is not None:
            ctx.has_bias = True
            bf_bias = quantize_elemwise_op(
                bias.float(), mx_specs=mx_specs, round=mx_specs["round_weight"]
            ).to(dtype)
        else:
            ctx.has_bias = False
            bf_bias = None

        if mx_specs["quantize_backprop"]:
            ctx.save_for_backward(bf_in, bf_weight)
        else:
            ctx.save_for_backward(input, weight)

        # MX quantize everything along input size
        qis_input = quantize_mx_op(
            bf_in,
            mx_specs,
            elem_format=mx_specs['a_elem_format'],
            scale_mode=mx_specs['a_scale_mode'],
            axes=[-1],
            round=mx_specs["round_mx_output"],
        )
        qis_weight = quantize_mx_op(
            bf_weight,
            mx_specs,
            elem_format=mx_specs['w_elem_format'],
            scale_mode=mx_specs['w_scale_mode'],
            axes=[-1],
            round=mx_specs["round_mx_output"],
        )
        qis_input = qis_input.to(dtype)
        qis_weight = qis_weight.to(dtype)

        # compute output
        output = f_linear(qis_input, qis_weight)

        output = quantize_elemwise_op(
            output, mx_specs=mx_specs, round=mx_specs["round_output"]
        )

        if bias is not None:
            output = output + bf_bias
            output = quantize_elemwise_op(
                output, mx_specs=mx_specs, round=mx_specs["round_output"]
            )

        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        ctx.name = name

        #============================ Statistics
        if apply_stats:
            with torch.no_grad():
                stats = dict()
                stats['stats_finp_ravg'] = bf_in.mean().item()
                stats['stats_finp_rstd'] = bf_in.std().item()
                stats['stats_qinp_ravg'] = qis_input.mean().item()
                stats['stats_qinp_rstd'] = qis_input.std().item()
                stats['stats_qterr_inp'] = (bf_in-qis_input).norm().item()
                stats['stats_fwei_ravg'] = bf_weight.mean().item()
                stats['stats_fwei_rstd'] = bf_weight.std().item()
                stats['stats_qwei_ravg'] = qis_weight.mean().item()
                stats['stats_qwei_rstd'] = qis_weight.std().item()
                stats['stats_qterr_wei'] = (bf_weight-qis_weight).norm().item()
        else:
            stats = None
        return output, stats

    @staticmethod
    def backward(ctx, grad_output, grad_stat):
        # load context
        input, weight = ctx.saved_tensors

        out_dim = weight.shape[0]
        in_dim = weight.shape[1]

        grad_output = quantize_elemwise_op(
            grad_output,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        #####################################################
        # perform madtile operation for grad_weight, grad_bias
        #####################################################
        # if the input is 2D, quantize everything along examples (batches)
        # if the input is 3D, quantize everything along the first axis
        qex_input = quantize_mx_op(
            input,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format_bp_ex'],
            axes=[-2],
            round=ctx.mx_specs["round_mx_input_grad_weight"],
        )
        qex_grad_output = quantize_mx_op(
            grad_output,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format_bp_ex'],
            axes=[-2],
            round=ctx.mx_specs["round_mx_grad_output_grad_weight"],
        )

        # compute grad_weight [out_features, in_features]
        qex_grad_output = qex_grad_output.reshape(-1, out_dim)
        qex_input = qex_input.reshape(-1, in_dim)

        # Compute grad_weight
        grad_weight = torch_matmul(qex_grad_output.transpose(0, 1), qex_input)
        grad_weight = quantize_elemwise_op(
            grad_weight,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_weight"],
        )

        #####################################################
        # perform madtile operation for grad_input
        #####################################################
        # compute grad_input, quantize everything along output size
        qos_weight = quantize_mx_op(
            weight,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['w_elem_format_bp'],
            axes=[0],
            round=ctx.mx_specs["round_mx_weight_grad_input"],
        )
        # grad_output shape is (B, seq, out_dim)
        qos_grad_output = quantize_mx_op(
            grad_output,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format_bp_os'],
            axes=[-1],
            round=ctx.mx_specs["round_mx_grad_output_grad_input"],
        )

        # Compute grad_input
        grad_input = torch_matmul(qos_grad_output, qos_weight)
        grad_input = quantize_elemwise_op(
            grad_input,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        #####################################################
        # Compute grad_bias
        #####################################################
        if not ctx.has_bias:
            grad_bias = None
        else:
            grad_bias = grad_output.reshape(-1, out_dim).sum(0)
            grad_bias = quantize_elemwise_op(
                grad_bias,
                mx_specs=ctx.mx_specs,
                round=ctx.mx_specs["round_grad_weight"],
            )

        return (grad_input, grad_weight, grad_bias, None, None, None, None)


def linear(
    input,
    weight,
    bias=None,
    mx_specs=None,
    name=None,
    args=None,
    apply_stats=False,
):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_linear(input, weight, bias=bias)

    mx_specs = apply_mx_specs(mx_specs)

    output, stats = LinearFunction.apply(input, weight, bias, mx_specs, name, args, apply_stats)

    return output, stats


class Linear(torch.nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        mx_specs=None,
        name=None,
        args=None,
    ):
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None

        self.name = name
        self.mx_specs = apply_mx_specs(mx_specs)
        super().__init__(in_features, out_features, bias)
        self.args = args
        if self.args.save_stats:
            self.stats_momentum = 0.1
            self.register_backward_hook(self.save_gradient_stats)
            self.register_buffer('stats_output_grad_std', torch.zeros(1))
            self.register_buffer('stats_output_grad_mean', torch.zeros(1))
            self.register_buffer('stats_finp_ravg', torch.zeros(1))
            self.register_buffer('stats_finp_rstd', torch.zeros(1))
            self.register_buffer('stats_qinp_ravg', torch.zeros(1))
            self.register_buffer('stats_qinp_rstd', torch.zeros(1))
            self.register_buffer('stats_qterr_inp', torch.zeros(1))
            self.register_buffer('stats_fwei_ravg', torch.zeros(1))
            self.register_buffer('stats_fwei_rstd', torch.zeros(1))
            self.register_buffer('stats_qwei_ravg', torch.zeros(1))
            self.register_buffer('stats_qwei_rstd', torch.zeros(1))
            self.register_buffer('stats_qterr_wei', torch.zeros(1))

    def save_gradient_stats(self, module, grad_input, grad_output):
        with torch.no_grad():
            self.stats_output_grad_std = grad_output[0].std().data
            self.stats_output_grad_mean = grad_output[0].mean().data

    def apply_mx_specs(self, mx_specs):
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None
        self.mx_specs = apply_mx_specs(mx_specs)

    def append_name(self, postfix):
        self.name += postfix

    def forward(self, inputs):
        if self.mx_none:
            return super().forward(inputs)

        # Hadamard transform (QuaRot)
        if hasattr(self, "online_full_had"):
            inputs = hadamard_utils.matmul_hadU_cuda(inputs, self.had_K, self.K)
        elif hasattr(self, "online_partial_had"):
            init_shape = inputs.shape
            if self.K == 1:
                inputs = fast_hadamard_transform.hadamard_transform(
                    inputs.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim).transpose(1, 2),
                    scale=1/math.sqrt(init_shape[-1]//self.had_dim)
                ).transpose(1, 2)
            else:
                inputs = (self.had_K.to(inputs.dtype).to(inputs.device) @ inputs.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim)) / math.sqrt(init_shape[-1]//self.had_dim)
            inputs = inputs.reshape(init_shape)

        out, stats = linear(
            input=inputs,
            weight=self.weight,
            bias=self.bias,
            mx_specs=self.mx_specs,
            name=self.name,
            args=self.args,
            apply_stats=self.args.save_stats,
        )
    
        if stats is not None:
            momentum = self.stats_momentum
            for k in stats.keys():
                self.state_dict()[k].mul_(momentum).add_(stats[k]*(1-momentum))

        return out
