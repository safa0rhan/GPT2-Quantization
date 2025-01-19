import torch
import torch.nn as nn
import numpy as np
from my_int8_mm import int8_matmul


class QuantLinear(nn.Module):
    def __init__(self, float_weight: torch.Tensor, bias: torch.Tensor or None,
                 w_scale: float, a_scale: float = None):
        """
        Args:
            float_weight: the original FP32 weight. We'll convert this to int8.
            bias: the original FP32 bias (optional).
            w_scale: scale factor for weight (symmetric).
            a_scale: scale factor for activation (symmetric).
                     If None, we do not quantize activation.
        """
        super().__init__()
        self.out_features = float_weight.size(0)
        self.in_features  = float_weight.size(1)
        
        self.register_buffer("w_scale", torch.tensor([w_scale], dtype=torch.float32))

        if a_scale is not None:
            self.register_buffer("a_scale", torch.tensor([a_scale], dtype=torch.float32))
        else:
            self.a_scale = None

        w_int8 = torch.round(float_weight / w_scale).clamp(-128, 127).to(torch.int8)
        # Store as a parameter or buffer
        self.register_parameter("weight_int8", nn.Parameter(w_int8, requires_grad=False))

        if bias is not None:
            self.register_parameter("bias", nn.Parameter(bias.clone().detach(), requires_grad=False))
        else:
            self.bias = None

    def forward(self, x):
        shape_original = x.shape
        if len(shape_original) == 3:
            B, T, E = shape_original
            x = x.reshape(B*T, E)

        if self.a_scale is not None:
            x_int8 = torch.round(x / self.a_scale).clamp(-128, 127).to(torch.int8)
        else:
            x_int8 = x.to(torch.int8)

        w_for_kernel = self.weight_int8.transpose(0,1)

        out_int32_2d = int8_matmul.int8_gemm_forward(x_int8, w_for_kernel)

        out_fp_2d = out_int32_2d.float() * (self.w_scale * (self.a_scale or 1.0))

        if len(shape_original) == 3:
            out_fp_3d = out_fp_2d.view(B, T, self.out_features)
        else:
            out_fp_3d = out_fp_2d

        if self.bias is not None:
            out_fp_3d = out_fp_3d + self.bias

        return out_fp_3d


def compute_weight_scale_symmetric(weight_tensor):
    # min_val = weight_tensor.min().item()
    # max_val = weight_tensor.max().item()
    # We do symmetric => range = max(|min_val|, |max_val|)
    absmax = weight_tensor.abs().max().item()
    scale = absmax / 127.0
    # zero_point = 0 for symmetric
    return scale

def collect_linear_weight_scales(model, prefix="", weight_scales=None):
    """
    Recursively gather weight scales for each nn.Linear in the model,
    storing them in `weight_scales` using a full dotted path.
    """
    if weight_scales is None:
        weight_scales = {}

    for child_name, child_module in model.named_children():
        full_name = prefix + "." + child_name if prefix else child_name
        
        if isinstance(child_module, nn.Linear):
            scale = compute_weight_scale_symmetric(child_module.weight.data)
            weight_scales[full_name] = scale
        else:
            # Recurse
            collect_linear_weight_scales(child_module, prefix=full_name, weight_scales=weight_scales)

    return weight_scales


def create_quant_linear_from_linear(linear: nn.Linear):
    w_scale = compute_weight_scale_symmetric(linear.weight.data)
    bias = linear.bias.data if linear.bias is not None else None
    qlinear = QuantLinear(linear.weight.data, bias, w_scale, a_scale=None)
    return qlinear

def collect_activation_stats(model, dataloader):
    """
    We'll register forward hooks on each layer we want to quantize,
    record min/max of the activation, then compute scale afterwards.
    """
    act_mins = {}
    act_maxs = {}

    def hook_fn(m, inp, out):
        tensor = out if isinstance(out, torch.Tensor) else out[0]
        # track min/max
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        if m not in act_mins:
            act_mins[m] = min_val
            act_maxs[m] = max_val
        else:
            act_mins[m] = min(min_val, act_mins[m])
            act_maxs[m] = max(max_val, act_maxs[m])

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for input_ids in dataloader:
            _ = model(input_ids)

    return act_mins, act_maxs

def compute_activation_scale_symmetric(min_val, max_val):
    absmax = max(abs(min_val), abs(max_val))
    return absmax / 127.0


def replace_linear_with_quantlinear(module, weight_scales, activation_scales, prefix=""):
    for child_name, child_module in module.named_children():
        full_name = prefix + "." + child_name if prefix else child_name
        
        replace_linear_with_quantlinear(child_module, weight_scales, activation_scales, prefix=full_name)
        
        if isinstance(child_module, nn.Linear):
            w_scale = weight_scales[full_name]

            a_scale = None
            if activation_scales is not None and full_name in activation_scales:
                a_scale = activation_scales[full_name]

            bias = child_module.bias
            if bias is not None:
                bias = bias.data

            qlinear = QuantLinear(
                float_weight=child_module.weight.data,
                bias=bias,
                w_scale=w_scale,
                a_scale=a_scale
            )
            
            setattr(module, child_name, qlinear)
