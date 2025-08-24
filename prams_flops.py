import math
import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
import model.tulip as tulip
from main_lidar_upsampling import get_args_parser
from typing import Tuple

# flops_params.py
import math, inspect, torch, torch.nn as nn

# ---------- Params ----------
def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}

def fmt_params(n: int):
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.2f}K"
    return str(n)

# ---------- Custom counters (ptflops-style) ----------
def count_multihead_attention(m: nn.MultiheadAttention, inputs, output):
    # supports batch_first True/False
    q = inputs[0]
    batch_first = getattr(m, 'batch_first', False)
    if batch_first:
        B, L, E = q.shape
    else:
        L, B, E = q.shape
    h = m.num_heads
    d = E // h
    macs = 0
    # Q,K,V projections
    macs += 3 * B * L * E * E
    # attention scores + value mix
    macs += B * h * L * L * d * 2
    # out projection
    macs += B * L * E * E
    m.__flops__ += int(macs)


def count_mamba_block(m, inputs, output):
    # Approximate; assumes input [B, L, D]
    x = inputs[0]
    B, L, D = x.shape
    expand = getattr(m, 'expand', 2)
    D_in = expand * D
    k = getattr(m, 'd_conv', 4)
    macs = 0
    # gated proj to 2*D_in
    macs += B * L * D * (2 * D_in)
    # depthwise conv
    macs += B * D_in * L * k
    # scan core (affine + mixing, constant ~4 ops/token)
    macs += 4 * B * L * D_in
    # out proj D_in->D
    macs += B * L * D_in * D
    m.__flops__ += int(macs)

def _collect_custom_hooks():
    hooks = {nn.MultiheadAttention: count_multihead_attention}
    # try to register your Mamba class if available
    for path in (
        "mamba_ssm.Mamba",
        "mamba_ssm.modules.mamba_simple.Mamba",
        "mamba_ssm.modules.mamba_layer.Mamba",
    ):
        try:
            mod_path, cls_name = path.rsplit(".", 1)
            M = getattr(__import__(mod_path, fromlist=[cls_name]), cls_name)
            hooks[M] = count_mamba_block
            break
        except Exception:
            continue
    return hooks

# ---------- Main API ----------
def flops_and_params(model: nn.Module, input_size, device='cuda', as_flops=False):
    """
    input_size: tuple WITHOUT batch. E.g. (C,H,W) for images; (L,D) if model expects [B,L,D].
    Returns: (macs_or_flops, params_int)
    """
    model = model.to(device).eval()
    custom_hooks = _collect_custom_hooks()

    # Build kwargs compatible with your ptflops version
    sig = inspect.signature(get_model_complexity_info)
    kwargs = {}
    if 'custom_modules_hooks' in sig.parameters:
        kwargs['custom_modules_hooks'] = custom_hooks
    elif 'custom_ops' in sig.parameters:
        kwargs['custom_ops'] = custom_hooks  # older ptflops
    # else: no custom hook support; conv/linear will still be counted

    with torch.no_grad():
        macs, params = get_model_complexity_info(
            model, input_res=input_size,
            input_constructor=input_constructor,
            as_strings=False, print_per_layer_stat=False, verbose=False,
            **kwargs
        )
    return ((macs * 2) if as_flops else macs), params

# ---- Wrapper so ptflops can call model(x) with a single arg -----------------
class InferenceWrapper(nn.Module):
    """
    Adapts TULIP.forward(x, target, eval, mc_drop) -> forward(x) for ptflops.
    We create a dummy target of the expected high-res size and set mc_drop=True
    to bypass loss computation.
    """
    def __init__(self, core: nn.Module, in_chans: int, target_size: Tuple[int, int]):
        super().__init__()
        self.core = core
        self.in_chans = in_chans
        self.target_size = target_size  # (H_high, W_high)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        Hh, Wh = self.target_size
        dummy_target = torch.zeros(B, self.in_chans, Hh, Wh, device=x.device, dtype=x.dtype)
        out = self.core(x, dummy_target, eval=True, mc_drop=True)
        # TULIP returns the tensor directly when mc_drop=True; if your code ever returns a tuple, take [0]
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out
# -----------------------------------------------------------------------------

# -------------------- [FLOPS-GPU FIX] input constructor -----------------
def input_constructor(input_res):
    B = 1  # or your batch size for FLOPs
    # IMPORTANT: create the dummy input on the SAME DEVICE
    x = torch.randn(B, *input_res, device=device)
    return x
# -----------------------------------------------------------------------

if __name__=="__main__":
    args = get_args_parser()
    args = args.parse_args()

    model = tulip.__dict__[args.model_select](
        img_size = tuple(args.img_size_low_res),
        target_img_size = tuple(args.img_size_high_res),
        patch_size = tuple(args.patch_size),
        in_chans = args.in_chans,
        window_size = args.window_size,
        swin_v2 = args.swin_v2,
        pixel_shuffle = args.pixel_shuffle,
        circular_padding = args.circular_padding,
        log_transform = args.log_transform,
        patch_unmerging = args.patch_unmerging
    )
    device = torch.device(args.device)
    model = model.to(device).eval()

    # Wrap so ptflops can just do forward(x)
    wrapped = InferenceWrapper(model, in_chans=args.in_chans,
                               target_size=tuple(args.img_size_high_res))

    # IMPORTANT: run ptflops on CPU; it builds a dummy input internally
    input_size = (args.in_chans, ) + tuple(args.img_size_low_res)  # (C,H,W)
    macs, _ = flops_and_params(wrapped, input_size=input_size, device=args.device, as_flops=False)

    params = count_params(model)  # params of the actual core model

    print("=================================")
    print("Input size: ", tuple(args.img_size_low_res))
    print("Model:", args.model_select)
    print(f"Params: total {fmt_params(params['total'])}, trainable {fmt_params(params['trainable'])}")
    if macs is None:
        print("MACs: (ptflops failed to trace) None")
    else:
        print(f"MACs (1 input): {macs/1e9:.2f} GMac  | FLOPs ~ {2*macs/1e9:.2f} GFLOPs")
