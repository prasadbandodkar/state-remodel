#!/usr/bin/env python3
# Prints a clean, explicit, deeply-detailed summary of a LLaMA architecture:
# - Full config with all fields and values
# - Module tree with __init__ and forward signatures (defaults included)
# - Layer-wise parameters (name, shape, dtype, requires_grad)
# - Parameter counts per-submodule and total
#
# Tip: By default this constructs the model from defaults (no weights download).
#      To load actual weights, pass --pretrained <repo_id> (may require auth and lots of RAM).

import argparse
import inspect
import json
import sys
from typing import Any, Dict, Iterable, Tuple

import torch
from transformers import (
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
)

# Optional: allow real model loading (heavy). If not installed or not needed, you can ignore.
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    _HAS_AUTO = True
except Exception:
    _HAS_AUTO = False


def signature_str(fn: Any) -> str:
    try:
        sig = inspect.signature(fn)
        return str(sig)
    except Exception:
        return "(signature unavailable)"


def format_value(val: Any) -> Any:
    # Make things JSON/YAML-friendly
    if isinstance(val, (int, float, bool, str)) or val is None:
        return val
    if isinstance(val, (list, tuple)):
        return [format_value(v) for v in val]
    if isinstance(val, dict):
        return {str(k): format_value(v) for k, v in val.items()}
    # Fallback to repr, but keep it short
    r = repr(val)
    if len(r) > 200:
        r = r[:197] + "..."
    return r


def dump_config(cfg: LlamaConfig) -> str:
    # Include every config field explicitly
    d = cfg.to_dict()
    # Sort keys for stable, clean output
    d_sorted = {k: d[k] for k in sorted(d.keys())}
    return json.dumps(d_sorted, indent=2, sort_keys=True)


def param_count(module: torch.nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def typename(obj: Any) -> str:
    t = type(obj)
    mod = t.__module__
    name = t.__name__
    if mod and mod != "builtins":
        return f"{mod}.{name}"
    return name


def describe_module(
    module: torch.nn.Module,
    name: str,
    depth: int,
    max_depth: int,
    lines: Iterable[str],
):
    indent = "  " * depth

    cls = type(module)
    cls_name = typename(module)

    init_sig = signature_str(cls.__init__)
    fwd_sig = signature_str(getattr(module, "forward", None)) if hasattr(module, "forward") else "(no forward)"

    trn_cnt, tot_cnt = param_count(module)

    lines.append(f"{indent}- module: {name}")
    lines.append(f"{indent}  class: {cls_name}")
    lines.append(f"{indent}  __init__{init_sig}")
    lines.append(f"{indent}  forward{fwd_sig}")
    lines.append(f"{indent}  parameters:")
    for p_name, p in module.named_parameters(recurse=False):
        lines.append(
            f"{indent}    - name: {p_name}, shape: {list(p.shape)}, dtype: {p.dtype}, requires_grad: {p.requires_grad}"
        )
    lines.append(f"{indent}  param_counts: {{trainable: {trn_cnt}, total: {tot_cnt}}}")

    if depth < max_depth:
        children = list(module.named_children())
        if children:
            lines.append(f"{indent}  submodules:")
            for child_name, child in children:
                describe_module(child, child_name, depth + 1, max_depth, lines)


def summarize_model(model: torch.nn.Module, max_depth: int = 99) -> str:
    lines: list[str] = []
    describe_module(model, name=model.__class__.__name__, depth=0, max_depth=max_depth, lines=lines)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Print detailed LLaMA architecture and parameters.")
    parser.add_argument(
        "--variant",
        type=str,
        default="causal-lm",
        choices=["backbone", "causal-lm"],
        help="Which LLaMA class to instantiate from config",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=99,
        help="Max depth for module tree (default: 99)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Model dtype (construction only when from config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to place the model on (default: cpu)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Optional HF repo id to load weights, e.g. meta-llama/Llama-2-7b-hf (heavy).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass through to from_pretrained when using --pretrained.",
    )
    args = parser.parse_args()

    # Build config with all defaults explicitly present in the output
    cfg = LlamaConfig()  # defaults; we will print them explicitly

    # Print environment info
    print("env:")
    print(f"  python: {sys.version.split()[0]}")
    print(f"  torch:  {torch.__version__}")
    try:
        import transformers  # noqa: F401
        import transformers.utils  # noqa: F401
        print(f"  transformers: {transformers.__version__}")
    except Exception:
        print("  transformers: (unknown)")
    print()

    # Explicitly dump ALL config fields and values
    print("config:")
    print(dump_config(cfg))
    print()

    # Construct model backbone or head-on-top using config, unless --pretrained is specified
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    if args.pretrained:
        if not _HAS_AUTO:
            print("ERROR: transformers Auto* classes not available; install full transformers.", file=sys.stderr)
            sys.exit(1)
        print(f"loading_pretrained: {args.pretrained}")
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained,
            device_map=None,          # keep on CPU unless you move it manually
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        model.to(args.device)
    else:
        if args.variant == "backbone":
            model = LlamaModel(cfg)
        else:
            model = LlamaForCausalLM(cfg)
        model.to(dtype=dtype, device=args.device)

    # Top-level parameter counts
    trn_cnt, tot_cnt = param_count(model)
    print("model:")
    print(f"  class: {typename(model)}")
    print(f"  device: {next(model.parameters(), torch.tensor([])).device if any(True for _ in model.parameters()) else args.device}")
    print(f"  dtype: {next(model.parameters(), torch.tensor([], dtype=dtype)).dtype if any(True for _ in model.parameters()) else dtype}")
    print(f"  parameters: {{trainable: {trn_cnt}, total: {tot_cnt}}}")
    print()

    # Print a clean, explicit module tree with signatures and parameters
    print("architecture:")
    print(summarize_model(model, max_depth=args.max_depth))


if __name__ == "__main__":
    main()