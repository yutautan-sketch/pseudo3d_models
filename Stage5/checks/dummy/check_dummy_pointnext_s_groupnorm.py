from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

from checks.dummy.check_dummy_pointnext_s_forward import make_window_dummy_h5, move_batch_to_device
from stage5.datasets import Pseudo3DPointCloudDataset, pad_point_window_collate
from stage5.models import build_stage5_model
from stage5.models.model_factory import _extract_state_dict
from stage5.models.norm_layers import Stage5GroupNorm, resolve_pointnext_norm_args
from stage5.models.pointnext_s_segmentor import build_pointnext_s_config


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ----------------------------------------------------------------------------
# S5-11 Step D2/D3 structural checks (require CUDA: OpenPoints ball
# query/FPS are CUDA-only, same constraint as every other PointNeXt-S checker).
# ----------------------------------------------------------------------------


def count_batchnorm_modules(model: nn.Module) -> int:
    return sum(1 for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm))


def count_groupnorm_modules(model: nn.Module) -> list[nn.GroupNorm]:
    return [module for module in model.modules() if isinstance(module, nn.GroupNorm)]


def build_dummy_batch(*, work_dir: Path, args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], int]:
    h5_path = work_dir / "dummy_pointnext_s_groupnorm_grid.h5"
    make_window_dummy_h5(
        h5_path,
        num_frames=args.num_frames,
        points_per_frame=args.points_per_frame,
        seed=args.seed,
    )
    dataset = Pseudo3DPointCloudDataset(
        [h5_path],
        num_points=None,
        features="intensity,confidence",
        window_mode="overlap",
        window_size_frames=args.window_size_frames,
        window_stride_frames=args.window_stride_frames,
        include_tail_window=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_point_window_collate,
    )
    batch = move_batch_to_device(next(iter(loader)), device)
    return batch, dataset.num_feature_channels


def build_model(*, norm: str, norm_groups: int, feature_dim: int, args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = build_stage5_model(
        "pointnext_s",
        num_classes=2,
        feature_dim=feature_dim,
        width=args.width,
        expansion=args.expansion,
        dropout=args.dropout,
        pointnext_radius=args.radius,
        pointnext_nsample=args.nsample,
        pointnext_sa_layers=args.sa_layers,
        pointnext_norm=norm,
        pointnext_norm_groups=norm_groups,
    ).to(device)
    return model


def run_backward_compat_checks(
    *, feature_dim: int, batch: dict[str, Any], args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    model = build_model(norm="batchnorm", norm_groups=args.norm_groups, feature_dim=feature_dim, args=args, device=device)
    require(model.pointnext_norm == "batchnorm", "Default pointnext_norm must resolve to 'batchnorm'")

    bn_count = count_batchnorm_modules(model)
    gn_count = len(count_groupnorm_modules(model))
    require(bn_count > 0, "Default (batchnorm) model must contain BatchNorm modules")
    require(gn_count == 0, f"Default (batchnorm) model must contain no GroupNorm modules, found {gn_count}")

    default_norm_args = resolve_pointnext_norm_args("batchnorm", args.norm_groups)
    require(
        default_norm_args == {"norm": "bn"},
        f"pointnext_norm='batchnorm' must resolve to the historical {{'norm': 'bn'}} config, "
        f"got {default_norm_args}",
    )
    cfg = build_pointnext_s_config(
        num_classes=2, feature_dim=feature_dim, width=args.width, norm="batchnorm", norm_groups=args.norm_groups
    )
    for site in ("encoder_args", "decoder_args", "cls_args"):
        require(
            dict(cfg[site]["norm_args"]) == {"norm": "bn"},
            f"batchnorm config's {site}.norm_args must be exactly {{'norm': 'bn'}}, "
            f"got {dict(cfg[site]['norm_args'])}",
        )

    model.eval()
    with torch.inference_mode():
        output = model(batch)
    logits = output["logits"]
    expected_shape = (batch["points"].shape[0], batch["points"].shape[1], 2)
    require(tuple(logits.shape) == expected_shape, f"Unexpected logits shape: {tuple(logits.shape)}")
    require(bool(torch.isfinite(logits).all()), "Backward-compat (batchnorm) logits contain NaN/Inf")

    legacy_report: dict[str, Any] = {"legacy_checkpoint_strict_load": None}
    if args.legacy_checkpoint:
        checkpoint = torch.load(args.legacy_checkpoint, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        result = model.load_state_dict(state_dict, strict=True)
        require(
            not result.missing_keys and not result.unexpected_keys,
            f"Legacy checkpoint strict load failed: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}",
        )
        with torch.inference_mode():
            legacy_output = model(batch)
        require(
            bool(torch.isfinite(legacy_output["logits"]).all()),
            "Legacy-checkpoint forward produced non-finite logits",
        )
        legacy_report["legacy_checkpoint_strict_load"] = "passed"
        legacy_report["legacy_checkpoint"] = str(args.legacy_checkpoint)

    return {
        "batchnorm_module_count": bn_count,
        "groupnorm_module_count_in_batchnorm_model": gn_count,
        **legacy_report,
    }


def run_groupnorm_structure_checks(
    *, feature_dim: int, batch: dict[str, Any], args: argparse.Namespace, device: torch.device, expected_norm_sites: int
) -> dict[str, Any]:
    model = build_model(
        norm="groupnorm", norm_groups=args.norm_groups, feature_dim=feature_dim, args=args, device=device
    )
    require(model.pointnext_norm == "groupnorm", "pointnext_norm must resolve to 'groupnorm'")

    bn_count = count_batchnorm_modules(model)
    gn_modules = count_groupnorm_modules(model)
    require(bn_count == 0, f"GroupNorm model must contain 0 BatchNorm modules, found {bn_count}")
    require(
        len(gn_modules) == expected_norm_sites,
        f"GroupNorm module count {len(gn_modules)} != expected normalization site count "
        f"{expected_norm_sites} (from the batchnorm control model)",
    )
    for module in gn_modules:
        require(
            module.num_channels % module.num_groups == 0,
            f"GroupNorm module has num_channels={module.num_channels} not divisible by "
            f"num_groups={module.num_groups}",
        )
        require(isinstance(module, Stage5GroupNorm), "All GroupNorm sites must use the Stage5GroupNorm adapter")

    cfg = build_pointnext_s_config(
        num_classes=2, feature_dim=feature_dim, width=args.width, norm="groupnorm", norm_groups=args.norm_groups
    )
    for site in ("encoder_args", "decoder_args", "cls_args"):
        norm_args = dict(cfg[site]["norm_args"])
        require(
            norm_args.get("norm") is Stage5GroupNorm and norm_args.get("num_groups") == args.norm_groups,
            f"groupnorm config's {site}.norm_args must use Stage5GroupNorm with num_groups="
            f"{args.norm_groups}, got {norm_args}",
        )

    model.eval()
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        eval_logits = model(batch)["logits"].clone()

    model.train()
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        train_logits = model(batch)["logits"].clone()
    model.eval()

    max_abs_diff = float((train_logits - eval_logits).abs().max().item())
    require(
        torch.allclose(train_logits, eval_logits, atol=1e-4),
        f"GroupNorm train()/eval() logits differ (max abs diff={max_abs_diff}); GroupNorm has no "
        "running-statistics dependency and must behave identically in both modes",
    )
    require(bool(torch.isfinite(eval_logits).all()), "GroupNorm logits contain NaN/Inf")

    # Invalid group count must fail fast, not silently adjust.
    invalid_channels = args.width  # stem width is never divisible by a too-large odd factor below
    invalid_groups = invalid_channels + 1
    raised = False
    try:
        Stage5GroupNorm(invalid_channels, num_groups=invalid_groups)
    except ValueError:
        raised = True
    require(raised, "Stage5GroupNorm must reject a num_groups that does not divide num_channels")

    # Checkpoint save/reload + logits parity.
    checkpoint_path = Path(args.work_dir) / "groupnorm_model_checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": model.get_config()}, checkpoint_path)
    reloaded = build_model(
        norm="groupnorm", norm_groups=args.norm_groups, feature_dim=feature_dim, args=args, device=device
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    result = reloaded.load_state_dict(checkpoint["model"], strict=True)
    require(
        not result.missing_keys and not result.unexpected_keys,
        f"GroupNorm checkpoint reload mismatch: missing={result.missing_keys}, "
        f"unexpected={result.unexpected_keys}",
    )
    reloaded.eval()
    with torch.inference_mode():
        reloaded_logits = reloaded(batch)["logits"]
    require(
        torch.equal(reloaded_logits, eval_logits),
        "GroupNorm logits after checkpoint reload do not exactly match the pre-save logits",
    )

    return {
        "groupnorm_module_count": len(gn_modules),
        "num_groups": args.norm_groups,
        "train_eval_max_abs_diff": max_abs_diff,
        "checkpoint_reload": "passed",
        "invalid_group_count_rejected": True,
    }


# ----------------------------------------------------------------------------
# Synthetic self-test: CPU-only, no CUDA / OpenPoints / real PointNeXt-S needed.
# ----------------------------------------------------------------------------


def test_stage5_groupnorm_forward_shapes() -> None:
    layer_1d = Stage5GroupNorm(8, num_groups=4)
    x_1d = torch.randn(2, 8, 5)
    out_1d = layer_1d(x_1d)
    require(tuple(out_1d.shape) == tuple(x_1d.shape), "Stage5GroupNorm must preserve [B, C, N] shape")
    require(bool(torch.isfinite(out_1d).all()), "Stage5GroupNorm [B, C, N] output must be finite")

    layer_2d = Stage5GroupNorm(8, num_groups=4)
    x_2d = torch.randn(2, 8, 3, 6)
    out_2d = layer_2d(x_2d)
    require(tuple(out_2d.shape) == tuple(x_2d.shape), "Stage5GroupNorm must preserve [B, C, *, *] shape")
    require(bool(torch.isfinite(out_2d).all()), "Stage5GroupNorm [B, C, *, *] output must be finite")


def test_stage5_groupnorm_rejects_indivisible_groups() -> None:
    raised = False
    try:
        Stage5GroupNorm(10, num_groups=3)
    except ValueError:
        raised = True
    require(raised, "Stage5GroupNorm(10, num_groups=3) must raise ValueError (10 %% 3 != 0)")


def test_stage5_groupnorm_train_eval_identical() -> None:
    layer = Stage5GroupNorm(8, num_groups=4)
    x = torch.randn(3, 8, 7)
    layer.eval()
    with torch.no_grad():
        eval_out = layer(x)
    layer.train()
    with torch.no_grad():
        train_out = layer(x)
    require(
        torch.equal(eval_out, train_out),
        "Stage5GroupNorm has no running-statistics buffers; train()/eval() outputs must be identical",
    )
    require(
        not any(True for _ in layer.buffers()),
        "Stage5GroupNorm must not register any running-statistics buffers",
    )


def test_resolve_pointnext_norm_args_batchnorm_matches_legacy_default() -> None:
    result = resolve_pointnext_norm_args("batchnorm", 8)
    require(result == {"norm": "bn"}, f"batchnorm norm_args must equal the historical {{'norm': 'bn'}}, got {result}")


def test_resolve_pointnext_norm_args_groupnorm() -> None:
    result = resolve_pointnext_norm_args("groupnorm", 8)
    require(result.get("norm") is Stage5GroupNorm, "groupnorm norm_args must use the Stage5GroupNorm class")
    require(result.get("num_groups") == 8, f"groupnorm norm_args must carry num_groups=8, got {result}")


def test_resolve_pointnext_norm_args_rejects_unknown_and_nonpositive() -> None:
    raised = False
    try:
        resolve_pointnext_norm_args("layernorm", 8)
    except ValueError:
        raised = True
    require(raised, "Unsupported pointnext_norm values must raise ValueError")

    raised = False
    try:
        resolve_pointnext_norm_args("groupnorm", 0)
    except ValueError:
        raised = True
    require(raised, "pointnext_norm_groups <= 0 must raise ValueError")


def test_stage5_pointnext_decoder_honors_norm_args() -> None:
    # Regression test for a real bug found while running this checker: the
    # official OpenPoints PointNextDecoder accepts norm_args/act_args via
    # **kwargs but never uses them, so its decoder always built BatchNorm
    # regardless of what Stage5 configured. See pointnext_decoder_patch.py.
    from stage5.models.pointnext_decoder_patch import ensure_stage5_pointnext_decoder_registered
    from stage5.models.pointnext_s_segmentor import _ensure_openpoints_importable

    _ensure_openpoints_importable()
    ensure_stage5_pointnext_decoder_registered()
    from openpoints.models.build import MODELS

    decoder_cls = MODELS.get("Stage5PointNextDecoder")
    require(decoder_cls is not None, "Stage5PointNextDecoder must be registered")

    encoder_channel_list = [8, 16, 32, 64, 128]
    decoder_bn = decoder_cls(encoder_channel_list=encoder_channel_list, norm_args={"norm": "bn"})
    bn_norm_modules = [m for m in decoder_bn.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    require(
        len(bn_norm_modules) > 0,
        "Stage5PointNextDecoder(norm_args={'norm': 'bn'}) must build BatchNorm modules",
    )
    require(
        not any(isinstance(m, nn.GroupNorm) for m in decoder_bn.modules()),
        "batchnorm-configured decoder must contain no GroupNorm modules",
    )

    decoder_gn = decoder_cls(
        encoder_channel_list=encoder_channel_list,
        norm_args={"norm": Stage5GroupNorm, "num_groups": 4},
    )
    gn_norm_modules = [m for m in decoder_gn.modules() if isinstance(m, nn.GroupNorm)]
    require(
        len(gn_norm_modules) == len(bn_norm_modules),
        f"groupnorm-configured decoder must replace every norm site 1:1, got "
        f"{len(gn_norm_modules)} GroupNorm vs {len(bn_norm_modules)} BatchNorm sites",
    )
    require(
        not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in decoder_gn.modules()),
        "groupnorm-configured decoder must contain 0 BatchNorm modules "
        "(this is exactly the bug this patch fixes)",
    )


def test_build_pointnext_s_config_propagates_norm_to_all_sites() -> None:
    cfg_bn = build_pointnext_s_config(num_classes=2, feature_dim=2, width=32, norm="batchnorm", norm_groups=8)
    for site in ("encoder_args", "decoder_args", "cls_args"):
        require(
            dict(cfg_bn[site]["norm_args"]) == {"norm": "bn"},
            f"batchnorm config {site}.norm_args must equal {{'norm': 'bn'}}",
        )

    cfg_gn = build_pointnext_s_config(num_classes=2, feature_dim=2, width=32, norm="groupnorm", norm_groups=8)
    for site in ("encoder_args", "decoder_args", "cls_args"):
        norm_args = dict(cfg_gn[site]["norm_args"])
        require(
            norm_args.get("norm") is Stage5GroupNorm and norm_args.get("num_groups") == 8,
            f"groupnorm config {site}.norm_args must use Stage5GroupNorm with num_groups=8, got {norm_args}",
        )
    # The three sites must not share a single mutated dict instance.
    cfg_gn["encoder_args"]["norm_args"]["num_groups"] = 999
    require(
        dict(cfg_gn["decoder_args"]["norm_args"])["num_groups"] == 8,
        "norm_args dicts must be independent copies across encoder/decoder/head sites",
    )


def run_self_test() -> None:
    test_stage5_groupnorm_forward_shapes()
    test_stage5_groupnorm_rejects_indivisible_groups()
    test_stage5_groupnorm_train_eval_identical()
    test_resolve_pointnext_norm_args_batchnorm_matches_legacy_default()
    test_resolve_pointnext_norm_args_groupnorm()
    test_resolve_pointnext_norm_args_rejects_unknown_and_nonpositive()
    test_stage5_pointnext_decoder_honors_norm_args()
    test_build_pointnext_s_config_propagates_norm_to_all_sites()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-11 Step D2/D3: PointNeXt-S BatchNorm backward-compat and GroupNorm structure checks"
    )
    parser.add_argument("--self_test", action="store_true", help="Run CPU-only synthetic tests and exit")
    parser.add_argument("--work_dir", default=str(REPO_ROOT / "work_dirs" / "_dummy_pointnext_s_groupnorm_check"))
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--points_per_frame", type=int, default=80)
    parser.add_argument("--window_size_frames", type=int, default=8)
    parser.add_argument("--window_stride_frames", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--radius", type=float, default=0.1)
    parser.add_argument("--nsample", type=int, default=16)
    parser.add_argument("--sa_layers", type=int, default=2)
    parser.add_argument("--norm_groups", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--legacy_checkpoint",
        default=None,
        help="Optional: an existing production BatchNorm checkpoint (e.g. teacher v6 pilot best.pt) "
        "to strict-load into the default (batchnorm) wrapper as an extra real-world backward-compat check",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        print("Stage5 PointNeXt-S GroupNorm self-test passed.")
        return

    if args.device != "cuda":
        raise RuntimeError("PointNeXt-S GroupNorm structure check requires --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device(args.device)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    batch, feature_dim = build_dummy_batch(work_dir=work_dir, args=args, device=device)

    backward_compat_report = run_backward_compat_checks(
        feature_dim=feature_dim, batch=batch, args=args, device=device
    )
    groupnorm_report = run_groupnorm_structure_checks(
        feature_dim=feature_dim,
        batch=batch,
        args=args,
        device=device,
        expected_norm_sites=backward_compat_report["batchnorm_module_count"],
    )

    summary = {
        "status": "passed",
        "backward_compat": backward_compat_report,
        "groupnorm": groupnorm_report,
    }
    with (work_dir / "groupnorm_structure_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Stage5 PointNeXt-S GroupNorm structure check passed.")
    print(f"batchnorm module count (control): {backward_compat_report['batchnorm_module_count']}")
    print(f"groupnorm module count: {groupnorm_report['groupnorm_module_count']}")
    print(f"train/eval max abs diff: {groupnorm_report['train_eval_max_abs_diff']:.3e}")
    if backward_compat_report.get("legacy_checkpoint_strict_load"):
        print(f"legacy checkpoint strict load: {backward_compat_report['legacy_checkpoint_strict_load']}")
    print(f"work_dir: {work_dir}")


if __name__ == "__main__":
    main()
