"""CLI wiring for the independent fixed-model replay workflow."""


def sample_limit(value):
    import argparse

    if value == "full":
        return value
    try:
        count = int(value)
        if count > 0:
            return count
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Expected a positive sample count or full")


def add_parser(subparsers):
    from .assets import MODEL_SETS

    root = subparsers.add_parser(
        "reproduce",
        help="Evaluate fixed FP32 and INT8 models and compare results",
        description="Prepare and evaluate existing models. This workflow reuses fixed ONNX artifacts without export, calibration or AdaRound training.",
    )
    commands = root.add_subparsers(dest="reproduce_command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Verify selected artifacts and bind ordered data"
    )
    prepare.add_argument(
        "--bindings",
        required=True,
        help="Private JSON: artifact root IDs to local directories",
    )
    prepare.add_argument(
        "--limit",
        type=sample_limit,
        default=3,
        help="Positive image/pair count, or full for each dataset (default: 3)",
    )
    run = commands.add_parser(
        "run", help="Replay prepared models twice and save raw tensors/predictions"
    )
    run.add_argument(
        "--package", required=True, help="Output directory from reproduce prepare"
    )
    run.add_argument("--bindings", required=True)
    run.add_argument(
        "--device", default="cuda:0", help="cuda:N (default: cuda:0) or cpu"
    )
    run.add_argument(
        "--storage",
        choices=("raw", "predictions"),
        default="raw",
        help="Full tensors (default) or predictions plus raw SHA256. Compact resume recomputes raw",
    )
    configs = commands.add_parser(
        "configs",
        help="Generate per-condition settings for the ordinary runner",
    )
    compare = commands.add_parser(
        "compare",
        help="Compare two captured runs and identify missing references",
    )
    compare.add_argument("--actual", required=True)
    compare.add_argument("--expected", required=True)
    report = commands.add_parser(
        "report", help="Export saved evaluation results without inference"
    )
    report.add_argument("--run", required=True)
    report.add_argument("--package", required=True)
    report.add_argument(
        "--table-profile", choices=("opennpu_raw",), default="opennpu_raw"
    )
    for parser in (prepare, run, compare, configs, report):
        if parser in (prepare, configs):
            parser.add_argument(
                "--selection",
                help="Custom selection manifest (default: packaged original records)",
            )
            group = parser.add_mutually_exclusive_group()
            group.add_argument(
                "--model-set",
                choices=MODEL_SETS,
                help="Model names only, not experiment settings. Default: extended (34 models) for packaged records, all for a custom --selection",
            )
            group.add_argument(
                "--models",
                help="Comma-separated model names instead of the default model set",
            )
        else:
            parser.add_argument(
                "--models",
                help="Comma-separated model names (default: all in the prepared package or saved runs)",
            )
        parser.add_argument(
            "--conditions",
            help="Comma-separated fp32,asym_pc_float_rtn,... (default: all)",
        )
        parser.add_argument("--output", required=True)
        parser.set_defaults(handler=handle)


def handle(args):
    from importlib import import_module

    keys = {
        "prepare": ("bindings", "output", "selection", "limit", "model_set"),
        "run": ("package", "bindings", "output", "device", "storage"),
        "configs": ("output", "selection", "model_set"),
        "compare": ("actual", "expected", "output"),
        "report": ("run", "package", "output", "table_profile"),
    }
    kwargs = {k: getattr(args, k) for k in keys[args.reproduce_command]}
    for key in ("models", "conditions"):
        value = getattr(args, key)
        kwargs[key] = (
            [v.strip() for v in value.split(",") if v.strip()]
            if value is not None
            else None
        )
    module = {"run": "execution", "configs": "settings"}.get(
        args.reproduce_command, args.reproduce_command
    )
    return getattr(import_module("." + module, __package__), args.reproduce_command)(
        **kwargs
    )
