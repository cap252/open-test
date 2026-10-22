"""CLI for setup checks, export, run, reproduce and saved-result reporting.

build_parser() defines every option; each command has one handler function. Errors the
user can fix (bad settings, missing files) exit with code 2 and a one-line message.
"""

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

from . import __version__

DATASETS = (
    "imagenet",
    "coco",
    "coco_keypoints",
    "widerface",
    "voc",
    "lfw_mobilefacenet",
)
# Commands that only read result files and never import numerical runtimes.
METADATA_COMMANDS = ("report", "results")
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)
FILESYSTEM_HINTS = {
    FileExistsError: "Choose a new --output path; existing files are preserved.",
    IsADirectoryError: "Pass a file path instead of a directory.",
    NotADirectoryError: "Check that each parent path is a directory.",
    PermissionError: "Check read/write permission for the specified path.",
}


def _path_options(parser, *, output=False):
    group = parser.add_argument_group(
        "Paths",
        "Precedence: CLI > OPENNPU_QUANT_* environment variables > local YAML "
        "> defaults. Directory defaults are relative to home.",
    )
    group.add_argument(
        "--home", help="Workspace root (OPENNPU_QUANT_HOME, default: ./workspace)"
    )
    group.add_argument(
        "--models-dir",
        help="Exported models (OPENNPU_QUANT_MODELS, default: <home>/models)",
    )
    group.add_argument(
        "--data-dir",
        help="Dataset manifests (OPENNPU_QUANT_DATA, default: <home>/data)",
    )
    group.add_argument(
        "--cache-dir",
        help="Calibration, input and download caches (OPENNPU_QUANT_CACHE, default: <home>/cache)",
    )
    group.add_argument(
        "--output",
        help=(
            "Run parent folder. Writes <output>/<experiment name> (OPENNPU_QUANT_RUNS, default: <home>/runs)"
            if output
            else argparse.SUPPRESS
        ),
    )
    group.add_argument(
        "--weights-dir",
        help="Original checkpoints and upstream sources (OPENNPU_QUANT_WEIGHTS, default: <home>/weights)",
    )
    group.add_argument(
        "--local-config",
        help="Path settings YAML (default: ./opennpu_quant.local.yaml)",
    )


def _add_run_options(run):
    run.add_argument(
        "-c",
        "--config",
        default="core10",
        help="Packaged experiment (core10 or cnn30) or YAML path (default: core10)",
    )
    run.add_argument(
        "--models",
        help="Comma-separated model names from 'opennpu-quant models' or local recipes",
    )
    run.add_argument(
        "--conditions",
        help="Comma-separated keys from the selected config. core10: fp32,pot_rtn,pot_adaround,float_rtn,float_adaround. fp32 is included automatically",
    )
    run.add_argument(
        "--device",
        help="Evaluation and AdaRound device: cuda:N or cpu. Defaults to the config (cuda:0 in general presets). Also used for calibration when calibration.device=run",
    )
    from .models.evaluation_profiles import PROFILES

    run.add_argument(
        "--evaluation-profile",
        choices=PROFILES,
        help="Advanced comparison override for all selected models. Defaults match each model's final reference convention. Use evaluation.profiles in YAML for per-model comparisons",
    )
    run.add_argument(
        "--limit",
        type=int,
        help="Maximum evaluation samples. LFW requires at least 6000 pairs",
    )
    run.add_argument(
        "--calibration-samples",
        type=int,
        help="Override calibration sample counts for all datasets",
    )
    run.add_argument(
        "--adaround-steps",
        type=int,
        help="Override AdaRound steps per layer (packaged configs: 10000)",
    )
    memory = run.add_argument_group(
        "AdaRound memory", "Optional activation cache budgets. Unused by FP32 and RTN."
    )
    memory.add_argument(
        "--host-cache-gib",
        type=float,
        help="Activation cache RAM budget in GiB (default: 4)",
    )
    memory.add_argument(
        "--gpu-cache-mib",
        "--gpu-window-mib",
        dest="gpu_window_mib",
        metavar="MIB",
        type=float,
        help="Activation cache GPU budget in MiB (default: 512). Does not change adaround.window_bytes",
    )
    memory.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the AdaRound intermediate activation cache",
    )
    run.add_argument(
        "--no-input-cache",
        action="store_true",
        help="Decode calibration inputs instead of using the disk cache",
    )
    run.add_argument(
        "--recalibrate",
        action="store_true",
        help="Recollect FP32 calibration and rebuild selected INT8 conditions",
    )
    run.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print the evaluation summary as JSON and send progress messages to stderr",
    )
    for name, help_text in {
        "force": "Bypass completed-result reuse and recompute selected results. Calibration caches and AdaRound checkpoints may still be reused",
        "save-models": "Save a quantized model.onnx for each INT8 condition",
        "save-predictions": "Save per-sample predictions for comparison and debugging to predictions.jsonl.gz",
        "strict-versions": "Fail unless ONNX Runtime is the verified version 1.20.2",
        "dry-run": "Print effective configuration and paths without running inference",
    }.items():
        run.add_argument("--" + name, action="store_true", help=help_text)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="opennpu-quant",
        description="ONNX QDQ: float/PoT RTN and fixed-encoding AdaRound",
        epilog="See README.md for model preparation, quantization and reproduction examples.",
    )
    parser.add_argument(
        "--version", action="version", version="opennpu-quant " + __version__
    )
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser(
        "info",
        help="Environment and path diagnostics",
        description="Show environment and path diagnostics without downloading assets.",
    )
    _path_options(info, output=True)
    info.set_defaults(handler=_info)

    demo = sub.add_parser(
        "demo",
        help="Check quantization and evaluation with a demo model",
        description="Run a demo model without downloads. Scores check the execution path and do not measure pretrained accuracy.",
    )
    demo.add_argument(
        "--adaround", action="store_true", help="Also run fixed-encoding AdaRound"
    )
    demo.add_argument(
        "--device",
        default="cuda:0",
        help="Execution device: cuda:N (default: cuda:0) or cpu",
    )
    demo.set_defaults(handler=_demo)

    models = sub.add_parser(
        "models",
        help="List packaged model recipes",
        description="List the packaged model recipe names for export and run.",
    )
    models.set_defaults(handler=_models)

    export = sub.add_parser(
        "export",
        help="Create a model from its pinned original checkpoint",
        description="Export a model from its pinned original checkpoint into the models folder.",
    )
    export.add_argument("model", help="Model name from 'opennpu-quant models'")
    _path_options(export)
    export.add_argument(
        "--weights", help="Use a local original checkpoint with required checksum"
    )
    export.add_argument("--source-dir", help="Local pinned upstream checkout")
    export.add_argument(
        "--python", help="Optional exporter interpreter (default: current interpreter)"
    )
    export.add_argument(
        "--validation",
        choices=("strict", "functional"),
        default="strict",
        help="Functional mode records numerical parity failures. Structural checks remain mandatory",
    )
    export.set_defaults(handler=_export)

    data = sub.add_parser(
        "data",
        help="Prepare manifests and aligned LFW crops",
        description="Prepare dataset manifests or inspect their status.",
    )
    data_commands = data.add_subparsers(dest="data_command", required=True)
    prepare = data_commands.add_parser(
        "prepare",
        help="Prepare a dataset manifest",
        description="Prepare image manifests. LFW also writes aligned face crops.",
    )
    prepare.add_argument(
        "dataset",
        choices=DATASETS,
        help="Dataset whose calibration and validation lists will be prepared",
    )
    _path_options(prepare)
    prepare.add_argument(
        "--root", required=True, help="Dataset folder containing the existing images"
    )
    prepare.add_argument(
        "--source",
        choices=["hf", "folder"],
        default="folder",
        help="folder: local images (default). hf: ImageNet Hugging Face format",
    )
    prepare.add_argument(
        "--download",
        action="store_true",
        help="Download ImageNet (--source hf) or pinned LFW assets explicitly",
    )
    prepare.set_defaults(handler=_data_prepare)
    status = data_commands.add_parser(
        "status",
        help="Show prepared dataset counts",
        description="Show calibration and validation counts from existing dataset manifests.",
    )
    _path_options(status)
    status.set_defaults(handler=_data_status)

    run = sub.add_parser(
        "run",
        help="Run a packaged or local YAML experiment",
        description="Quantize and evaluate prepared models using a packaged or local YAML experiment. Models and data must already be available.",
    )
    _path_options(run, output=True)
    _add_run_options(run)
    run.set_defaults(handler=_run)

    report = sub.add_parser(
        "report",
        help="Rebuild tables from a run folder",
        description="Rebuild tables and summaries from an existing run folder without inference.",
    )
    report.add_argument(
        "run_dir",
        help="Run folder <output>/<experiment name>, containing effective_config.json",
    )
    report.add_argument(
        "--format", help="Comma-separated md,csv,tsv,json (default: all)"
    )
    report.add_argument(
        "--stdout",
        choices=["tsv", "csv"],
        help="Print only the raw table to stdout (report files are still written)",
    )
    report.add_argument("--table-profile", choices=("opennpu_raw",))
    report.set_defaults(handler=_report)

    results = sub.add_parser(
        "results",
        help="Export measurement records without inference",
        description="Read a portable measurement file such as records.json without inference.",
    )
    results.add_argument(
        "records_json",
        help="Measurement records file, e.g. records.json (not a folder)",
    )
    results.add_argument(
        "--output", help="Destination folder for CSV, TSV and JSON exports"
    )
    results.add_argument(
        "--stdout",
        choices=["tsv", "csv"],
        help="Print only the raw table for Excel or redirection",
    )
    results.add_argument("--table-profile", choices=("opennpu_raw",))
    results.set_defaults(handler=_results)
    from .reproduction.cli import add_parser

    add_parser(sub)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    metadata_only = args.command in METADATA_COMMANDS or (
        args.command == "reproduce" and args.reproduce_command == "report"
    )
    if not metadata_only:
        # Respect deliberate user limits; these defaults apply before numerical imports.
        for key in THREAD_VARIABLES:
            os.environ.setdefault(key, "1")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            import cv2

            cv2.setNumThreads(1)
        except ImportError:
            pass
    try:
        value = dispatch(args)
        if value is not None:
            if args.command == "run" and not args.dry_run and not args.json_output:
                _print_run_summary(value)
            else:
                print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    except tuple(FILESYSTEM_HINTS) as error:
        hint = FILESYSTEM_HINTS[type(error)]
        parser.exit(2, f"{type(error).__name__}: {error}. {hint}\n")
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")


def dispatch(args):
    """Run the handler selected by build_parser()."""
    return args.handler(args)


def _names(text):
    return [name.strip() for name in text.split(",") if name.strip()]


def _paths(args):
    from .paths import Paths

    keys = ["home", "models_dir", "data_dir", "cache_dir", "output", "weights_dir"]
    return Paths.resolve(
        **{k: getattr(args, k, None) for k in keys},
        local=getattr(args, "local_config", None),
    )


def _print_table(rows, delimiter):
    from .results import RAW_FIELDS, raw_export_row, table_text

    print(
        table_text(
            [raw_export_row(row) for row in rows],
            RAW_FIELDS,
            delimiter="\t" if delimiter == "tsv" else ",",
        ),
        end="",
    )


def _info(args):
    from .envcheck import info

    paths = _paths(args)
    return dict(**info(), paths=paths.to_dict())


def _demo(args):
    from .demo import run

    return run(adaround=args.adaround, device=args.device)


def _models(args):
    from .models.spec import model_names

    return model_names()


def _export(args):
    from .export.launcher import export

    return export(
        args.model,
        _paths(args),
        weights=args.weights,
        source=args.source_dir,
        python=args.python,
        **({"validation": args.validation} if args.validation != "strict" else {}),
    )


def _data_prepare(args):
    from .data.prepare import prepare

    manifest = _paths(args).data / args.dataset / "manifest.json"
    prepared = prepare(
        args.dataset,
        args.root,
        manifest,
        source=args.source,
        download=args.download,
    )
    return dict(
        manifest=str(manifest),
        validation=len(prepared["validation"]),
        calibration=len(prepared["calibration"]),
    )


def _data_status(args):
    from ._io import read_json

    data = _paths(args).data
    status = {}
    for name in DATASETS:
        path = data / name / "manifest.json"
        if path.exists():
            manifest = read_json(path)
            status[name] = dict(
                calibration=len(manifest["calibration"]),
                validation=len(manifest["validation"]),
                root=manifest["root"],
            )
    return status


def _report(args):
    from .report import report

    result = report(args.run_dir, formats=args.format, table_profile=args.table_profile)
    if args.stdout:
        if args.table_profile:
            from .results import table_text

            print(
                table_text(
                    result["raw_data"]["full"],
                    delimiter="," if args.stdout == "csv" else "\t",
                ),
                end="",
            )
        else:
            _print_table(result["rows"], args.stdout)
        return None
    return result["mean_recovery"]


def _results(args):
    from .results import export_records, read_records, tables

    if args.table_profile and not args.output:
        raise ValueError(
            "--table-profile requires --output to retain partial tables and metadata"
        )
    payload = read_records(args.records_json)
    rows = tables(payload["records"])[0]
    if args.output:
        export_records(
            payload["records"],
            args.output,
            metadata=payload.get("metadata"),
            table_profile=args.table_profile,
        )
    if args.stdout:
        if args.table_profile:
            from .raw_report import build_report
            from .results import table_text

            print(
                table_text(
                    build_report(payload["records"], profile=args.table_profile)[
                        "full"
                    ],
                    delimiter="," if args.stdout == "csv" else "\t",
                ),
                end="",
            )
        else:
            _print_table(rows, args.stdout)
        return None
    return dict(
        records=len(rows),
        completed=sum(r["complete"] for r in rows),
        output=args.output,
    )


def _run(args):
    from .configuration import load_config, validate
    from .models.spec import model_names

    paths = _paths(args)
    config = load_config(args.config)
    if args.models:
        config["models"] = _names(args.models)
    packaged = model_names()
    for name in config["models"]:
        local = paths.models / name / "recipe.yaml"
        if name not in packaged and not local.is_file():
            raise ValueError(
                f"Unknown model '{name}'. Packaged models: {', '.join(packaged)}; "
                f"a local model needs {local}"
            )
    if args.conditions:
        chosen = _names(args.conditions)
        unknown = [n for n in chosen if n not in config["conditions"]]
        if unknown:
            raise ValueError(
                f"Unknown condition(s): {', '.join(unknown)}. "
                f"Available: {', '.join(config['conditions'])}"
            )
        if "fp32" not in chosen:
            if "fp32" not in config["conditions"]:
                raise ValueError(
                    "conditions.fp32 is required for --conditions to include the baseline. Add fp32: {} to conditions."
                )
            chosen.insert(0, "fp32")
        config["conditions"] = {k: config["conditions"][k] for k in chosen}
    _apply_run_overrides(config, args)
    validate(config, models_dir=paths.models)
    if args.strict_versions:
        from .ort.session import runtime
        from .ort.config import OrtConfig

        runtime(OrtConfig.cpu(strict_versions=True))
    if args.dry_run:
        return dict(config=config, paths=paths.to_dict(), will_download=False)
    from .runner import run

    progress = (
        contextlib.redirect_stdout(sys.stderr)
        if args.json_output
        else contextlib.nullcontext()
    )
    with progress:
        result = run(config, paths, force=args.force, recalibrate=args.recalibrate)
    return dict(
        output=str(paths.runs / config["name"]),
        device=config["runtime"]["device"],
        rows=result["rows"],
        mean_recovery=result["mean_recovery"],
    )


def _print_run_summary(result):
    from .results import SUCCESS, percent

    def percentage(value):
        return "—" if value is None else f"{value:.4f}%"

    metrics = [json.loads(row["metrics_json"]) for row in result["rows"]]
    show_top5 = any("top5" in values for values in metrics)
    header = ["Model", "Condition", "Metric", "Accuracy"]
    if show_top5:
        header.append("Top-5")
    header += ["Recovery", "Samples", "Evaluation", "State"]
    lines = [header]
    for row, values in zip(result["rows"], metrics):
        line = [
            row["model"],
            row["condition"],
            row["metric"],
            percentage(row["value_pct"]),
        ]
        if show_top5:
            line.append(percentage(percent(values.get("top5"))))
        success = row["state"] in SUCCESS
        line += [
            percentage(row["recovery_pct"]),
            f"{row['samples'] if row['samples'] is not None else '—'} / {row['expected_samples'] if row['expected_samples'] is not None else '—'}",
            ("full" if row["complete"] else "partial") if success else "—",
            row["state"],
        ]
        lines.append(line)
    widths = [max(len(line[i]) for line in lines) for i in range(len(header))]
    print("Device:", result["device"])
    for line in lines:
        print(
            "  ".join(value.ljust(width) for value, width in zip(line, widths)).rstrip()
        )
    print("\nFull-evaluation mean recovery:")
    for condition, value in result["mean_recovery"].items():
        detail = (
            percentage(value)
            if value is not None
            else "not aggregated (requires full evaluation and matched FP32 results)"
        )
        print(f"  {condition}: {detail}")
    print("\nOutput:", result["output"])
    print("Summary:", str(Path(result["output"]) / "summary.md"))


def _apply_run_overrides(config, args):
    """Copy run flags into the experiment mapping; validate() checks the result."""
    if args.device:
        config["runtime"]["device"] = args.device
    if getattr(args, "evaluation_profile", None):
        config["evaluation"].setdefault("profiles", {}).update(
            {model: args.evaluation_profile for model in config["models"]}
        )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("Positive limit required")
        config["evaluation"]["limit"] = args.limit
    if args.calibration_samples is not None:
        if args.calibration_samples < 1:
            raise ValueError("Positive calibration count required")
        config["calibration"]["samples"] = {
            k: args.calibration_samples for k in config["calibration"]["samples"]
        }
    if args.adaround_steps is not None:
        config["adaround"]["steps"] = args.adaround_steps
    if args.save_models:
        config["output"]["save_qdq_models"] = True
    if args.save_predictions:
        config["output"]["save_predictions"] = True
    cache = config.setdefault("activation_cache", {})
    if args.host_cache_gib is not None:
        cache["host_bytes"] = int(args.host_cache_gib * 1024**3)
    if args.gpu_window_mib is not None:
        cache["device_bytes"] = int(args.gpu_window_mib * 1024**2)
    if args.no_cache:
        cache["enabled"] = False
    if args.no_input_cache:
        config["calibration"].setdefault("input_cache", {})["enabled"] = False
