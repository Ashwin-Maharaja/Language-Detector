"""
main.py - Master Runner Script for the Language Detection ML Pipeline
======================================================================
Orchestrates every phase of the project: dataset generation, exploratory
data analysis, model training / evaluation, and deployment.

Usage:
    python main.py [step]

Steps:
    dataset   - Generate the synthetic language dataset
    eda       - Run exploratory data analysis and preprocessing
    train     - Train and evaluate all models
    deploy    - Start the Flask deployment server
    all       - Run dataset → eda → train (excludes deploy)

Examples:
    python main.py all          # run the full pipeline (no server)
    python main.py train        # only (re-)train the models
    python main.py deploy       # launch the web server
"""

import os
import sys
import time
import argparse
import importlib
import logging

# ---------------------------------------------------------------------------
# Configure logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root (directory containing this script)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Ensure the project root is on sys.path so that `src.*` imports work
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _banner(title: str) -> None:
    """Print a formatted banner for a pipeline step."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _elapsed(start: float) -> str:
    """Return a human-readable elapsed-time string."""
    secs = time.time() - start
    if secs < 60:
        return f"{secs:.2f} seconds"
    mins, secs = divmod(secs, 60)
    return f"{int(mins)}m {secs:.2f}s"


def _safe_import(module_path: str):
    """
    Import a module by dotted path, returning None on failure so the
    caller can handle missing dependencies gracefully.
    """
    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        logger.error("Could not import '%s': %s", module_path, e)
        logger.error(
            "Make sure all dependencies are installed: pip install -r requirements.txt"
        )
        return None
    except Exception as e:
        logger.error("Unexpected error importing '%s': %s", module_path, e)
        return None


# ---------------------------------------------------------------------------
# Pipeline step functions
# ---------------------------------------------------------------------------
def run_dataset() -> bool:
    """
    Step 1 – Generate the language dataset.
    Imports and executes src/dataset.py.
    Returns True on success, False on failure.
    """
    _banner("Step 1: Dataset Generation")
    start = time.time()

    # Import create_dataset from the project root
    sys.path.insert(0, PROJECT_ROOT)
    try:
        import create_dataset as mod
    except ImportError as e:
        logger.error("Could not import create_dataset: %s", e)
        return False

    try:
        if hasattr(mod, "create_dataset"):
            mod.create_dataset()
        elif hasattr(mod, "main"):
            mod.main()
        else:
            logger.warning(
                "create_dataset has no create_dataset() or main(). "
                "Attempting to reload the module."
            )
            importlib.reload(mod)

        logger.info("Dataset generation completed in %s", _elapsed(start))
        return True
    except Exception as e:
        logger.error("Dataset generation failed: %s", e, exc_info=True)
        return False


def run_eda() -> bool:
    """
    Step 2 – Exploratory Data Analysis and preprocessing.
    Imports and executes src/eda.py.
    Returns True on success, False on failure.
    """
    _banner("Step 2: EDA & Preprocessing")
    start = time.time()

    mod = _safe_import("src.eda_preprocessing")
    if mod is None:
        return False

    try:
        if hasattr(mod, "main"):
            mod.main()
        elif hasattr(mod, "run_eda"):
            # run_eda needs the dataframe, call main instead
            mod.main()
        else:
            logger.warning(
                "src.eda has no main() or run_eda(). "
                "Attempting to reload the module."
            )
            importlib.reload(mod)

        logger.info("EDA & Preprocessing completed in %s", _elapsed(start))
        return True
    except Exception as e:
        logger.error("EDA & Preprocessing failed: %s", e, exc_info=True)
        return False


def run_train() -> bool:
    """
    Step 3 – Train and evaluate all models.
    Imports and executes src/train.py.
    Returns True on success, False on failure.
    """
    _banner("Step 3: Model Training & Evaluation")
    start = time.time()

    mod = _safe_import("src.model_training")
    if mod is None:
        return False

    try:
        if hasattr(mod, "main"):
            mod.main()
        elif hasattr(mod, "train_models"):
            mod.train_models()
        else:
            logger.warning(
                "src.train has no main() or train_models(). "
                "Attempting to reload the module."
            )
            importlib.reload(mod)

        logger.info("Model Training & Evaluation completed in %s", _elapsed(start))
        return True
    except Exception as e:
        logger.error("Model Training & Evaluation failed: %s", e, exc_info=True)
        return False


def run_deploy() -> None:
    """
    Step 4 – Start the Flask deployment server.
    This is a blocking call; the server runs until interrupted.
    """
    _banner("Step 4: Deployment Server")

    mod = _safe_import("src.deploy")
    if mod is None:
        return

    try:
        if hasattr(mod, "start_server"):
            mod.start_server(debug=True)
        else:
            logger.error("src.deploy has no start_server() function.")
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
    except Exception as e:
        logger.error("Deployment server failed: %s", e, exc_info=True)


def run_all() -> None:
    """
    Run the complete pipeline: dataset → eda → train.
    Deploy is intentionally excluded (it's a blocking server).
    """
    _banner("Running Full Pipeline: dataset → eda → train")
    pipeline_start = time.time()

    results = {}

    # Step 1
    results["dataset"] = run_dataset()

    # Step 2
    results["eda"] = run_eda()

    # Step 3
    results["train"] = run_train()

    # --- Summary -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Pipeline Summary")
    print("=" * 60)
    for step, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {step:<15} {status}")
    print("-" * 60)
    print(f"  Total time: {_elapsed(pipeline_start)}")
    print("=" * 60)

    # Exit with non-zero code if any step failed
    if not all(results.values()):
        logger.warning("One or more pipeline steps failed. Review the logs above.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Language Detection ML Pipeline – Master Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=["dataset", "eda", "train", "deploy", "all"],
        help=(
            "Pipeline step to execute (default: all)\n"
            "  dataset  - Generate the language dataset\n"
            "  eda      - Run EDA and preprocessing\n"
            "  train    - Train and evaluate all models\n"
            "  deploy   - Start the deployment server\n"
            "  all      - Run dataset → eda → train"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    """Main entry point – dispatch to the requested pipeline step."""
    args = parse_args()

    print("=" * 60)
    print("  🌐 Language Detection – ML Pipeline Runner")
    print("=" * 60)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Selected step: {args.step}")
    print("=" * 60)

    dispatch = {
        "dataset": run_dataset,
        "eda": run_eda,
        "train": run_train,
        "deploy": run_deploy,
        "all": run_all,
    }

    step_fn = dispatch[args.step]

    overall_start = time.time()

    try:
        result = step_fn()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(130)

    # For individual steps (not 'all' / 'deploy'), print a quick summary
    if args.step not in ("all", "deploy"):
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"\n  {args.step}: {status}  ({_elapsed(overall_start)})")


if __name__ == "__main__":
    main()
