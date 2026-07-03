"""Run LLM adjudication workflow for reference candidates."""
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from procurement_graph.experiments.reference_llm_adjudication import cli_main


if __name__ == "__main__":
    cli_main()
