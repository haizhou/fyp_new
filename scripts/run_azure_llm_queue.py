"""Run queued LLM adjudication tasks with Azure OpenAI."""
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from procurement_graph.experiments.llm_queue_runner import cli_main


if __name__ == "__main__":
    cli_main()
