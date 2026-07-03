"""Build and validate contract document URL availability."""
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from procurement_graph.documents.url_validation import cli_main


if __name__ == "__main__":
    cli_main()

