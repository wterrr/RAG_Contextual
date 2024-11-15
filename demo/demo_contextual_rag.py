import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import threading
from src.settings import setting
from src.embedding import RAG

GREEN = "\033[92m"
RESET = "\033[0m"

parser = argparse.ArgumentParser(description="Demo for Contextual RAG")
parser.add_argument(
    "--q",
    type=str,
    help="Query",
    required=True,
)
parser.add_argument(
    "--compare",
    action="store_true",
    help="Compare the original RAG and the contextual RAG",
)
parser.add_argument(
    "--debug", action="store_true", help="Run in debug mode for contextual search."
)

args = parser.parse_args()
q = args.q

rag = RAG(setting)

if args.compare:
    thread = [
        threading.Thread(
            target=lambda: print(
                f"\n\n{GREEN}Origin RAG: {RESET}{rag.origin_rag_search(q)}"
            )
        ),
        threading.Thread(
            target=lambda: print(
                f"\n\n{GREEN}Contextual RAG: {RESET}{rag.contextual_rag_search(q, debug=args.debug)}"
            )
        ),
    ]

    for t in thread:
        t.start()

    for t in thread:
        t.join()
else:
    print(
        f"{GREEN}Contextual RAG: {RESET}{rag.contextual_rag_search(q, debug=args.debug)}"
    )
