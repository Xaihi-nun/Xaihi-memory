import argparse
from src.chroma_client import chroma_client

parser = argparse.ArgumentParser(description="List recent memories from ChromaDB")
parser.add_argument("-n", "--number", type=int, default=10, help="Number of recent memories to show (default: 10)")
args = parser.parse_args()

memories = chroma_client.get_all()

for m in memories[-args.number:]:
    print("-" * 50)
    # print(m["metadata"]["importance"], m["metadata"]["access_count"], m["metadata"]["base_importance"], m["content"])
    print(f"create={m['metadata']['created_at']}, imp={m['metadata']['importance']:.4f}, access_count={m['metadata']['access_count']}, base={m['metadata']['base_importance']:.4f}", m["content"])
    # print(m["metadata"].keys())
    # break
