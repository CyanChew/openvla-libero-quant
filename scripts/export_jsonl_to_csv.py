import json
import csv
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, default="outputs/results.jsonl")
    parser.add_argument("--csv", type=str, default="outputs/results.csv")
    args = parser.parse_args()

    results = []
    with open(args.jsonl, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        print("No results to export.")
        return

    keys = list(results[0].keys())
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for res in results:
            writer.writerow(res)
    print(f"Exported {len(results)} rows to {args.csv}")

if __name__ == "__main__":
    main()