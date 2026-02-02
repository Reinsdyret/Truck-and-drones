import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot local search score progression.")
    parser.add_argument(
        "--input",
        default="local_search_scores.csv",
        help="Path to the CSV written by local_search",
    )
    parser.add_argument(
        "--output",
        default="local_search_scores.png",
        help="Output image path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Missing input file: {input_path}")

    iterations = []
    best_scores = []
    incumbent_scores = []
    has_incumbent = False
    with input_path.open("r", encoding="utf-8") as handle:
        header = handle.readline()
        if not header:
            raise SystemExit("Empty input file")
        columns = [col.strip() for col in header.strip().split(",")]
        has_incumbent = "incumbent_score" in columns
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            iterations.append(int(parts[0]))
            best_scores.append(float(parts[1]))
            if has_incumbent and len(parts) >= 3:
                incumbent_scores.append(float(parts[2]))

    plt.figure(figsize=(10, 5))
    plt.plot(iterations, best_scores, linewidth=1.5)
    plt.title("Best Score Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Best Score")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Wrote plot to {args.output}")

    if has_incumbent and len(incumbent_scores) == len(iterations):
        incumbent_output = str(Path(args.output).with_name(Path(args.output).stem + "_incumbent" + Path(args.output).suffix))
        plt.figure(figsize=(10, 5))
        plt.plot(iterations, incumbent_scores, linewidth=1.5)
        plt.title("Incumbent Score Over Time")
        plt.xlabel("Iteration")
        plt.ylabel("Incumbent Score")
        plt.tight_layout()
        plt.savefig(incumbent_output, dpi=150)
        print(f"Wrote plot to {incumbent_output}")


if __name__ == "__main__":
    main()