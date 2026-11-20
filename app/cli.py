
import argparse

from app.core import (
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Retail Basket Prediction Analyzer"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hello")
    sub.add_parser("run")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Retail Basket Prediction Analyzer is ready.")

    elif args.command == "run":
        print("\nMilestone 1: Generating dataset...")
        baskets = milestone_1()
        print("Generated", len(baskets), "baskets.")

        print("\nMilestone 2: Training N-gram model...")
        milestone_2(baskets)
        print("N-gram model trained.")

        print("\nMilestone 3: Training Small Language Model...")
        milestone_3(baskets)
        print("Small Language Model trained.")

        print("\nMilestone 4: Evaluating models...")
        results = milestone_4(baskets)

        print("\nMilestone 5: Comparing models...")
        milestone_5(results)

        print("\nProject completed successfully!")


if __name__ == "__main__":
    main()
