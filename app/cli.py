
import argparse

from app.core import (
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5
)


def run_project():
    """Execute all five project milestones."""

    print("\nMilestone 1: Generating dataset...")

    baskets = milestone_1(num_baskets=1000)

    print(f"Generated {len(baskets)} baskets.")

    print("\nMilestone 2: Training N-gram model...")

    ngram_model = milestone_2(baskets)

    print("N-gram model trained.")

    print("\nMilestone 3: Training Small Language Model...")

    neural_model = milestone_3(baskets)

    print("Small Language Model trained.")

    print("\nMilestone 4: Evaluating models...")

    # Evaluate baseline, N-gram and neural models.
    results = milestone_4(baskets)

    print("\nMilestone 5: Comparing models...")

    # Pass actual evaluation results to milestone 5.
    comparison = milestone_5(results)

    print("\nProject completed successfully!")

    return comparison



def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Retail Basket Prediction Analyzer"
    )

    parser.add_argument(
        "command",
        choices=["hello", "run"],
        help="Command to execute"
    )

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Retail Basket Prediction Analyzer is ready!")

    elif args.command == "run":
        run_project()


if __name__ == "__main__":
    main()
