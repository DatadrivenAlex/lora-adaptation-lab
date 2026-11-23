"""Command-line entry point for the reproducible synthetic experiment."""

from __future__ import annotations

import argparse
import json

from lora_adaptation_lab.demo import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic LoRA adaptation demo")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    print(json.dumps(run_demo(seed=args.seed, steps=args.steps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
