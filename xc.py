import sys
from dataclasses import dataclass
from typing import List, Dict, Set, Any, Tuple
from enum import Enum
import argparse


@dataclass
class Problem:
    items: Set[str]
    options: List[Set[str]]

    def solve_(self, current_solution: List[str]):
        # If no items remain to be covered, print solution and return.
        if len(self.items) == 0:
            print("Found solution!")
            print(current_solution)
            return

        if len(self.options) == 0:
            return

        # Choose an item to cover.
        item_to_cover = next(iter(self.items))

        # Iterate over all options with that item
        for option_to_try in self.options:
            if item_to_cover in option_to_try:
                new_items = self.items.copy()
                covered_items = set()
                for item in option_to_try:
                    new_items.remove(item)
                    covered_items.add(item)

                new_options = [
                    option.copy()
                    for option in self.options
                    if option.isdisjoint(covered_items)
                ]

                new_problem = Problem(new_items, new_options)
                new_problem.solve_(current_solution + [option_to_try])

    def solve(self):
        return self.solve_([])


def load_problem(filename: str) -> Problem:
    options = []
    with open(filename) as f:
        lines = f.readlines()

        items = lines[0].strip().split(" ")
        for line in lines[1:]:
            option = line.strip().split(" ")
            for item in option:
                if item not in items:
                    raise KeyError(f"Unknown item {item} in option {option}")
            options.append(set(option))

    return Problem(items, options)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()

    problem = load_problem(args.input_file)

    problem.solve()


if __name__ == "__main__":
    main()
