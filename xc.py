import sys
from dataclasses import dataclass
from typing import List, Dict, Set, Any, Tuple
from enum import Enum
import argparse


@dataclass
class Problem:
    items: Set[str]
    options: List[Set[str]]
    open_option_idxs: List[int]

    def solve_(self, current_solution: List[int]):
        # If no items remain to be covered, print solution and return.
        if len(self.items) == 0:
            print("Found solution!")
            print(current_solution)
            return

        if len(self.options) == 0:
            return

        # Choose an item to cover.
        item_to_cover, max_len = None, 1000000
        for item in self.items:
            options_for_item = len(
                [
                    1
                    for option_idx in self.open_option_idxs
                    if item in self.options[option_idx]
                ]
            )
            if options_for_item < max_len:
                max_len = options_for_item
                item_to_cover = item

        # Iterate over all options with that item
        for option_to_try in [self.options[i] for i in self.open_option_idxs]:
            if item_to_cover in option_to_try:
                new_items = self.items.copy()
                covered_items = set()
                for item in option_to_try:
                    new_items.remove(item)
                    covered_items.add(item)

                new_options_idxs = [
                    option_idx
                    for option_idx in self.open_option_idxs
                    if self.options[option_idx].isdisjoint(covered_items)
                ]

                new_problem = Problem(new_items, self.options, new_options_idxs)
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

    open_option_idxs = list(range(len(options)))

    return Problem(items, options, open_option_idxs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()

    problem = load_problem(args.input_file)

    problem.solve()


if __name__ == "__main__":
    main()
