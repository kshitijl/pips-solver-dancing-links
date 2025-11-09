import sys
import re
from dataclasses import dataclass
from typing import List, Dict, Set, Any, Tuple
from enum import Enum
import argparse


@dataclass
class Problem:
    items: Dict[str, int]
    options: List[Dict[str, int]]
    open_option_idxs: List[int]

    def solve_(self, current_solution: List[int]) -> None:
        # If no items remain to be covered, print solution and return.
        if len(self.items) == 0:
            print("Found solution!")
            for option_idx in current_solution:
                print(self.options[option_idx])
            return

        if len(self.options) == 0:
            return

        # Choose an item to cover, using MRV heuristic
        item_to_cover, min_len = None, 1000000
        for item in self.items.keys():
            options_for_item = len(
                [
                    1
                    for option_idx in self.open_option_idxs
                    if item in self.options[option_idx]
                ]
            )
            if options_for_item < min_len:
                min_len = options_for_item
                item_to_cover = item

        print(f"Choosing item {item_to_cover} to cover with {min_len} open options")

        # Iterate over all options with that item
        for option_to_try_idx in self.open_option_idxs:
            option_to_try = self.options[option_to_try_idx]
            if item_to_cover in option_to_try:
                new_items = self.items.copy()
                covered_items = set()
                for item, weight in option_to_try.items():
                    new_items[item] -= weight
                    if new_items[item] == 0:
                        del new_items[item]
                        covered_items.add(item)

                new_options_idxs = [
                    option_idx
                    for option_idx in self.open_option_idxs
                    if self.options[option_idx].keys().isdisjoint(covered_items)
                ]

                new_problem = Problem(new_items, self.options, new_options_idxs)
                new_problem.solve_(current_solution + [option_to_try_idx])

    def solve(self) -> None:
        return self.solve_([])


def load_problem(filename: str) -> Problem:
    items: Dict[str, int] = {}
    options: List[Dict[str, int]] = []
    with open(filename) as f:
        lines = f.readlines()

        for item in lines[0].strip().split(" "):
            m = re.match("([a-zA-Z0-9_]+)\\[([0-9]+):([0-9]+)\\]", item)
            if m:
                item_name, upper, lower = m.groups()
                upper, lower = int(upper), int(lower)
            else:
                item_name, upper, lower = item, 1, 1

            if upper != lower:
                raise ValueError(
                    f"Unsupported: upper bound {upper} != lower bound {lower} for {item_name}"
                )

            items[item_name] = lower

        for line in lines[1:]:
            option = line.strip().split(" ")
            option_dict = {}
            for item in option:
                m = re.match("([a-zA-Z0-9_]+)=([0-9]+)", item)
                if m:
                    item_name, weight = m.groups()
                    weight = int(weight)
                else:
                    item_name, weight = item, 1
                if item_name not in items:
                    raise KeyError(f"Unknown item {item_name} in option {option}")
                option_dict[item_name] = weight
            options.append(option_dict)

    open_option_idxs = list(range(len(options)))

    return Problem(items, options, open_option_idxs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()

    problem = load_problem(args.input_file)

    print(problem)

    problem.solve()


if __name__ == "__main__":
    main()
