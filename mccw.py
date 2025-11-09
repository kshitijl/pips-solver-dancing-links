import sys
import re
from dataclasses import dataclass
from typing import List, Dict, Set, Any, Tuple
from enum import Enum
import argparse
import copy


@dataclass
class PrimaryItemData:
    bound: int


@dataclass
class SecondaryItemData:
    color: int | None


type ItemData = PrimaryItemData | SecondaryItemData


@dataclass
class PrimaryOptionData:
    weight: int


@dataclass
class SecondaryOptionData:
    color: int | None


type OptionData = PrimaryOptionData | SecondaryOptionData


@dataclass
class Problem:
    items: Dict[str, ItemData]
    options: List[Dict[str, OptionData]]
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

        # print(f"\n\nChoosing item {item_to_cover} to cover with {min_len} open options")

        # Iterate over all options with that item that don't push its bound below 0
        for option_to_try_idx in self.open_option_idxs:
            option_to_try = self.options[option_to_try_idx]
            if item_to_cover in option_to_try:
                # print(f"Trying option {option_to_try_idx}: {option_to_try}")
                new_items = copy.deepcopy(self.items)
                covered_items = set()
                colored_items: Dict[str, int] = {}
                for item, option_data in option_to_try.items():
                    match option_data:
                        case PrimaryOptionData(weight=weight):
                            item_data = new_items[item]
                            assert isinstance(item_data, PrimaryItemData)
                            item_data.bound -= weight
                            if item_data.bound == 0:
                                del new_items[item]
                                covered_items.add(item)
                            if item_data.bound < 0:
                                assert False
                        case SecondaryOptionData(color=color):
                            if color is not None:
                                item_data = new_items[item]
                                assert isinstance(item_data, SecondaryItemData)
                                item_data.color = color
                                colored_items[item] = color

                new_option_idxs: List[int] = []
                # print(f"covered items: {covered_items}")
                for option_idx in self.open_option_idxs:
                    if option_idx == option_to_try_idx:
                        continue
                    compatible = True
                    if not self.options[option_idx].keys().isdisjoint(covered_items):
                        compatible = False
                        continue
                    for item_name, option_data in self.options[option_idx].items():
                        if item_name in colored_items:
                            assert isinstance(option_data, SecondaryOptionData)
                            if option_data.color != colored_items[item_name]:
                                compatible = False
                                break
                        if isinstance(option_data, PrimaryOptionData):
                            if option_data.weight > new_items[item_name].bound:
                                compatible = False
                                break
                    if compatible:
                        new_option_idxs.append(option_idx)

                # print(f"new items: {new_items}, new option idxs: {new_option_idxs}")
                new_problem = Problem(new_items, self.options, new_option_idxs)
                new_problem.solve_(current_solution + [option_to_try_idx])

    def solve(self) -> None:
        return self.solve_([])


def load_problem(filename: str) -> Problem:
    items: Dict[str, ItemData] = {}
    options: List[Dict[str, OptionData]] = []

    with open(filename) as f:
        lines = f.readlines()

        in_primary_items_section = True
        for item in lines[0].strip().split(" "):
            if item.strip() == "|":
                in_primary_items_section = False
                continue

            m = re.match("([a-zA-Z0-9_]+)\\[([0-9]+):([0-9]+)\\]", item)
            if m:
                if not in_primary_items_section:
                    raise ValueError(f"Secondary item {item} has multiplicity")
                item_name, upper, lower = m.groups()
                upper, lower = int(upper), int(lower)
            else:
                item_name, upper, lower = item, 1, 1

            if upper != lower:
                raise ValueError(
                    f"Unsupported: upper bound {upper} != lower bound {lower} for {item_name}"
                )

            item_data: ItemData

            if in_primary_items_section:
                item_data = PrimaryItemData(bound=lower)
            else:
                item_data = SecondaryItemData(color=None)

            items[item_name] = item_data

        for line in lines[1:]:
            option = line.strip().split(" ")
            option_dict = {}

            for item in option:
                got_weight, got_color = False, False
                m1 = re.match("([a-zA-Z0-9_]+)=([0-9]+)", item)
                m2 = re.match("([a-zA-Z0-9_]+):([0-9]+)", item)
                if m1:
                    item_name, weight_s = m1.groups()
                    weight = int(weight_s)
                    got_weight = True
                    # print("got weight")
                elif m2:
                    item_name, color_s = m2.groups()
                    color = int(color_s)
                    got_color = True
                    # print("got color")
                else:
                    item_name, weight, color = item, 1, None

                if item_name not in items:
                    # print(items)
                    raise KeyError(f"Unknown item {item_name} in option {option}")

                option_data: OptionData
                if isinstance(items[item_name], PrimaryItemData):
                    if got_color:
                        raise ValueError(
                            f"Got color {color} for primary item {item_name}"
                        )
                    option_data = PrimaryOptionData(weight=weight)

                if isinstance(items[item_name], SecondaryItemData):
                    if got_weight:
                        # print(item, option)
                        raise ValueError(
                            f"Got weight {weight} for secondary item {item_name}"
                        )
                    option_data = SecondaryOptionData(color=color)

                option_dict[item_name] = option_data
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
