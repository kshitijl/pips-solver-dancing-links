import sys
import re
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Any, Tuple
from enum import Enum
import argparse
import copy

num_solutions = 0
debug_print = False
print_solution_summaries = False
print_solution_details = False


@dataclass
class PrimaryItemData:
    bound: int
    slack: int

    def within_limits(self):
        if self.bound >= 0 and self.bound <= self.slack:
            return True
        return False


@dataclass
class SecondaryItemData:
    color: int | None


@dataclass
class PrimaryOptionData:
    weight: int


@dataclass
class SecondaryOptionData:
    color: int | None


@dataclass
class Option:
    primaries: Dict[str, PrimaryOptionData]
    secondaries: Dict[str, SecondaryOptionData]


@dataclass
class Problem:
    primary_items: Dict[str, PrimaryItemData]
    secondary_items: Dict[str, SecondaryItemData]
    options: List[Option]
    open_option_idxs: List[int]

    def all_primaries_within_limits(self) -> bool:
        for item in self.primary_items.values():
            if not item.within_limits():
                return False
        return True

    def solve_(self, current_solution: List[int]) -> None:
        global num_solutions
        # If no primary items remain to be covered, print solution and return.
        if len(self.primary_items) == 0 or self.all_primaries_within_limits():
            if print_solution_summaries:
                print(f"Found solution! {current_solution}")
            num_solutions += 1
            if num_solutions % 100 == 0:
                print(f"{num_solutions} solutions")
            if print_solution_details:
                for option_idx in current_solution:
                    print(self.options[option_idx])
            return

        if len(self.options) == 0:
            return

        # Choose a primary item to cover, using MRV heuristic
        item_to_cover, min_len = None, 1000000
        for item, data in self.primary_items.items():
            if data.within_limits():
                continue
            options_for_item = len(
                [
                    1
                    for option_idx in self.open_option_idxs
                    if item in self.options[option_idx].primaries
                ]
            )
            if options_for_item < min_len:
                min_len = options_for_item
                item_to_cover = item

        assert item_to_cover is not None

        if debug_print:
            print(
                f"\n\nChoosing item {item_to_cover} to cover with {min_len} open options"
            )

        remaining_weight = 0
        for oidx in self.open_option_idxs:
            o = self.options[oidx]
            if item_to_cover in o.primaries:
                remaining_weight += o.primaries[item_to_cover].weight

        # The total remaining weight cannot possibly cover this item; return early,
        idata = self.primary_items[item_to_cover]
        if remaining_weight < idata.bound - idata.slack:
            if debug_print:
                print(
                    f"Remaining weight {remaining_weight} is too small to satisfy limits on {idata}"
                )
            return

        # Iterate over all options with that item that don't push its bound below 0
        # Also! We cannot include any option indices that we've already branched over.
        # I think this is (part of) what Knuth is doing with tweaking.

        already_branched_on: Set[int] = set()
        for option_to_try_idx in self.open_option_idxs:
            option_to_try = self.options[option_to_try_idx]
            if item_to_cover in option_to_try.primaries:
                if debug_print:
                    print(f"Trying option {option_to_try_idx}: {option_to_try}")
                new_primary_items = copy.deepcopy(self.primary_items)
                new_secondary_items = copy.deepcopy(self.secondary_items)
                covered_items = set()
                colored_items: Dict[str, int] = {}

                for item, poption_data in option_to_try.primaries.items():
                    pitem_data = new_primary_items[item]
                    weight = poption_data.weight
                    pitem_data.bound -= weight
                    if pitem_data.bound == 0:
                        del new_primary_items[item]
                        covered_items.add(item)
                    if pitem_data.bound < 0:
                        assert False

                for item, soption_data in option_to_try.secondaries.items():
                    sitem_data = new_secondary_items[item]
                    color = soption_data.color
                    if color is not None:
                        sitem_data.color = color
                        colored_items[item] = color

                new_option_idxs: List[int] = []
                if debug_print:
                    print(f"covered items: {covered_items}")
                    print(f"colored items: {colored_items}")
                for option_idx in self.open_option_idxs:
                    if option_idx == option_to_try_idx:
                        continue
                    if option_idx in already_branched_on:
                        continue
                    compatible = True
                    if (
                        not self.options[option_idx]
                        .primaries.keys()
                        .isdisjoint(covered_items)
                    ):
                        compatible = False
                        continue

                    for item_name, poption_data in self.options[
                        option_idx
                    ].primaries.items():
                        if poption_data.weight > new_primary_items[item_name].bound:
                            compatible = False
                            break

                    for item_name, soption_data in self.options[
                        option_idx
                    ].secondaries.items():
                        if item_name in colored_items:
                            if soption_data.color != colored_items[item_name]:
                                compatible = False
                                break

                    if compatible:
                        new_option_idxs.append(option_idx)

                already_branched_on.add(option_to_try_idx)
                if debug_print:
                    print(
                        f"new items: {new_primary_items}, secondary {new_secondary_items}, new option idxs: {new_option_idxs}"
                    )
                new_problem = Problem(
                    new_primary_items,
                    new_secondary_items,
                    self.options,
                    new_option_idxs,
                )
                new_problem.solve_(current_solution + [option_to_try_idx])

    def solve(self) -> None:
        return self.solve_([])


def get_unique_color(seen_colors: Set[int]) -> int:
    if not seen_colors:
        return 1
    return max(seen_colors) + 1


def load_problem(filename: str) -> Problem:
    primary_items: Dict[str, PrimaryItemData] = {}
    secondary_items: Dict[str, SecondaryItemData] = {}
    options: List[Option] = []

    seen_colors: Set[int] = set()

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
                item_name, lower, upper = m.groups()
                upper, lower = int(upper), int(lower)
            else:
                item_name, upper, lower = item, 1, 1

            if lower < 0:
                raise ValueError(f"Bad lower bound {lower}")
            if upper < lower or upper == 0:
                raise ValueError(f"Bad upper bound {upper}")

            assert upper >= lower
            assert lower >= 0
            assert upper > 0

            if in_primary_items_section:
                primary_items[item_name] = PrimaryItemData(
                    bound=upper, slack=upper - lower
                )
            else:
                secondary_items[item_name] = SecondaryItemData(color=None)

        for line in lines[1:]:
            option = line.strip().split(" ")
            option_primaries: Dict[str, PrimaryOptionData] = {}
            option_secondaries: Dict[str, SecondaryOptionData] = {}

            for item in option:
                got_weight, got_color = False, False
                m1 = re.match("([a-zA-Z0-9_]+)=([0-9]+)", item)
                m2 = re.match("([a-zA-Z0-9_]+):([0-9]+)", item)
                if m1:
                    item_name, weight_s = m1.groups()
                    weight = int(weight_s)
                    if weight == 0:
                        raise ValueError("Illegal weight 0")
                    got_weight = True
                    # print("got weight")
                elif m2:
                    item_name, color_s = m2.groups()
                    color = int(color_s)
                    seen_colors.add(color)
                    got_color = True
                    # print("got color")
                else:
                    item_name, weight, color = item, 1, None

                if item_name not in primary_items and item_name not in secondary_items:
                    # print(items)
                    raise KeyError(f"Unknown item {item_name} in option {option}")

                if item_name in primary_items:
                    if got_color:
                        raise ValueError(
                            f"Got color {color} for primary item {item_name}"
                        )
                    option_primaries[item_name] = PrimaryOptionData(weight=weight)

                if item_name in secondary_items:
                    if got_weight:
                        # print(item, option)
                        raise ValueError(
                            f"Got weight {weight} for secondary item {item_name}"
                        )
                    if color is None:
                        color = get_unique_color(seen_colors)
                        seen_colors.add(color)
                    option_secondaries[item_name] = SecondaryOptionData(color=color)

            options.append(
                Option(primaries=option_primaries, secondaries=option_secondaries)
            )

    open_option_idxs = list(range(len(options)))

    return Problem(primary_items, secondary_items, options, open_option_idxs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()

    problem = load_problem(args.input_file)

    if debug_print:
        print(json.dumps(asdict(problem), indent=2))

    problem.solve()

    print(f"Found {num_solutions} total solutions")


if __name__ == "__main__":
    main()
