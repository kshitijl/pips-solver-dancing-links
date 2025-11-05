from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from typing import List, Dict, Set, Any, Tuple
from enum import Enum
import argparse


@dataclass
class Domino:
    idx: int
    end1: int
    end2: int


@dataclass(order=True, frozen=True)
class GridLoc:
    x: int
    y: int


@dataclass
class EmptyRegion:
    pass


@dataclass
class EqualsRegion:
    pass


class SumOperator(Enum):
    Equal = 0
    Greater = 1
    Less = 2


@dataclass
class SumRegion:
    target: int
    operator: SumOperator


type RegionType = EmptyRegion | EqualsRegion | SumRegion


@dataclass
class Region:
    idx: int
    kind: RegionType
    indices: List[GridLoc]


@dataclass
class GridInfo:
    all_positions: List[GridLoc]
    all_positions_set: Set[GridLoc]
    max_x: int
    max_y: int
    grid2region: Dict[GridLoc, Region]
    all_sum_regions: List[Region]
    all_domino_end_pips_sorted_desc: List[int]


class Orientation(Enum):
    Horizontal = 0
    Vertical = 1


@dataclass
class Puzzle:
    regions: List[Region]
    dominoes: List[Domino]

    @classmethod
    def load(cls, puzzle_dict: Dict[str, Any]) -> Puzzle:
        dominoes = []
        for idx, [x, y] in enumerate(puzzle_dict["dominoes"]):
            dominoes.append(Domino(idx, x, y))

        regions = []
        for idx, region in enumerate(puzzle_dict["regions"]):
            kind: RegionType
            match region["type"]:
                case "equals":
                    kind = EqualsRegion()
                case "sum":
                    kind = SumRegion(region["target"], SumOperator.Equal)
                case "greater":
                    kind = SumRegion(region["target"], SumOperator.Greater)
                case "less":
                    kind = SumRegion(region["target"], SumOperator.Less)
                case "empty":
                    kind = EmptyRegion()
                case other:
                    raise ValueError(f"Unimplemented region type {other}")

            indices = []
            for [x, y] in region["indices"]:
                indices.append(GridLoc(x, y))

            regions.append(Region(idx, kind, indices))

        return cls(regions, dominoes)

    def grid_info(self) -> GridInfo:
        all_positions = []
        grid2region = {}
        all_sum_regions = []

        for region in self.regions:
            match region.kind:
                case SumRegion():
                    all_sum_regions.append(region)

            for gridloc in region.indices:
                all_positions.append(gridloc)
                grid2region[gridloc] = region

        all_positions = list(sorted(all_positions))

        max_x = max([pos.x for pos in all_positions])
        max_y = max([pos.y for pos in all_positions])

        all_domino_end_pips = []
        for domino in self.dominoes:
            all_domino_end_pips.append(domino.end1)
            all_domino_end_pips.append(domino.end2)

        all_domino_end_pips = list(reversed(sorted(all_domino_end_pips)))

        return GridInfo(
            all_positions,
            set(all_positions),
            max_x,
            max_y,
            grid2region,
            all_sum_regions,
            all_domino_end_pips,
        )

    def get_sum_greater_upper_bound(self, gi: GridInfo, region_size: int) -> int:
        return sum(gi.all_domino_end_pips_sorted_desc[:region_size])

    def generate_items(self, gi: GridInfo) -> None:
        primaries, secondaries = [], []
        for gridloc in gi.all_positions:
            primaries.append(f"p_{gridloc.x}_{gridloc.y}")

        for idx, _domino in enumerate(self.dominoes):
            primaries.append(f"d_{idx}")

        for region in self.regions:
            match region.kind:
                case EqualsRegion():
                    secondaries.append(f"R_{region.idx}")
                case SumRegion(target=m, operator=op):
                    match op:
                        case SumOperator.Equal:
                            lower, upper = m, m
                        case SumOperator.Greater:
                            lower, upper = (
                                m + 1,
                                self.get_sum_greater_upper_bound(
                                    gi, len(region.indices)
                                ),
                            )
                        case SumOperator.Less:
                            lower, upper = 0, m - 1

                    assert upper >= lower

                    primaries.append(f"#R_{region.idx}[{lower}:{upper}]")
                    for domino in self.dominoes:
                        for end, pips in enumerate([domino.end1, domino.end2]):
                            secondaries.append(f"E_{domino.idx}_{end}R_{region.idx}")
                            for p in range(1, pips + 1):
                                primaries.append(
                                    f"E_{domino.idx}_{end}R_{region.idx}_W_{p}"
                                )

        print(" ".join(primaries) + " | " + " ".join(secondaries))

    def place_domino(
        self,
        gi: GridInfo,
        start_pos: GridLoc,
        domino: Domino,
        orientation: Orientation,
        flipped: bool,
    ) -> Tuple[GridLoc, GridLoc, int, int, int, int] | None:
        if start_pos not in gi.all_positions_set:
            return None

        if orientation == Orientation.Horizontal:
            answer = GridLoc(start_pos.x, start_pos.y + 1)
        else:
            answer = GridLoc(start_pos.x + 1, start_pos.y)

        if answer not in gi.all_positions_set:
            return None

        e1, e2 = 0, 1
        d1, d2 = domino.end1, domino.end2
        if flipped:
            d1, d2 = d2, d1
            e1, e2 = e2, e1
        return start_pos, answer, d1, d2, e1, e2

    def equals_region(self, gi: GridInfo, pos: GridLoc) -> int | None:
        region = gi.grid2region[pos]
        match region.kind:
            case EqualsRegion():
                return region.idx

        return None

    def sum_region(
        self, gi: GridInfo, pos: GridLoc, domino: Domino, end: int, row: List[str]
    ) -> None:
        sum_regions_this_end_is_not_in = list(gi.all_sum_regions)

        region = gi.grid2region[pos]
        match region.kind:
            case SumRegion():
                row.append(f"E_{domino.idx}_{end}R_{region.idx}:1")
                sum_regions_this_end_is_not_in.remove(region)

        for region in sum_regions_this_end_is_not_in:
            row.append(f"E_{domino.idx}_{end}R_{region.idx}:0")

    def generate_options(self, gi: GridInfo) -> None:
        answer = []
        all_start_pos = [
            GridLoc(x, y) for x in range(gi.max_x + 1) for y in range(gi.max_y + 1)
        ]
        for domino in self.dominoes:
            for orientation in [Orientation.Horizontal, Orientation.Vertical]:
                for flipped in [False, True]:
                    if domino.end1 == domino.end2 and flipped:
                        continue
                    for start_pos in all_start_pos:
                        result = self.place_domino(
                            gi, start_pos, domino, orientation, flipped
                        )
                        if result is None:
                            continue
                        p1, p2, d1, d2, end1, end2 = result

                        row = [
                            f"d_{domino.idx}",
                            f"p_{p1.x}_{p1.y}",
                            f"p_{p2.x}_{p2.y}",
                        ]

                        # This is to make it easier to read off the solution.
                        if flipped:
                            row[1], row[2] = row[2], row[1]

                        self.sum_region(gi, p1, domino, end1, row)
                        self.sum_region(gi, p2, domino, end2, row)

                        ero1 = self.equals_region(gi, p1)
                        ero2 = self.equals_region(gi, p2)

                        if ero1 is not None and ero1 == ero2 and d1 != d2:
                            # Placing a domino here will immediately violate an
                            # equal region constraint; skip.
                            continue

                        if ero1 is not None and ero1 == ero2:
                            assert d1 == d2
                            row.append(f"R_{ero1}:{d1}")
                        else:
                            assert ero1 is None or ero1 != ero2
                            if ero1 is not None:
                                row.append(f"R_{ero1}:{d1}")
                            if ero2 is not None:
                                row.append(f"R_{ero2}:{d2}")

                        answer.append(" ".join(row))

        for region in self.regions:
            match region.kind:
                case SumRegion():
                    for domino in self.dominoes:
                        for end, pips in enumerate([domino.end1, domino.end2]):
                            row = [f"E_{domino.idx}_{end}R_{region.idx}:0"]
                            for p in range(1, pips + 1):
                                row.append(f"E_{domino.idx}_{end}R_{region.idx}_W_{p}")
                            answer.append(" ".join(row))
                            for p in range(1, pips + 1):
                                answer.append(
                                    f"E_{domino.idx}_{end}R_{region.idx}:1 E_{domino.idx}_{end}R_{region.idx}_W_{p} #R_{region.idx}"
                                )

        print("\n".join(answer))

    def has_zeros(self) -> bool:
        for region in self.regions:
            if hasattr(region.kind, "target"):
                if region.kind.target == 0:
                    return True

        return False

    def transform_to_get_rid_of_zeros(self) -> None:
        for region in self.regions:
            if hasattr(region.kind, "target"):
                region.kind.target += len(region.indices)

        for domino in self.dominoes:
            domino.end1 += 1
            domino.end2 += 1

    def generate_mcc(self) -> None:
        if self.has_zeros():
            self.transform_to_get_rid_of_zeros()
        gi = self.grid_info()
        self.generate_items(gi)
        self.generate_options(gi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument(
        "--difficulty",
        "-d",
        type=str,
        help="which puzzle in the input file to solve",
    )
    args = parser.parse_args()
    puzzle = Puzzle.load(json.load(open(args.input_file))[args.difficulty])
    # print(json.dumps(asdict(puzzle), indent=2))
    puzzle.generate_mcc()


if __name__ == "__main__":
    main()
