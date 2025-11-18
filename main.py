from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from typing import List, Dict, Set, Any, Tuple
from enum import Enum
import argparse
from collections import defaultdict


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent: List[int] = [i for i in range(size)]
        self.rank: List[int] = [0] * size

    def find(self, i: int) -> int:
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])  # Path compression
        return self.parent[i]

    def union(self, a: int, b: int) -> None:
        rootA: int = self.find(a)
        rootB: int = self.find(b)

        if rootA != rootB:
            # Union by rank
            if self.rank[rootA] < self.rank[rootB]:
                self.parent[rootA] = rootB
            elif self.rank[rootA] > self.rank[rootB]:
                self.parent[rootB] = rootA
            else:
                self.parent[rootB] = rootA
                self.rank[rootA] += 1

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def count_sets(self) -> int:
        return sum(1 for i in range(len(self.parent)) if i == self.parent[i])

    def get_set_elements(self, i: int) -> List[int]:
        root = self.find(i)
        return [x for x in range(len(self.parent)) if self.find(x) == root]

    def get_set_cardinalities(self) -> Dict[int, int]:
        answer: Dict[int, int] = defaultdict(int)
        for i in range(len(self.parent)):
            answer[self.find(i)] += 1
        return answer


@dataclass
class Params:
    weighted_solver: bool
    region_counts: bool


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


@dataclass
class UnequalRegion:
    pass


class SumOperator(Enum):
    Equal = 0
    Greater = 1
    Less = 2


@dataclass
class SumRegion:
    target: int
    operator: SumOperator


type RegionType = EmptyRegion | EqualsRegion | SumRegion | UnequalRegion


@dataclass
class Region:
    idx: int
    kind: RegionType
    indices: List[GridLoc]
    skip_because_zero_region: bool


@dataclass
class UpdatedRegion:
    """Data we need to decide whether to reject a candidate placement of a domino
    that intersects a region.
    """

    kind: RegionType  # with updated target
    size: int
    pips_placed: List[int]


@dataclass
class GridInfo:
    all_positions: List[GridLoc]
    all_positions_set: Set[GridLoc]
    max_x: int
    max_y: int
    grid2region: Dict[GridLoc, Region]
    all_sum_regions: List[Region]
    all_sum_equal_region_idxs: Set[int]
    all_domino_end_pips_sorted_desc: List[int]
    grid2idx: Dict[GridLoc, int]
    graph_edges: List[Tuple[int, int]]
    odd_region_result_memo: Dict[Tuple[GridLoc, GridLoc], bool]
    residual_region: Region


class Orientation(Enum):
    Horizontal = 0
    Vertical = 1


def would_result_in_odd_region(gi: GridInfo, p1: GridLoc, p2: GridLoc) -> bool:
    if (p1, p2) in gi.odd_region_result_memo:
        return gi.odd_region_result_memo[(p1, p2)]
    uf = UnionFind(len(gi.all_positions))
    p1_idx = gi.grid2idx[p1]
    p2_idx = gi.grid2idx[p2]

    uf.union(p1_idx, p2_idx)

    for n1, n2 in gi.graph_edges:
        if n1 != p1_idx and n1 != p2_idx and n2 != p1_idx and n2 != p2_idx:
            uf.union(n1, n2)

    cardinalities = uf.get_set_cardinalities()
    for cardinality in cardinalities.values():
        if cardinality % 2 == 1:
            gi.odd_region_result_memo[(p1, p2)] = True
            gi.odd_region_result_memo[(p2, p1)] = True
            return True

    gi.odd_region_result_memo[(p1, p2)] = False
    gi.odd_region_result_memo[(p2, p1)] = False
    return False


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
            skip_because_zero_region = False
            match region["type"]:
                case "equals":
                    kind = EqualsRegion()
                case "unequal":
                    kind = UnequalRegion()
                case "sum":
                    if region["target"] == 0:
                        skip_because_zero_region = True
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

            regions.append(Region(idx, kind, indices, skip_because_zero_region))

        return cls(regions, dominoes)

    def grid_info(self) -> GridInfo:
        all_positions = []
        grid2region = {}
        all_sum_regions = []
        all_sum_equal_region_idxs = set()
        total_sum_equal_regions_sum = 0
        residual_indices = set()

        for region in self.regions:
            is_residual = True
            match region.kind:
                case SumRegion():
                    if not region.skip_because_zero_region:
                        all_sum_regions.append(region)
                        if region.kind.operator == SumOperator.Equal:
                            all_sum_equal_region_idxs.add(region.idx)
                            total_sum_equal_regions_sum += region.kind.target
                            is_residual = False
            if is_residual:
                residual_indices.update(region.indices)

            for gridloc in region.indices:
                all_positions.append(gridloc)
                grid2region[gridloc] = region

        grid2idx: Dict[GridLoc, int] = {}
        all_positions = list(sorted(all_positions))

        for idx, pos in enumerate(all_positions):
            grid2idx[pos] = idx

        max_x = max([pos.x for pos in all_positions])
        max_y = max([pos.y for pos in all_positions])

        all_domino_end_pips = []
        for domino in self.dominoes:
            all_domino_end_pips.append(domino.end1)
            all_domino_end_pips.append(domino.end2)

        all_domino_end_pips = list(reversed(sorted(all_domino_end_pips)))
        total_pips_sum = sum(all_domino_end_pips)

        set_all_positions = set(all_positions)
        graph_edges = []
        for pos in all_positions:
            for dx, dy in [(0, 1), (1, 0)]:
                neighbor = GridLoc(x=pos.x + dx, y=pos.y + dy)
                if neighbor in set_all_positions:
                    graph_edges.append((grid2idx[pos], grid2idx[neighbor]))

        residual_region = Region(
            idx=max([r.idx for r in self.regions]) + 1000,
            kind=SumRegion(
                operator=SumOperator.Equal,
                target=total_pips_sum - total_sum_equal_regions_sum,
            ),
            indices=list(residual_indices),
            skip_because_zero_region=total_pips_sum == total_sum_equal_regions_sum,
        )

        return GridInfo(
            all_positions,
            set_all_positions,
            max_x,
            max_y,
            grid2region,
            all_sum_regions,
            all_sum_equal_region_idxs,
            all_domino_end_pips,
            grid2idx,
            graph_edges,
            odd_region_result_memo={},
            residual_region=residual_region,
        )

    def get_sum_greater_upper_bound(self, gi: GridInfo, region_size: int) -> int:
        return sum(gi.all_domino_end_pips_sorted_desc[:region_size])

    def generate_items(self, gi: GridInfo, params: Params) -> None:
        primaries, secondaries = [], []
        for gridloc in gi.all_positions:
            primaries.append(f"p_{gridloc.x}_{gridloc.y}")

        for idx, _domino in enumerate(self.dominoes):
            primaries.append(f"d_{idx}")

        regions = self.regions
        if params.weighted_solver:
            regions = regions + [gi.residual_region]

        for region in regions:
            if region.skip_because_zero_region:
                continue
            region_size = len(region.indices)
            match region.kind:
                case EqualsRegion():
                    secondaries.append(f"R_{region.idx}")
                case UnequalRegion():
                    min_pips = gi.all_domino_end_pips_sorted_desc[-1]
                    max_pips = gi.all_domino_end_pips_sorted_desc[0]
                    for pip in range(min_pips, max_pips + 1):
                        secondaries.append(f"R_{region.idx}W_{pip}")
                case SumRegion(target=m, operator=op):
                    match op:
                        case SumOperator.Equal:
                            lower, upper = m, m
                            if params.region_counts:
                                primaries.append(
                                    f"R_{region.idx}_count[{region_size}:{region_size}]"
                                )
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

                    primaries.append(f"R_{region.idx}[{lower}:{upper}]")

                    if not params.weighted_solver:
                        for domino in self.dominoes:
                            for end, pips in enumerate([domino.end1, domino.end2]):
                                secondaries.append(
                                    f"E_{domino.idx}_{end}R_{region.idx}"
                                )
                                for p in range(1, pips + 1):
                                    primaries.append(
                                        f"#E_{domino.idx}_{end}R_{region.idx}_W_{p}"
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
        """
        Returns None if a domino in the given orientation cannot be placed with
        its first (top or left) end at [start_pos].
        """
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

    def sum_region_no_weights(
        self, gi: GridInfo, pos: GridLoc, domino: Domino, end: int, row: List[str]
    ) -> None:
        sum_regions_this_end_is_not_in = list(gi.all_sum_regions)

        region = gi.grid2region[pos]
        match region.kind:
            case SumRegion():
                if not region.skip_because_zero_region:
                    row.append(f"E_{domino.idx}_{end}R_{region.idx}:1")
                    sum_regions_this_end_is_not_in.remove(region)

        for region in sum_regions_this_end_is_not_in:
            row.append(f"E_{domino.idx}_{end}R_{region.idx}:0")

    def sum_region_weights(
        self,
        gi: GridInfo,
        p1: GridLoc,
        p2: GridLoc,
        d1: int,
        d2: int,
        row: List[str],
    ) -> None:
        def gen(r1, r2):
            pips_in_region = []

            if r1.idx == r2.idx:
                pips_in_region = [(r1, d1 + d2)]
            else:
                pips_in_region = [(r1, d1), (r2, d2)]

            for region, pips in pips_in_region:
                if pips == 0:
                    continue
                match region.kind:
                    case SumRegion():
                        if not region.skip_because_zero_region:
                            row.append(f"R_{region.idx}={pips}")

        r1 = gi.grid2region[p1]
        r2 = gi.grid2region[p2]
        gen(r1, r2)

        dummy = Region(
            idx=-1, kind=EmptyRegion(), indices=[], skip_because_zero_region=True
        )
        res1, res2 = dummy, dummy
        if p1 in gi.residual_region.indices:
            res1 = gi.residual_region
        if p2 in gi.residual_region.indices:
            res2 = gi.residual_region

        gen(res1, res2)

    def unequal_region(
        self,
        gi: GridInfo,
        pos: GridLoc,
        domino: Domino,
        end: int,
    ) -> str | None:
        region = gi.grid2region[pos]
        match region.kind:
            case UnequalRegion():
                pips = [domino.end1, domino.end2][end]
                return f"R_{region.idx}W_{pips}"

        return None

    def get_affected_regions(
        self, gi: GridInfo, p1: GridLoc, p2: GridLoc, d1: int, d2: int
    ) -> List[UpdatedRegion]:
        def gen(region1, region2, answer):
            if region1.idx == region2.idx:
                answer += [self.updated_region_if_placed(gi, region1, [d1, d2], 2)]
            else:
                answer += [
                    self.updated_region_if_placed(gi, region1, [d1], 1),
                    self.updated_region_if_placed(gi, region2, [d2], 1),
                ]

        answer = []
        region1 = gi.grid2region[p1]
        region2 = gi.grid2region[p2]
        gen(region1, region2, answer)

        dummy = Region(
            idx=-1, kind=EmptyRegion(), indices=[], skip_because_zero_region=True
        )
        res1, res2 = dummy, dummy
        if p1 in gi.residual_region.indices:
            res1 = gi.residual_region
        if p2 in gi.residual_region.indices:
            res2 = gi.residual_region

        gen(res1, res2, answer)

        return answer

    def updated_region_if_placed(
        self,
        gi: GridInfo,
        region: Region,
        pips_placed: List[int],
        total_num_dominos_placed: int,
    ) -> UpdatedRegion:
        total_num_pips_placed = sum(pips_placed)
        kind: RegionType
        match region.kind:
            case SumRegion(target=m, operator=op):
                kind = SumRegion(target=m - total_num_pips_placed, operator=op)
            case other:
                kind = other
        return UpdatedRegion(
            kind,
            size=len(region.indices) - total_num_dominos_placed,
            pips_placed=pips_placed,
        )

    def sorted_all_pips_except(self, domino: Domino) -> List[int]:
        all_dominos_except = [d for d in self.dominoes if d != domino]
        all_pips_except = [pip for d in all_dominos_except for pip in [d.end1, d.end2]]

        return list(sorted(all_pips_except))

    def cannot_possibly_be_viable(
        self, region: UpdatedRegion, pips_left: List[int]
    ) -> bool:
        top_n_pips_left = sum(pips_left[::-1][: region.size])
        bot_n_pips_left = sum(pips_left[: region.size])
        match region.kind:
            case SumRegion(target=m, operator=SumOperator.Equal):
                if m < 0:
                    return True
                if top_n_pips_left < m:
                    return True
                if bot_n_pips_left > m:
                    return True
            case SumRegion(target=m, operator=SumOperator.Greater):
                if top_n_pips_left <= m:
                    return True
            case SumRegion(target=m, operator=SumOperator.Less):
                if bot_n_pips_left >= m:
                    return True
            case EqualsRegion():
                pips_placed = set(region.pips_placed)
                if len(pips_placed) > 1:
                    return True
                pip_placed = pips_placed.pop()
                m = pips_left.count(pip_placed)
                if m < region.size:
                    return True

        return False

    def add_region_counts(
        self, gi: GridInfo, p1: GridLoc, p2: GridLoc, row: List[str]
    ) -> None:
        region1 = gi.grid2region[p1]
        region2 = gi.grid2region[p2]
        if region1.idx == region2.idx:
            if not region1.skip_because_zero_region:
                match region1.kind:
                    case SumRegion(operator=SumOperator.Equal):
                        row.append(f"R_{region1.idx}_count=2")
        else:
            for region in [region1, region2]:
                if not region.skip_because_zero_region:
                    match region.kind:
                        case SumRegion(operator=SumOperator.Equal):
                            row.append(f"R_{region.idx}_count=1")

    def generate_options(self, gi: GridInfo, params: Params) -> None:
        answer = []
        num_odd_region_placements_rejected = 0
        num_options_rejected = 0

        all_start_pos = [
            GridLoc(x, y) for x in range(gi.max_x + 1) for y in range(gi.max_y + 1)
        ]
        for domino in self.dominoes:
            remaining_pips = self.sorted_all_pips_except(domino)
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

                        if would_result_in_odd_region(gi, p1, p2):
                            num_odd_region_placements_rejected += 1
                            continue

                        # Here we apply a bunch of rules that try to figure out
                        # if this placement is immediately impossible.
                        affected_regions = self.get_affected_regions(gi, p1, p2, d1, d2)

                        placement_is_viable = True
                        for affected_region in affected_regions:
                            if self.cannot_possibly_be_viable(
                                affected_region, remaining_pips
                            ):
                                placement_is_viable = False
                                break

                        if not placement_is_viable:
                            num_options_rejected += 1
                            continue

                        row = [
                            f"d_{domino.idx}",
                            f"p_{p1.x}_{p1.y}",
                            f"p_{p2.x}_{p2.y}",
                        ]

                        if params.region_counts:
                            self.add_region_counts(gi, p1, p2, row)

                        # This is to make it easier to read off the solution.
                        if flipped:
                            row[1], row[2] = row[2], row[1]

                        if params.weighted_solver:
                            self.sum_region_weights(gi, p1, p2, d1, d2, row)
                        else:
                            self.sum_region_no_weights(gi, p1, domino, end1, row)
                            self.sum_region_no_weights(gi, p2, domino, end2, row)

                        uer1 = self.unequal_region(gi, p1, domino, end1)
                        uer2 = self.unequal_region(gi, p2, domino, end2)

                        if uer1 == uer2 and uer1 is not None:
                            # Placing a domino here will immediately violate an
                            # unequal region constraint; skip.
                            continue

                        if uer1 is not None:
                            row.append(uer1)
                        if uer2 is not None:
                            row.append(uer2)

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

        if not params.weighted_solver:
            for region in gi.all_sum_regions:
                for domino in self.dominoes:
                    for end, pips in enumerate([domino.end1, domino.end2]):
                        row = [f"E_{domino.idx}_{end}R_{region.idx}:0"]
                        for p in range(1, pips + 1):
                            row.append(f"#E_{domino.idx}_{end}R_{region.idx}_W_{p}")
                        answer.append(" ".join(row))
                        for p in range(1, pips + 1):
                            answer.append(
                                f"E_{domino.idx}_{end}R_{region.idx}:1 #E_{domino.idx}_{end}R_{region.idx}_W_{p} R_{region.idx}"
                            )

        print(
            f"Rejected {num_options_rejected} options, {num_odd_region_placements_rejected} placements that would result in odd regions",
            file=sys.stderr,
        )
        print("\n".join(answer))

    def translate_less_than_1_regions_into_zero(self) -> None:
        for region in self.regions:
            match region.kind:
                case SumRegion(target=1, operator=SumOperator.Less):
                    print(
                        "Translating region with <1 into a sum region of 0",
                        file=sys.stderr,
                    )
                    region.kind = SumRegion(target=0, operator=SumOperator.Equal)
                    region.skip_because_zero_region = True

    def generate_mcc(self, params: Params) -> None:
        self.translate_less_than_1_regions_into_zero()
        gi = self.grid_info()
        self.generate_items(gi, params)
        self.generate_options(gi, params)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument(
        "--difficulty",
        "-d",
        type=str,
        help="which puzzle in the input file to solve",
    )
    parser.add_argument(
        "--weighted",
        "-w",
        default=False,
        action="store_true",
        help="The solver supports weights, so don't generate auxiliary counting items and options",
    )
    parser.add_argument(
        "--region-counts",
        default=False,
        action="store_true",
        help="Generate constraints for each region saying This region must have exactly this many ends placed in it",
    )
    args = parser.parse_args()

    if args.region_counts and not args.weighted:
        raise ValueError("Region counts require weight support")

    params = Params(weighted_solver=args.weighted, region_counts=args.region_counts)

    puzzle = Puzzle.load(json.load(open(args.input_file))[args.difficulty])
    # print(json.dumps(asdict(puzzle), indent=2))
    puzzle.generate_mcc(params)


if __name__ == "__main__":
    main()
