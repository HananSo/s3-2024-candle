#!/usr/bin/env python3
#
# Copyright (C) 2023 Alexandre Jesus <https://adbjesus.com>, Carlos M. Fonseca <cmfonsec@dei.uc.pt>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import TextIO, Optional, Any
from collections.abc import Iterable, Hashable, Sequence
from dataclasses import dataclass

import logging

Objective = Any
@dataclass
class Point:
    x: int
    y: int

@dataclass
class Village:
    coord: Point
    init_height: int
    burn_rate: int
    v_id: int

    @property
    def cid(self) -> Hashable:
        return self.v_id

VillageList = Sequence[Village]
DistMatrix = tuple[tuple[float, ...], ...]

def manhatten_distance(pointA, pointB):
    dist_sum = abs(pointA.x - pointB.x) + abs(pointA.y - pointB.y)
    return dist_sum

def distance_matrix(villages: VillageList) -> DistMatrix:
    mat = []
    for a in villages:
        row = []
        for b in villages:
            row.append(manhatten_distance(a.coord, b.coord))
        mat.append(tuple(row))
    return tuple(mat)

@dataclass
class Component:
    coord: Point
    current_time: int
    current_height: int
    v_id: int

    @property
    def cid(self) -> Hashable:
        return self.v_id

class LocalMove:
    index:int
    move_type: bool
    insert_id: int

class Solution:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.path = []
        self.used = []
        self.unused = []
        self.burn_rate_list = []
        self.dist_traveled = []
        self.score = 0


    def output(self) -> str:
        """
        Generate the output string for this solution
        """
        output_str = ""
        output_str += str(self.objective()) + "\n"
        for p in self.path:
            output_str += str(p.v_id) + "\n"

        return output_str

    def copy(self) -> Solution:
        """
        Return a copy of this solution.

        Note: changes to the copy must not affect the original
        solution. However, this does not need to be a deepcopy.
        """
        raise NotImplementedError

    def is_feasible_indexes(self, index, path) -> bool:
        """
        Return whether the solution is feasible or not
        """
        flist= []
        for i in range(index, len(path)):
            if self.dist_traveled[i] * self.burn_rate[i] <= path[i].init_height:
                flist.append(1)
            else:
                flist.append(0)
        return flist
    
    def is_feasible(self) -> bool:
        """
        Return whether the solution is feasible or not
        """
        for i in range(0, len(self.path)):
            if self.dist_traveled[i] * self.burn_rate[i] <= self.path[i].init_height:
                continue
            else:
                return False
        return True

    def objective(self) -> Optional[Objective]:
        """
        Return the objective value for this solution if defined, otherwise
        should return None
        """
        score = 0
        for i, village in enumerate(self.path):
            score -= (village.init_height - (self.dist_traveled[i]*village.burn_rate))

        return score

    def lower_bound(self) -> Optional[Objective]:
        """
        Return the lower bound value for this solution if defined,
        otherwise return None
        """
        total = self.score
        for i, village in enumerate(self.unused):
            vil_score = (self.dist_traveled[-1] + self.problem.distance_matrix[self.path[-1].v_id][village.v_id]) * village.burn_rate
            if vil_score < 0:
                continue
            total += (vil_score * -1)
        return total

    def add_moves(self) -> Iterable[Component]:
        """
        Return an iterable (generator, iterator, or iterable object)
        over all components that can be added to the solution
        """
        move_list = []
        for i, village in enumerate(self.unused):
            vil_score = (self.dist_traveled[-1] + self.problem.distance_matrix[self.path[-1].v_id][village.v_id]) * village.burn_rate
            if vil_score < 0:
                continue
            move_list.append(village)
        return move_list

    def local_moves(self) -> Iterable[LocalMove]:
        """
        Return an iterable (generator, iterator, or iterable object)
        over all local moves that can be applied to the solution
        """
        local_moves = []
        for i in range(1, len(self.path)):
            local_moves.append(LocalMove(i, 0, None))
            for j in self.unused:
                    
                local_moves.append(LocalMove(i, 1, j))

        
        return local_moves

    

    def random_local_move(self) -> Optional[LocalMove]:
        """
        Return a random local move that can be applied to the solution.

        Note: repeated calls to this method may return the same
        local move.
        """
        raise NotImplementedError

    def random_local_moves_wor(self) -> Iterable[LocalMove]:
        """
        Return an iterable (generator, iterator, or iterable object)
        over all local moves (in random order) that can be applied to
        the solution.
        """
        raise NotImplementedError
            
    def heuristic_add_move(self) -> Optional[Component]:
        """
        Return the next component to be added based on some heuristic
        rule.
        """
        raise NotImplementedError

    def add(self, village: Village) -> None:
        """
        Add a component to the solution.

        Note: this invalidates any previously generated components and
        local moves.
        """
        
        self.used.append(village)
        self.unused.remove(village)
        self.dist_traveled.append(self.dist_traveled[-1] + self.problem.distance_matrix[self.path[-1].v_id][village.v_id])
        self.score = self.objective()
        self.burn_rate_list.append(self.burn_rate_list[-1] + village.burn_rate)
        self.path.append(village)


    def step(self, lmove: LocalMove) -> None:
        """
        Apply a local move to the solution.

        Note: this invalidates any previously generated components and
        local moves.
        """
        if lmove.move_type == 1:
            test_path = self.path.copy()
            test_path.insert(i, j)
            findexes = self.is_feasible_indexes(i, test_path)
            f

    def objective_incr_local(self, lmove: LocalMove) -> Optional[Objective]:
        """
        Return the objective value increment resulting from applying a
        local move. If the objective value is not defined after
        applying the local move return None.
        """
        raise NotImplementedError

    def lower_bound_incr_add(self, component: Component) -> Optional[Objective]:
        """
        Return the lower bound increment resulting from adding a
        component. If the lower bound is not defined after adding the
        component return None.
        """
        raise NotImplementedError

    def perturb(self, ks: int) -> None:
        """
        Perturb the solution in place. The amount of perturbation is
        controlled by the parameter ks (kick strength)
        """
        raise NotImplementedError

    def components(self) -> Iterable[Component]:
        """
        Returns an iterable to the components of a solution
        """
        return self.path

class Problem:
    def __init__(self, villages: VillageList) -> None:
        self.nnodes = len(villages)
        self.villages = villages
        self.dist = distance_matrix(villages)
        
    @classmethod
    def from_textio(cls, f: TextIO) -> Problem:
        """ 
        Create a problem from a text I/O source `f`
        """
        n = int(f.readline())
        start_village = Village(Point(map(int, f.readline().split())), 0, 0)
        villages = [start_village]
        for i in range(1, n):
            x, y, init_height, burn_rate = map(int, f.readline().split())
            villages.append(Village(Point(x, y), init_height, burn_rate, i))
        return cls(n, villages)

    def empty_solution(self) -> Solution:
        """
        Create an empty solution (i.e. with no components).
        """
        return Solution(self)


if __name__ == '__main__':
    import api.solvers as apis
    from time import perf_counter
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--log-level',
                        choices=['critical', 'error', 'warning', 'info', 'debug'],
                        default='warning')
    parser.add_argument('--log-file', type=argparse.FileType('w'), default=sys.stderr)
    parser.add_argument('--csearch',
                        choices=['beam', 'grasp', 'greedy', 'heuristic', 'as', 'mmas', 'none'],
                        default='none')
    parser.add_argument('--cbudget', type=float, default=5.0)
    parser.add_argument('--lsearch',
                        choices=['bi', 'fi', 'ils', 'rls', 'sa', 'none'],
                        default='none')
    parser.add_argument('--lbudget', type=float, default=5.0)
    parser.add_argument('--input-file', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--output-file', type=argparse.FileType('w'), default=sys.stdout)
    args = parser.parse_args()

    logging.basicConfig(stream=args.log_file,
                        level=args.log_level.upper(),
                        format="%(levelname)s;%(asctime)s;%(message)s")

    p = Problem.from_textio(args.input_file)
    s: Optional[Solution] = p.empty_solution()

    start = perf_counter()

    if s is not None:
        if args.csearch == 'heuristic':
            s = apis.heuristic_construction(s)
        elif args.csearch == 'greedy':
            s = apis.greedy_construction(s)
        elif args.csearch == 'beam':
            s = apis.beam_search(s, 10)
        elif args.csearch == 'grasp':
            s = apis.grasp(s, args.cbudget, alpha = 0.01)
        elif args.csearch == 'as':
            ants = [s]*100
            s = apis.ant_system(ants, args.cbudget, beta = 5.0, rho = 0.5, tau0 = 1 / 3000.0)
        elif args.csearch == 'mmas':
            ants = [s]*100
            s = apis.mmas(ants, args.cbudget, beta = 5.0, rho = 0.02, taumax = 1 / 3000.0, globalratio = 0.5)

    if s is not None:
        if args.lsearch == 'bi':
            s = apis.best_improvement(s, args.lbudget)
        elif args.lsearch == 'fi':
            s = apis.first_improvement(s, args.lbudget) 
        elif args.lsearch == 'ils':
            s = apis.ils(s, args.lbudget)
        elif args.lsearch == 'rls':
            s = apis.rls(s, args.lbudget)
        elif args.lsearch == 'sa':
            s = apis.sa(s, args.lbudget, 30)

    end = perf_counter()

    if s is not None:
        print(s.output(), file=args.output_file)
        if s.objective() is not None:
            logging.info(f"Objective: {s.objective():.3f}")
        else:
            logging.info("Objective: None")
    else:
        logging.info("Objective: no solution found")

    logging.info(f"Elapsed solving time: {end-start:.4f}")

