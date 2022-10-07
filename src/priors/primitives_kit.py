import sys
import copy
import itertools
import numpy as np
from typing import TypeVar, Callable
from random import randint
from collections.abc import Iterable

sys.path.append('..')
from priors.primitives import Grid, Layer, Object


class PrimitivesKit:

    Grid = Grid
    Object = Object
    class Color(int): pass
    Size = tuple[int, int]
    T = TypeVar('T')
    S = TypeVar('S')

    @staticmethod
    def render(base: Grid, objects: list[Object] | None = []) -> Grid:
        out = Grid(np.copy(base)) if base is not None else Grid.unicolor((1, 1), 0)
        for i, j in itertools.product(range(out.shape[0]), range(out.shape[1])):
            for o in reversed(objects or []):
                o = o if o is not None else Object()
                oi = i - o.pos[0]
                oj = j - o.pos[1]
                if oi in range(o.shape[0]) and oj in range(o.shape[1]):
                    if o[oi, oj] >= 0:
                        out[i, j] = o[oi, oj]
                        break
        
        return out
    
    # @staticmethod
    # def merge_objectss(objs: list[Object]) -> Object:
    #     maxx = max(o.pos[1] + o.shape[1] for o in objs)
    #     maxy = max(o.pos[0] + o.shape[0] for o in objs)
    #     bg = Grid.transparent((maxy, maxx))
    #     return Object(PrimitivesKit.render(bg, objs))
    
    # @staticmethod
    # def stack_objs_horizontal(objs: list[Object]) -> list[Object]:
    #     stacked = []
    #     stack_pos = 0
    #     objs = [o for o in objs if o is not None] if objs else []
    #     for o in objs:
    #         o.pos = tuple([o.pos[0], stack_pos])
    #         stack_pos += o.shape[1]
    #         stacked.append(o)

    #     return stacked
    
    # @staticmethod
    # def stack_objs_vertical(objs: list[Object]) -> list[Object]:
    #     stacked = []
    #     stack_pos = 0
    #     objs = [o for o in objs if o is not None] if objs else []
    #     for o in objs:
    #         o.pos = tuple([stack_pos, o.pos[1]])
    #         stack_pos += o.shape[0]
    #         stacked.append(o)

    #     return stacked
    
    # @staticmethod
    # def grid_of_obj(obj: Object) -> Grid:
    #     return obj
    
    # @staticmethod
    # def obj_from_grid(grid: Grid) -> Object:
    #     return Object(grid)

    @staticmethod
    def create_unicolor_grid(size: Size, color: Color) -> Grid:
        return Grid.unicolor(size or (1, 1), color or 0)

    @staticmethod
    def create_transparent_grid(size: Size) -> Grid:
        return Grid.transparent(size or (1, 1))

    @staticmethod
    def get_size(grid: Grid) -> Size:
        return grid.shape if grid is not None else (0, 0)
    
    # @staticmethod
    # def get_objects(grid: Grid, colors: list[Color] | None = None) -> list[Object]:
    #     return grid.objects(colors=colors) if grid is not None else []
    
    # @staticmethod
    # def get_unicolor_objects(grid: Grid, colors: list[Color] | None = None) -> list[Object]:
    #     return grid.unicolor_objects(colors=colors) if grid is not None else []
    
    # @staticmethod
    # def get_mass(grid: Grid) -> int:
    #     return grid.mass() if grid is not None else 0
    
    @staticmethod
    def make_size(l: int, w: int) -> Size:
        return (l, w)

    @staticmethod
    def clrs_by_majority(grid: Grid) -> list[Color]:
        return grid.colors_by_majority() if grid is not None else []

    @staticmethod
    def frgnd_clrs_by_majority(grid: Grid) -> list[Color]:
        return grid.colors_by_majority()[1:] if grid is not None else []

    @staticmethod
    def select_first(lst: list[T]) -> T:
        return lst[0] if lst else None
    
    @staticmethod
    def select_last(lst: list[T]) -> T:
        return lst[-1] if lst else None
    
    @staticmethod
    def select_first_n(lst: list[T], n: int) -> list[T]:
        return lst[:n] if lst and n else []
    
    @staticmethod
    def select_last_n(lst: list[T], n: int) -> list[T]:
        return lst[-n:] if lst and n else []
    
    @staticmethod
    def random_color() -> Color:
        return PrimitivesKit.Color(randint(0, 9))

    # @staticmethod
    # def add(x: int = 0, y: int = 0) -> int:
    #     return (x or 0) + (y or 0)
    
    # @staticmethod
    # def sub(x: int = 0, y: int = 0) -> int:
    #     return (x or 0) - (y or 0)
    
    # @staticmethod
    # def abs(x: int = 0) -> int:
    #     return -x if x < 0 else x
    
    # @staticmethod
    # def mul(x: int = 0, y: int = 0) -> int:
    #     return x * y
    
    # @staticmethod
    # def sum(l: list[int] = []) -> int:
    #     return sum([i or 0 for i in l]) if l else 0
    
    @staticmethod
    def pass_through(x: T) -> T | None:
        return x 

    # @staticmethod
    # def count(lst: list[T]) -> int:
    #     return len(lst) if lst else 0
    
    # @staticmethod
    # def is_eq(x: T, y: T) -> bool:
    #     return x == y
    
    # @staticmethod
    # def is_lt(x: T, y: T) -> bool:
    #     return x < y
    
    # @staticmethod
    # def is_gt(x: T, y: T) -> bool:
    #     return x > y
    
    # @staticmethod
    # def is_lte(x: T, y: T) -> bool:
    #     return x <= y
    
    # @staticmethod
    # def is_gte(x: T, y: T) -> bool:
    #     return x >= y
    
    # @staticmethod
    # def map_list(func: Callable[[T], S], l: list[T]) -> list[S]:
    #     return list(map(func, l))
    
    # @staticmethod
    # def duplicate_item(x: T) -> list[T]:
    #     return [copy.deepcopy(x) for _ in range(2)]
    
    # @staticmethod
    # def duplicate_items(x: list[T]) -> list[T]:
    #     return [copy.deepcopy(i) for _ in range(2) for i in x]
    
    
