import operator
import itertools
import numpy as np
from turtle import bgcolor
from matplotlib.pyplot import grid
from collections import defaultdict
from collections.abc import Iterable

def iterable(x):
    return x if isinstance(x, Iterable) else [x]

class Grid(np.ndarray):
    def __new__(cls, input_array = [[]]):
        return np.asarray(input_array).view(cls)

    def __array_finalize__(self, obj) -> None:
        if obj is None: return
        # This attribute should be maintained!
        self.attr = getattr(obj, 'dtype', int)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):  # this method is called whenever you use a ufunc
        f = {
            "reduce": ufunc.reduce,
            "accumulate": ufunc.accumulate,
            "reduceat": ufunc.reduceat,
            "outer": ufunc.outer,
            "at": ufunc.at,
            "__call__": ufunc,
        }
        output = Grid(f[method](*(i.view(np.ndarray) if isinstance(i, Grid) else i for i in inputs), **kwargs))  # convert the inputs to np.ndarray to prevent recursion, call the function, then cast it back as ExampleTensor
        output.__dict__ = self.__dict__  # carry forward attributes
        return output
    
    @classmethod
    def unicolor(cls, size, color):
        return cls(np.full(size, fill_value=color, dtype=int))
    
    @classmethod
    def transparent(cls, size):
        return cls.unicolor(size, color=-1)
    
    def overlay(self, overlay, pos=(0, 0)):
        underlay = self[pos[0]:pos[0] + overlay.shape[0],
                        pos[1]:pos[1] + overlay.shape[1]]
        patch = np.where(overlay < 0, underlay, overlay)
        self[pos[0]:pos[0] + overlay.shape[0],
            pos[1]:pos[1] + overlay.shape[1]] = patch
    
    def invert_transparency(self, color=0):
        self[self < 0] = -2
        self[self >= 0] = -1
        self[self < -1] = color
    
    def unify_color(self, color=0):
        self[self >= 0] = color

    def paint(self, color, mask, pos=(0, 0)):
        overlay = mask.copy()
        overlay.invert_transparency(color=color)
        self.overlay(overlay, pos=pos)

    def hist(self):
        d = defaultdict(int)
        for i, j in np.ndindex(self.shape):
            d[self[i, j]] += 1
        return dict(d)
    
    def switch_color(self, color_to_be_replaced, new_color):
        self[self == color_to_be_replaced] = new_color

    def clear_colors(self, colors):
        for color in colors:
            self.switch_color(color, -1)

    def colors(self):
        return set(self.flatten())
    
    def colors_by_majority(self):
        return [i[0] for i in sorted(self.hist().items(), key=operator.itemgetter(1), reverse=True)]

    def major_color(self):
        return self.colors_by_majority()[0]
    
    def subgrid(self, ulc, lrc):
        return Grid(self[ulc[0]:lrc[0]+1, ulc[1]:lrc[1]+1])

    def foreground_rect(self, bgc=None):
        mc = bgc or self.major_color()
        foreground = np.where(self != mc)
        return (min(foreground[0], default=0), min(foreground[1], default=0)), (max(foreground[0], default=0), max(foreground[1], default=0))

    def filter(self, colors):
        return np.vectorize(lambda c: c if c in colors else -1)(self)

    def wrapping_rect(self):
        foreground = np.where(self >= 0)
        return (min(foreground[0], default=0), min(foreground[1], default=0)), (max(foreground[0], default=0), max(foreground[1], default=0))

    def density_of_color(self, color):
        fgrid = self.filter(colors=iterable(color))
        ulc, lrc = fgrid.wrapping_rect()
        area = abs(lrc[0] - ulc[0] + 1) * abs(lrc[1] - ulc[1] + 1)
        count = self.hist().get(color, 0)
        return count / area
    
    def objects(self, colors=None, edge_only=False):
        objs = []
        colors = colors or self.colors_by_majority()
        stained = Grid(np.full(self.shape, fill_value=-1, dtype=int))
        
        def is_candidate(p):
            c = self[p[0], p[1]]
            return c >= 0 and c in colors
        
        def candidate_neighbors(p):
            edge_neigbors = {()}
            xs = range(max(0, p[0]-1), min(self.shape[0], p[0]+2))
            ys = range(max(0, p[1]-1), min(self.shape[1], p[1]+2))
            return {t for t in itertools.product(xs, ys) if is_candidate(t) and t != p}

        def get_stain(points):
            for p in points:
                s = stained[p[0], p[1]]
                if s >= 0: return s

        def set_stain(points, s):
            for p in points:
                stained[p[0], p[1]] = s
        
        def stain_unstained_neighbors(p, s):
            set_stain({p}, s)
            objs[s].add(p)
            unstained_neighbors = {n for n in candidate_neighbors(p) if None == get_stain({n})}
            for n in unstained_neighbors:
                stain_unstained_neighbors(n, s)

        def objectify(points):
            xs, ys = list(zip(*points))
            w, h = (max(xs) - min(xs) + 1), (max(ys) - min(ys) + 1)
            minx, miny = min(xs), min(ys)
            o = Object(np.full((w, h), fill_value=-1, dtype=int))
            o.pos = (minx, miny)
            for p in points:
                o[p[0]-minx, p[1]-miny] = self[p[0], p[1]]
            return o
        
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                p = (i, j)
                if is_candidate(p) and None == get_stain({p}):
                    s = len(objs)
                    objs.append(set())
                    stain_unstained_neighbors(p, s)
        
        return [objectify(o) for o in objs]

    # def foreground_objects(self, edge_only=False):
    #     return self.objects(colors=self.colors_by_majority()[1:], edge_only=edge_only)

    def unicolor_objects(self, colors=None, edge_only=False):
        objs = []
        colors = colors or self.colors_by_majority()
        for color in colors:
            objs.extend(self.objects(colors=[color], edge_only=edge_only))
        
        return objs
    
    # def foreground_unicolor_objects(self, edge_only=False):
    #     return self.unicolor_objects(colors=self.colors_by_majority()[1:], edge_only=edge_only)
    
    def mass(self):
        return np.count_nonzero(self >= 0)

class Object(Grid):

    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)

    def __array_finalize__(self, obj) -> None:
        if obj is None: return
        # This attribute should be maintained!
        self.attr = getattr(obj, 'dtype', int)
        self.pos = (0, 0)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):  # this method is called whenever you use a ufunc
        f = {
            "reduce": ufunc.reduce,
            "accumulate": ufunc.accumulate,
            "reduceat": ufunc.reduceat,
            "outer": ufunc.outer,
            "at": ufunc.at,
            "__call__": ufunc,
        }
        output = Grid(f[method](*(i.view(np.ndarray) if isinstance(i, Grid) else i for i in inputs), **kwargs))  # convert the inputs to np.ndarray to prevent recursion, call the function, then cast it back as ExampleTensor
        output.__dict__ = self.__dict__  # carry forward attributes
        return output
    

class Layer(Object): pass
