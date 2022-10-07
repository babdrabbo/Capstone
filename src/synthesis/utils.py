
from types import NoneType, UnionType
from typing import Any, Union, TypeVar, Optional, get_origin, get_args


class Utils:
    @classmethod
    def predicated_remove(Cls, l, predicate=lambda x:False, alt_predicates=[]):
        alt_indices = [None] * len(alt_predicates)
        for i, x in enumerate(l):
            if predicate(x): return l.pop(i)
            else: 
                for pi, p in enumerate(alt_predicates):
                    if alt_indices[pi] == None: alt_indices[pi] = i if p(x) else None
        
        for alt_index in alt_indices:
            if alt_index != None: return l.pop(alt_index)

    @classmethod
    def remove_all(Cls, l, e):
        n = 0
        for i in reversed(range(len(l))):
            if l[i] == e: l.pop(i); n+=1
        return n
    
    @classmethod
    def flatten(Cls, list_of_lists):
        return [x for a_list in list_of_lists for x in a_list]
    

class TWrapper:
    def __init__(self, origin, args):
        self.origin = origin
        self.args = args
    
    def __wrap(self, subst_map={}):
        tupled_args = tuple( 
                arg.__wrap(subst_map) if isinstance(arg, self.__class__) else 
                subst_map[arg] if arg in subst_map else arg 
                for arg in self.args 
                )
        return Union[tupled_args] if self.origin == UnionType else self.origin[tupled_args]

    @classmethod
    def is_wrapped(Cls, t):
        return bool(len(get_args(t)))

    @classmethod
    def unwrap(Cls, t, ends):
        if Cls.is_wrapped(t):
            return Cls(get_origin(t), [Cls.unwrap(arg, ends) for arg in get_args(t)])
        else:
            ends.append(t)
            return t
    
    @classmethod
    def wrap(Cls, w, subst_map={}):
        return w.__wrap(subst_map) if isinstance(w, Cls) else subst_map[w] if w in subst_map else w

class TypesChecker:

    @classmethod
    def is_wrapped(Cls, t):
        return bool(len(get_args(t)))

    @classmethod
    def unwrap(Cls, t, stack, ends):
        if Cls.is_wrapped(t):
            stack.append(get_origin(t))
            for arg in get_args(t):
                Cls.unwrap(arg, stack, ends)
        else:
            ends.append(t)

    @classmethod
    def is_generic(Cls, t):
        return type(t) == TypeVar

    @classmethod
    def is_optional(Cls, t):
        return get_origin(t) in [Union, UnionType] and NoneType in get_args(t)

    @classmethod
    def remove_optional(Cls, t):
        return next(t for t in get_args(t) if t is not NoneType) if Cls.is_optional(t) else t

    @classmethod
    def drop_optional_types(Cls, types):
        return [t for t in types if not Cls.is_optional(t)]

    @classmethod
    def is_list(Cls, t):
        t = Cls.remove_optional(t)
        return t == list or get_origin(t) == list

    @classmethod
    def type_of_list(Cls, t):
        is_opt = Cls.is_optional(t)
        args = get_args(Cls.remove_optional(t))
        return (Optional[args[0]] if is_opt else args[0]) if Cls.is_list(t) and len(args) else None

    @classmethod
    def __reduce(Cls, a, b, bypass=lambda:False):
        for ea in [x for x in a if not bypass(x)]:
            Utils.predicated_remove(b, lambda t:t==ea, [lambda t:t==Any, lambda t:t==Optional[ea], lambda t:t==Optional[Any]]) and a.remove(ea)

    @classmethod
    def __reduction_pass(Cls, a, b, bypass):
        Cls.__reduce(a, b, bypass)
        Cls.__reduce(b, a, bypass)
    
    @classmethod
    def __are_outputs_optionally_compatible(Cls, inlet_type, outlet_type):
        return Cls.is_optional(inlet_type) or not Cls.is_optional(outlet_type)

    @classmethod
    def are_generically_compatible(Cls, tout, gout):
        tout, gout = map(Cls.remove_optional, (tout, gout))
        if tout == gout: return True, {}

        tends, gends = [], []
        tout_w = TWrapper.unwrap(tout, tends)
        gout_w = TWrapper.unwrap(gout, gends)
        if len(tends) != len(gends): return False, {}
        subst_map = {k:v for (k, v) in zip(gends, tends) if Cls.is_generic(k)} if len(tends) == len(gends) else {}

        return TWrapper.wrap(gout_w, subst_map) == tout, subst_map

    @classmethod
    def do_outputs_match(Cls, tout, gout):
        if not Cls.__are_outputs_optionally_compatible(tout, gout):
            return False, {}
            
        return Cls.are_generically_compatible(tout, gout)

    @classmethod
    def apply_substitution(Cls, types, sub_map={}):
        return [TWrapper.wrap(TWrapper.unwrap(t, []), sub_map) for t in types]

    @classmethod
    def __do_inputs_match(Cls, tin, gin):

        tin, gin = list(tin), list(gin)

        # check match of literals and Anys (leave Lists and optionals out)
        Cls.__reduction_pass(tin, gin, bypass = lambda t: t==Any or Cls.is_list(t) or Cls.is_optional(t))
        Cls.__reduction_pass(tin, gin, bypass = lambda t: t==Any or Cls.is_list(t))
        Cls.__reduction_pass(tin, gin, bypass = lambda t: t==Any)

        # match remaining types for lists in genotype inputs with individuals of the same type from the template inputs
        for glist in [t for t in gin if Cls.is_list(t)]:
            Utils.remove_all(tin, Cls.remove_optional(Cls.type_of_list(glist))) and gin.remove(glist)

        # drop optionals
        tin, gin = map(Cls.drop_optional_types, (tin, gin))
        
        # check remainings are emtpy and equal
        return tin == gin == []

    @classmethod
    def do_signatures_match(Cls, tsig, gsig):
        out_match, sub_map = Cls.do_outputs_match(tsig[0], gsig[0])
        return out_match and Cls.__do_inputs_match(tsig[1], Cls.apply_substitution(gsig[1], sub_map=sub_map))