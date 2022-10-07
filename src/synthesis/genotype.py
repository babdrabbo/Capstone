
import sys
import typing
import inspect
from typing import Any, Optional, Callable
from random import randint, shuffle
from collections import defaultdict
from abc import ABC, abstractmethod
from collections.abc import Iterable
from numpy.random import choice, triangular


sys.path.append('..')
# from inventory import Inventory
from synthesis.utils import Utils, TypesChecker



class BoundFunctor:
    def __init__(self, func, kwarg_deps={}, arg_deps=[]):
        self.func = func
        self.arg_deps = arg_deps
        self.kwarg_deps = kwarg_deps
    
    def resolve_kwarg_deps(self):
        return {k: v() for (k, v) in self.kwarg_deps.items()}
    
    def resolve_arg_deps(self):
        return [d() for d in self.arg_deps]

    def __call__(self):
        sig = inspect.signature(self.func)
        ba = sig.bind(*self.resolve_arg_deps(), **self.resolve_kwarg_deps())
        return self.func(*ba.args, **ba.kwargs)


class Genotype(ABC):
    def __init__(self, shape=None, depth=None, signature=None):
        self.shape = shape
        self.depth = depth
        self.signature = signature
    
    @abstractmethod
    def is_primitive(self):
        pass

    @abstractmethod
    def bind(self, inputs, inventory):
        pass

    def out_type(self):
        return self.signature and self.signature[0]

    def in_types(self):
        return self.signature and self.signature[1]
    
class PrimitiveGenotype(Genotype):
    def __init__(self, method):
        self.method = method
        type_hints = typing.get_type_hints(method)
        sig = (type_hints.pop('return', None), tuple(type_hints.values()))
        super().__init__(shape=(1,), depth=0, signature=sig)
    
    def is_primitive(self):
        return True
    
    def bind(self, inputs, _):
        def wrap(x): return lambda:x
        sig = inspect.signature(self.method)
        inputs = inputs if isinstance(inputs, Iterable) else (inputs,)
        binding = {k : v if callable(v) else wrap(v) for (k, v) in zip(sig.parameters.keys(), inputs)}

        return BoundFunctor(self.method, binding)
    
    def __str__(self):
        return f'PrimitiveGenotype:\n'                 + \
               f'- shape:  {self.shape}\n'             + \
               f'- depth:  {self.depth}\n'             + \
               f'- sig:    {self.signature}\n'         + \
               f'- method: {self.method.__name__}'

class FirstOrderGenotype(PrimitiveGenotype):
    def __init__(self, method):
        self.callable = method
        type_hints = typing.get_type_hints(method)
        sig = (type_hints.pop('return', None), tuple(type_hints.values()))
        def make_callable_sig() -> Callable[[*sig[1]], sig[0]] : pass
        super().__init__(method = make_callable_sig)
    
    def bind(self, _, __):
        return BoundFunctor(lambda : self.callable, {})
    
    def __str__(self):
        return f'FirstOrderGenotype:\n'                 + \
               f'- shape:    {self.shape}\n'             + \
               f'- depth:    {self.depth}\n'             + \
               f'- sig:      {self.signature}\n'         + \
               f'- callable: {self.callable.__name__}'

class CompositeGenotype(Genotype):
    def __init__(self, inventory=None, shape=None, signature=None, components=None):
        self.components = components
        self.n_comps = len(components)
        super().__init__(shape=shape, depth= 1 + self.__max_depth_of_components(inventory), signature=signature)
    
    def __max_depth_of_components(self, inventory):
        return max([inventory[c[0]].depth for c in self.components.values()])
    
    def is_primitive(self):
        return False
    
    def bind(self, inputs, inventory:'Inventory'):
        in_types = self.signature[1]
        inputs = inputs if isinstance(inputs, Iterable) else (inputs,)
        if len(in_types) != len(inputs):
            raise AssertionError(f'Number of inputs ({len(inputs)}) does not match the genotype expectation ({len(in_types)})')

        def bind_component(idx):
            def wrap(x): return x if callable(x) else lambda: x

            def make_list(*args):
                return list([item for item in args])

            def bind_connection(c):
                if not c: return wrap(None)
                elif isinstance(c, Iterable): return BoundFunctor(make_list, arg_deps=[bind_connection(sub_c) for sub_c in c])
                elif c >= self.n_comps: return wrap(inputs[c - self.n_comps])
                else: return bind_component(c)

            key, connections = self.components[idx]
            bindable_inputs = tuple(bind_connection(c) for c in connections)
            return inventory[key].bind(bindable_inputs, inventory)

        return bind_component(0) 
    
    def __str__(self):
        def comps_str():
            return '\n' + '\n'.join([f'\t{k}: {v}' for (k, v) in self.components.items()])

        return f'CompositeGenotype:\n'                + \
               f'- shape: {self.shape}\n'             + \
               f'- depth: {self.depth}\n'             + \
               f'- sig:   {self.signature}\n'         + \
               f'- comps: {comps_str()}'

        
class GenotypeTemplate:
    def __init__(self, shape=(1,), depth=1, is_exact:bool=True, signature=None):
        self.shape = shape
        self.depth = depth
        self.is_exact = is_exact
        self.signature = signature

    def does_match(self, genotype: Genotype) -> bool:
        depths_match = (self.depth == genotype.depth) if self.is_exact else (self.depth >= genotype.depth)
        signatures_match = TypesChecker.do_signatures_match(self.signature, genotype.signature)
        return (depths_match and signatures_match)
    
    def __str__(self) -> str:
        return f'Template:\n'               +\
            f'- shape: {self.shape}\n'      +\
            f'- depth: {self.depth}\n'      +\
            f'- exact: {self.is_exact}\n'   +\
            f'- sig:   {self.signature}'


class GenotypeBuilder:

    class __Node:
        def __init__(self, layer):
            self.key = None
            self.depth = None
            self.layer = layer
            self.index = None
            self.out_type = None
            # self.connections = {}
            self.connections = defaultdict(list)
            self.is_connection_a_list = defaultdict(lambda:False)
        
        def set_connection(self, inlet_id, node_or_graph_input_idx, is_a_list=False):
            self.connections[inlet_id].append(node_or_graph_input_idx)
            self.is_connection_a_list[inlet_id] = is_a_list

        def __str__(self):
            single_inlet_conn_str = lambda c : f"Node({c.index})" if isinstance(c, self.__class__) else f"Input({c})"
            inlet_connections_str = lambda cs : (', '.join([single_inlet_conn_str(c) for c in cs])) if cs else "None"
            connections_strings = [f'{i}:({inlet_connections_str(cs)})' for (i, cs) in self.connections.items()]
            return f'@{self.layer}; "{self.key}", {self.out_type} <== [{", ".join(connections_strings)}] '
    
    class __ConnectionOption:
        def __init__(self, node_id, inlet_id, inlet_type, keep=False):
            self.node_id = node_id
            self.inlet_id = inlet_id
            self.inlet_type = inlet_type
            self.keep = keep
        
        def __str__(self):
            return f'{self.node_id}.{self.inlet_id}: {self.inlet_type}'

    def __init__(self, template:GenotypeTemplate=None, inventory=None):
        # TODO: Check that first layer always has one and only one node? i.e. shape = (1, ...)
        self.__shape = template.shape or (1,)
        self.__depth = template.depth or 1
        self.__is_exact = template.is_exact
        self.__signature = template.signature
        self.__inventory = inventory
        self.__nodes = Utils.flatten([GenotypeBuilder.__Node(layer) for _ in range(layer_size)] for layer, layer_size in enumerate(self.__shape))
        for i, node in enumerate(self.__nodes): node.index = i
        self.__nodes[0].out_type = self.__signature[0]
        self.__connection_options = [[] for _ in range(len(self.__shape))]

    def __nodes_of_layer(self, layer):
        return (n for n in self.__nodes if n.layer == layer)
    
    def __blank_nodes_of_layer(self, layer):
        return [n for n in self.__nodes_of_layer(layer) if not n.out_type]

    def __is_last_uninstalled_in_layer(self, node):
        return 0 == len([n for n in self.__nodes_of_layer(node.layer) if (n != node) and (not self.__is_installed(n))])
    
    def __connection_options_of_layer(self, layer):
        return self.__connection_options[layer] if layer in range(len(self.__shape)) else []
    
    def __n_obligations_of_node(self, node):
        n_blank_nodes_next_layer = len(self.__blank_nodes_of_layer(node.layer + 1))
        n_options_next_layer = len(self.__connection_options_of_layer(node.layer + 1))
        list_options_exist = any(map(TypesChecker.is_list, self.__connection_options_of_layer(node.layer + 1)))
        return 0 if list_options_exist or not self.__is_last_uninstalled_in_layer(node) else max(0, (n_blank_nodes_next_layer - n_options_next_layer))

    def __create_signature(self, node:__Node):

        this_layer = node.layer
        next_layer = this_layer + 1
        prev_layer = this_layer - 1

        def re_evaluate_layer_options(layer):
            n_uninstalled = len([n for n in self.__nodes_of_layer(layer) if not self.__is_installed(n)])
            n_all_options = len(self.__connection_options_of_layer(layer))
            list_options = [o for o in self.__connection_options_of_layer(layer) if TypesChecker.is_list(o.inlet_type)]

            if not list_options and (n_uninstalled > n_all_options): 
                raise AssertionError('Options cannot be re-evaluated')

            n_min_list_options_to_replace = 1 if n_uninstalled > n_all_options else 0

            n_list_options_to_replace = randint(n_min_list_options_to_replace, len(list_options))

            list_options_to_replace = choice(list_options, n_list_options_to_replace) if list_options else []
            for list_option in list_options_to_replace:
                type_of_list = TypesChecker.type_of_list(list_option.inlet_type)
                self.__connection_options[layer].remove(list_option)
                self.__connection_options[layer].append(self.__class__.__ConnectionOption(
                    node_id = list_option.node_id, inlet_id = list_option.inlet_id, inlet_type = type_of_list, keep = True) )

        # Output type:
        if not node.out_type:
            re_evaluate_layer_options(this_layer)
            connection_options_of_this_layer = self.__connection_options_of_layer(this_layer)
            out_option = choice(connection_options_of_this_layer) if connection_options_of_this_layer else None
            if out_option is not None:
                node.out_type = out_option.inlet_type
                self.__nodes[out_option.node_id].set_connection(out_option.inlet_id, node, is_a_list=out_option.keep)
                if not out_option.keep:
                    self.__connection_options[this_layer].remove(out_option)
            else:
                raise AssertionError(f"Node {node.index}@{node.layer} is dangling; i.e. its outlet is not connected to any previous layers' inlets!")

        # Input types:
        n_blank_nodes_next_layer = len(self.__blank_nodes_of_layer(next_layer))
        n_options_next_layer = len(self.__connection_options_of_layer(next_layer))
        n_obligations = self.__n_obligations_of_node(node)
        graph_input_types = list(self.__signature[1])
        out_types_set_next_layer = [n.out_type for n in self.__nodes_of_layer(next_layer) if n.out_type]
        optional_input_types = graph_input_types + out_types_set_next_layer
        optional_input_types += ([Any for _ in range(n_blank_nodes_next_layer)]) if not n_obligations else []
        if next_layer in range(len(self.__shape)):
            # graph inputs are optional except for last layer
            optional_input_types = [Optional[t] for t in optional_input_types]
        mandatory_input_types = ([Any for _ in range(n_blank_nodes_next_layer)]) if n_obligations else []

        return (node.out_type or Any, tuple(mandatory_input_types + optional_input_types))


    def __is_valid(self, node: __Node):
        gynotype = self.__inventory[node.key]

        if not gynotype: return False
        if not node.out_type: return False
        if node.depth == None or node.index == None : return False
        if not TypesChecker.do_outputs_match(node.out_type, gynotype.out_type()): return False

        gints = gynotype.in_types()
        mandatory_gints = [i for i, t in enumerate(gints) if not TypesChecker.is_optional(t)]
        if(len(node.connections) < len(mandatory_gints)): return False

        if not all((i in node.connections) for i in mandatory_gints): return False

        graph_input_types = self.__signature[1]
        for i, cs in node.connections.items():
            inlet_type = gints[i]
            outlet_types = [c.out_type if isinstance(c, self.__class__.__Node) else graph_input_types[c] for c in cs]
            if len(set(outlet_types)) > 1: return False # single inlet connected to different types!
            if not TypesChecker.do_outputs_match(inlet_type, outlet_types[0]): return False
            if any(isinstance(c, self.__class__.__Node) and c.layer <= node.layer for c in cs): return False

        return True

    def __is_graph_valid(self):

        def connections_of_layers(layers):
            return [c for l in layers for n in self.__nodes_of_layer(l) for cl in n.connections.values() for c in cl]

        all_nodes_valid = all(map(self.__is_valid, self.__nodes))
        all_connections = connections_of_layers(range(len(self.__shape)))
        at_least_one_connection_to_graph_input = len([c for c in all_connections if not isinstance(c, self.__class__.__Node)]) > 0
        at_least_one_connection_from_each_layer_to_a_next = \
        all([any(map(lambda c: isinstance(c, self.__class__.__Node), connections_of_layers([l]))) for l in range(len(self.__shape) - 1)])

        return all([
            all_nodes_valid, 
            at_least_one_connection_to_graph_input,
            at_least_one_connection_from_each_layer_to_a_next])
    
    def __is_installed(self, node):
        return node.key != None

    def num_nodes(self):
        return len(self.__nodes)
    
    def create_template(self, node_index, shape=(1,)) -> GenotypeTemplate:

        def max_depth(nodes):
            return max(map(lambda n: n.depth or 0, nodes))

        node = self.__nodes[node_index]
        others = [other for other in self.__nodes if other != node]

        node_need_to_be_of_exact_depth = all([
            self.__is_exact,
            all(self.__is_installed(other) for other in others),
            (self.__depth - 1) > max_depth(others)
        ])

        return GenotypeTemplate(
            depth       = self.__depth - 1,
            shape       = shape,
            is_exact    = node_need_to_be_of_exact_depth,
            signature   = self.__create_signature(node)
        )

    def install(self, key=None, node_index=None):

        node = self.__nodes[node_index] if node_index in range(len(self.__nodes)) else None
        if not node: raise AssertionError('Node index out of range')
        genotype = self.__inventory[key]
        if not key: raise AssertionError('Could not find a matching genotype for generated template')
        if not genotype: raise AssertionError('Bad key')
        
        is_list = TypesChecker.is_list
        is_optional = TypesChecker.is_optional
        type_of_list = TypesChecker.type_of_list
        remove_optional = TypesChecker.remove_optional

        node.key = key
        node.depth = genotype.depth
        # !!! DO NOT SET out_type DIRECTLY FROM genotype; THIS WILL MESS UP GENERIC TYPES
        # node.out_type = genotype.out_type()

        this_layer = node.layer
        next_layer = this_layer + 1

        def is_in_out_compatible(in_t, out_t):
            are_optionality_compatible = (not is_optional(out_t)) or all(is_optional(t) for t in [in_t, out_t])
            are_types_compatible = remove_optional(in_t) == remove_optional(out_t) or \
                                   (is_list(in_t) and (remove_optional(type_of_list(in_t)) == remove_optional(out_t)) )
            return are_optionality_compatible and are_types_compatible

        def has_compatibility(in_t, layer):
            return any(is_in_out_compatible(in_t, out_t) for out_t in availabe_out_types_for_layer(layer))
        
        def availabe_out_types_for_layer(layer):
            graph_input_types = list(self.__signature[1])
            available_out_types_next_layer = [n.out_type for n in self.__nodes_of_layer(layer+1) if n.out_type]
            return available_out_types_next_layer + graph_input_types
        
        def have_to_split_a_list(node, gints):
            unconnected_mandatory_inlets = [(i, t) for (i, t) in gints if not is_optional(t) and not i in node.connections]
            unconnected_mandatory_list_inlets = [(i, t) for (i, t) in unconnected_mandatory_inlets if is_list(t)]
            blank_nodes_next_layer = [n for n in self.__nodes_of_layer(next_layer) if not n.out_type]

            return ( len(blank_nodes_next_layer) < len(unconnected_mandatory_inlets) ) and ( len(unconnected_mandatory_list_inlets) < 2 )


        match, subst_map = TypesChecker.are_generically_compatible(node.out_type, genotype.out_type())
        if not match:
            raise AssertionError('Something Wrong Happned: genotype being installed has incompatible out_type with the node!')

        gints = [(i, t) for i, t in enumerate(TypesChecker.apply_substitution(genotype.in_types(), subst_map))]
        shuffle(gints)
        
        # Add all optional inputs as options to the next layer
        for (inlet_id, inlet_type) in [(i, t) for (i, t) in gints if is_optional(t)]:
            if next_layer in range(len(self.__shape)):
                self.__connection_options[next_layer].append(
                self.__class__.__ConnectionOption(node_id=node_index, inlet_id=inlet_id, inlet_type=inlet_type))

        # Mandatory inlets that have no compatible options
        for (inlet_id, inlet_type) in [(i, t) for (i, t) in gints if not is_optional(t) and not has_compatibility(t, this_layer)]:

            blank_nodes_next_layer = [n for n in self.__nodes_of_layer(next_layer) if not n.out_type]
            blank_node = choice(blank_nodes_next_layer) if blank_nodes_next_layer else None
            if blank_node is not None:
                # blank_nodes_next_layer.remove(blank_node)
                if is_list(inlet_type) and (have_to_split_a_list(node, gints)):
                    node.set_connection(inlet_id, blank_node, is_a_list=True)
                    obligation.out_type = type_of_list(inlet_type)
                    self.__connection_options[next_layer].append(
                    self.__class__.__ConnectionOption(node_id=node_index, inlet_id=inlet_id, inlet_type=type_of_list(inlet_type), keep=True))
                else:
                    node.set_connection(inlet_id, blank_node, is_a_list=False)
                    blank_node.out_type = inlet_type
            else:
                raise AssertionError('A manadatory input has neither compatibilities nor blank nodes to connect to')
        
        
        # Remaining unconnected mandatory inlets
        for (inlet_id, inlet_type) in [(i, t) for (i, t) in gints if not is_optional(t) and not i in node.connections]:

            if self.__n_obligations_of_node(node):
                # need to connect obligations first
                blank_nodes_next_layer = self.__blank_nodes_of_layer(next_layer)
                obligation = choice(blank_nodes_next_layer) if blank_nodes_next_layer else None
                if obligation is not None:
                    if is_list(inlet_type) and (have_to_split_a_list(node, gints)):
                        node.set_connection(inlet_id, obligation, is_a_list=True)
                        obligation.out_type = type_of_list(inlet_type)
                        self.__connection_options[next_layer].append(
                        self.__class__.__ConnectionOption(node_id=node_index, inlet_id=inlet_id, inlet_type=type_of_list(inlet_type), keep=True))
                    else:
                        node.set_connection(inlet_id, obligation, is_a_list=False)
                        obligation.out_type = inlet_type
            else:
                # No obligations, match with any compatible input or node
                all_compatible_connections = \
                    [n for n in self.__nodes_of_layer(next_layer) if is_in_out_compatible(inlet_type, n.out_type) or not n.out_type] + \
                    [i for i, t in enumerate(self.__signature[1]) if is_in_out_compatible(inlet_type, t)]
                
                chosen_connection = choice(all_compatible_connections) if all_compatible_connections else None
                if chosen_connection is not None:
                    if isinstance(chosen_connection, self.__Node):
                        if not chosen_connection.out_type: 
                            chosen_connection.out_type = inlet_type
                            node.set_connection(inlet_id, chosen_connection, is_a_list=False)
                        elif is_list(inlet_type) and ( remove_optional(type_of_list(inlet_type)) == remove_optional(chosen_connection.out_type) ):
                            node.set_connection(inlet_id, chosen_connection, is_a_list=True)
                            if next_layer in range(len(self.__shape)):
                                self.__connection_options[next_layer].append(
                                self.__class__.__ConnectionOption(node_id=node_index, inlet_id=inlet_id, inlet_type=type_of_list(inlet_type), keep=True))
                    else:
                        node.set_connection(inlet_id, chosen_connection, is_a_list=False)
        
        # Only for last layer
        if next_layer >= len(self.__shape):
            # connect a random choice of optional inputs
            for (inlet_id, inlet_type) in [(i, t) for (i, t) in gints if is_optional(t)]:
                compatible_graph_inputs = [i for i, t in enumerate(self.__signature[1]) if is_in_out_compatible(inlet_type, t)]
                chosen_input =  choice(compatible_graph_inputs) if compatible_graph_inputs else None
                if chosen_input is not None:
                    node.set_connection(inlet_id, chosen_input)


        

    def build_genotype(self) -> Genotype:

        def compile_node_connections(node):
            def graph_input_index(c):
                return c + sum(self.__shape)

            def compile_inlet_connections(connections, is_a_list):
                group = [c.index if isinstance(c, self.__class__.__Node) else graph_input_index(c) for c in connections]
                return tuple(group) if is_a_list else group[0] if group else None

            genotype = self.__inventory[node.key]
            inlets_types = genotype.in_types() if genotype else []
            inlets_connections_lists = [(node.connections[inlet_index], node.is_connection_a_list[inlet_index]) for inlet_index, _ in enumerate(inlets_types)]

            return tuple([compile_inlet_connections(ics, is_a_list) for (ics, is_a_list) in inlets_connections_lists])


        if self.__is_graph_valid():
            return CompositeGenotype(
                inventory   = self.__inventory,
                shape       = self.__shape, 
                signature   = self.__signature,
                components  = { n: (node.key, compile_node_connections(node)) for n, node in enumerate(self.__nodes) }
            )    



class GenotypeSolver():
    def __init__(self, genotype, inventory):
        self.genotype = genotype
        self.inventory = inventory
    
    def __call__(self, input):
        return self.genotype.bind((input,), inventory=self.inventory)()
