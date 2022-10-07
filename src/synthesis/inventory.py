import sys
import inspect
from numpy.random import random, choice, normal, triangular

sys.path.append('..')
from synthesis.genotype import Genotype, PrimitiveGenotype, CompositeGenotype, GenotypeTemplate, GenotypeBuilder
from priors.primitives_kit import PrimitivesKit

from ARCSolver.hypothesis import Hypothesis
from helpers.task import Task
from helpers.task_vis import *
from priors.primitives import *



class InventoryController:
    def __init__(self, inventory:'Inventory' = None):
        self.inventory = inventory

    def max_size_subnode(self, template:GenotypeTemplate):
        return ( 1 + int((sum(template.shape) / template.depth) * max(0, template.depth - 1))  ) if template and template.depth else 0

    @classmethod
    def shape_of_size(Cls, size):
        # At least one output layer containing one and only one output node
        # The rest shall rougly take a triangular shape with the biggest mode roughly in the middle
        body_size = max(0, size - 1)
        body_len = 1 + int(triangular(0, body_size/2, body_size)) if body_size else 0
        remaining = body_size - body_len
        body = [1] * body_len

        for _ in range(remaining):
            index = min(len(body)-1, int(triangular(0, len(body) / 2, len(body))))
            body[index] += 1

        return tuple([1] + body)

class Inventory:
    def __init__(self, kit=PrimitivesKit, composite_genotypes=None, probability_mass=None, controller=None):
        self.controller = controller or InventoryController(inventory=self)
        self.primitive_genotypes = self.__create_primitives_inventory(kit)
        self.n_primitives = len(self.primitive_genotypes)
        self.genotypes = self.primitive_genotypes | (composite_genotypes or {})
        self.probability_mass = probability_mass or self.uniform_prob_mass()
        self.current_key = len(self.genotypes.keys())
        self.primitives_kit = kit
    

    @staticmethod
    def save(self, filename):
        pass

    @staticmethod
    def load(self, filename):
        pass
    
    def get_n_primitives(self):
        return self.n_primitives

    def get_n_genotypes(self):
        return self.current_key

    def uniform_prob_mass(self):
        return {k:(1/len(self.genotypes)) for k in self.genotypes.keys()}

    def eval_prob_mass(self):
        return self.uniform_prob_mass()

    def __getitem__(self, key):
        return self.genotypes[key] if (key and key in self.genotypes) else None
    
    def __setitem__(self, key, value):
        self.genotypes[key] = value
        self.probability_mass = self.eval_prob_mass()
    
    def __key_of_genotype(self, genotype:Genotype):
        for (k, g) in [(k, g) for (k, g) in self.genotypes.items() if isinstance(g, CompositeGenotype)]:
            if all([
                g.shape == genotype.shape,
                g.depth == genotype.depth,
                g.signature == genotype.signature,
                g.components.keys() == genotype.components.keys(),
                all(g.components[k] == genotype.components[k] for k in g.components.keys()),
            ]): return k
        return None

    def add(self, genotype:Genotype=None):
        if genotype:
            key = self.__key_of_genotype(genotype)
            if key:
                return key
            else:
                k = self.current_key
                self.current_key += 1
                self[k] = genotype
                return k

    def __create_primitives_inventory(self, kit):
        methods = inspect.getmembers(kit, predicate=inspect.isfunction)
        inventory = {}
        for i, method in enumerate(methods): 
            inventory[method[1].__name__] = PrimitiveGenotype(method[1])

        return inventory
    
    def __choose(self, keys=[]):
        s = sum((v for (k, v) in self.probability_mass.items() if k in keys))
        d = {k: v/s for (k, v) in self.probability_mass.items() if k in keys}
        chosen =  choice(list(d.keys()), 1, p=list(d.values()))[0] if len(d) > 0 else None
        return chosen

    def select(self, template:GenotypeTemplate=None):
        keys = [k for k in self.genotypes.keys() if (template and template.does_match(self[k]))]
        chosen = self.__choose(keys)
        return chosen
    
    def generate(self, temperature=0.5, template:GenotypeTemplate=None):
        if random() >= temperature or template.depth == 0:
            return self.select(template=template) or self.compose(temperature=temperature, template=template)
        else:
            return self.compose(temperature=temperature, template=template)

    def compose(self, temperature=0.5, template:GenotypeTemplate=None):
        builder = GenotypeBuilder(template=template, inventory=self)
        for n in range(builder.num_nodes()):
            for node_size in range(self.controller.max_size_subnode(template)):
                node_template = builder.create_template(n, shape=InventoryController.shape_of_size(node_size))
                key = self.generate(temperature=temperature, template=node_template)
                if key is None: 
                    raise AssertionError('A node cannot be generated')
                else: 
                    builder.install(key, n)
                    break

        return self.add(genotype=builder.build_genotype())
