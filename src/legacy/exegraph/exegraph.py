import networkx as nx
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Any


class Port(ABC):
    def __init__(self, node, index, type):
        self.type = type
        self.node = node
        self.index = index
    
    def __str__(self):
        return f'{self.node}.{"i" if isinstance(self, Inport) else "o"}[{self.index}]'
    
    def __repr__(self):
        return str(self)

class Inport(Port):
    def __init__(self, node, index, type):
        super().__init__(node, index, type)
class Outport(Port):
    def __init__(self, node, index, type):
        super().__init__(node, index, type)

class Exegraph():

    def __init__(self):
        self.graph = nx.MultiDiGraph()
    
    def resetNodes(self):
        for n in self.graph.nodes:
            n.reset()
    
    def disconnectAll(self):
        self.graph.remove_edges_from(list(self.graph.edges))

    def getInputValues(self, n):
        inputs = defaultdict(None)
        for p in self.graph.predecessors(n):
            for ed in self.graph.get_edge_data(p, n).values():
                inputs[ed['to']] = p[ed['from']]
        return [inputs[i] for i in range(max(inputs.keys())+1)]
    
    def getInConnections(self, node):
        connections = set()
        for src, dst in self.graph.in_edges(node):
            for c in self.graph.get_edge_data(src, dst).values():
                connections.add((src.o[c['from']], dst.i[c['to']]))
        return connections
    
    def getOutConnections(self, node):
        connections = set()
        for src, dst in self.graph.out_edges(node):
            for c in self.graph.get_edge_data(src, dst).values():
                connections.add((src.o[c['from']], dst.i[c['to']]))
        return connections

    def getConnectedOutports(self, i: Inport):
        outports = set()
        for src, dst in self.graph.in_edges(i.node):
            for c in self.graph.get_edge_data(src, dst).values():
                if(i == dst.i[c['to']]): 
                    outports.add(src.o[c['from']])
        return outports

    def getConnectedInports(self, o: Outport):
        inports = set()
        for src, dst in self.graph.out_edges(o.node):
            for c in self.graph.get_edge_data(src, dst).values():
                if(o == src.o[c['from']]): 
                    inports.add(dst.i[c['to']])
        return inports
    
    def getConnectedPorts(self, p: Port):
        return self.getConnectedOutports(p) if isinstance(p, Inport) else self.getConnectedInports(p)

    def isConncetd(self, p: Port):
        return len(self.getConnectedPorts(p)) > 0

    def connect(self, i: Inport, o: Outport):
        assert(i.type == o.type or any(t == Any for t in [i.type, o.type])), f'Input type "{i.type}" does not match output type "{o.type}"'
        i.node.setGraph(self)
        o.node.setGraph(self)
        self.graph.add_edge(i.node, o.node, **{'from': i.index, 'to': o.index})

    def disconnectPair(self, o: Outport, i: Inport):
        connections = set()
        ed = self.graph.get_edge_data(o.node, i.node)
        for k in ed:
            if ed[k]['from'] == o.index and ed[k]['to'] == i.index:
                connections.add((o.node, i.node, k))
        for connection in connections:
            self.graph.remove_edge(*connection)

    def disconnectInport(self, i: Inport):
        outports = self.getConnectedOutports(i)
        for o in outports:
            self.disconnectPair(o, i)
    
    def disconnectOutport(self, o: Outport):
        inports = self.getConnectedInports(o)
        for i in inports:
            self.disconnectPair(o, i)

    def disconnect(self, p: Port):
        if isinstance(p, Inport): self.disconnectInport(p) 
        else: self.disconnectOutport(p)

class Node(ABC):
    def __init__(self, intypes: List[type], outypes: List[type], outputs=None, name=None):
        assert(outputs==None or (len(outputs) == len(outypes) and all(type(x[0])==x[1] for x in zip(outputs, outypes))))
        self.name=name
        self.reset(outputs=outputs)
        self.graph = None
        self._inports = [Inport(self, i, t) for i, t in enumerate(intypes)]
        self._outports = [Outport(self, i, t) for i, t in enumerate(outypes)]
    
    def __str__(self):
        return self.name or type(self).__name__
    
    def __repr__(self):
        return str(self)
    
    def _getInports(self):
        return self._inports

    def _getOutports(self):
        return self._outports

    i = property(_getInports)
    o = property(_getOutports)


    @abstractmethod
    def execute(self, inputs):
        pass

    def reset(self, outputs=None):
        self.outputs = outputs
    
    def setGraph(self, graph: Exegraph):
        self.graph = graph
    
    def __getitem__(self, i):
        return self()[i]

    def __call__(self):
        self.outputs = self.outputs if self.outputs else self.execute(self.graph.getInputValues(self))
        return self.outputs

class Constant(Node):
    def __init__(self, types: List[type], consts):
        assert(len(consts) > 0)
        assert(len(types) == len(consts))
        assert(all(type(x[0])==x[1] for x in zip(consts, types)))
        super().__init__(intypes=[], outypes=types, outputs=consts)
    
    def execute(self, _):
        return super().outputs

class Operation(Node):
    def __init__(self, intypes: List[type], outypes: List[type]):
        super().__init__(intypes=intypes, outypes=outypes)

