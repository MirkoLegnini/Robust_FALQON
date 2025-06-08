import pennylane as qml

##################################
## Marker interface
##################################
class Problem():
    def __init__(self) -> None:
        pass
##################################
#  Example problem implementation
##################################    
class MaxCut(Problem):
    def __init__(self, graph) -> None:
        self.cost_h, self.driver_h=qml.qaoa.cost.maxcut(graph)
        self.comm_h=self._build_commuter(graph)
    
    def _build_commuter(self, graph):
        H = qml.Hamiltonian([], [])

        for k in graph.nodes:
            # Adds the terms in the first sum
            for edge in graph.edges:
                i, j = edge
                if k == i:
                    H += qml.PauliY(k) @ qml.PauliZ(j) 
                if k == j:
                    H += qml.PauliZ(i) @ qml.PauliY(k)  
        return H
    

