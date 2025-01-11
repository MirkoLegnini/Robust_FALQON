import pennylane as qml
from pennylane import numpy as np
from pennylane import qaoa as qaoa
from tqdm import trange
from problem import Problem


#####################################################
## Basic FALQON implementation without noise
#####################################################
class FALQON():
    ## Initialization block
    def __init__(self, problem: Problem, max_depth: int, delta_t: float, dev) -> None:
        self.dev=dev
        self.max_depth=max_depth
        self.delta_t=delta_t
        self.cost_h=problem.cost_h
        self.driver_h=problem.driver_h
        self.comm_h=problem.comm_h

    def _falqon_layer(self, beta_k):
        qml.ApproxTimeEvolution(self.cost_h, self.delta_t, 1)
        qml.ApproxTimeEvolution(self.driver_h, self.delta_t * beta_k, 1)
        return
    
    ## Build Ansatz for the circuit (fixed dimension)
    def _build_ansatz(self):
        @qml.simplify
        def ansatz(beta, **kwargs):
            layers = len(beta)
            for w in self.dev.wires:
                qml.Hadamard(wires=w)
            qml.layer(
                self._falqon_layer,
                layers,
                beta
            )

        return ansatz
    
    def _expval_circuit(self, beta):
        ansatz = self._build_ansatz()
        ansatz(beta)
        return qml.expval(self.comm_h), qml.expval(self.cost_h), qml.state()
    


    def falqon(self, beta_1, w, threshold=0): ## MODIFIED to add weight to beta
        cost_fn = qml.QNode(self._expval_circuit, self.dev, interface="autograd") # The ansatz + measurement circuit is executable
        betas = [beta_1] # Records each value of beta_k
        energies = [] # Records the value of the cost function at each step
        states = []
        for ii in trange(self.max_depth):
            # Adds a value of beta to the list and evaluates the cost function
            beta, energy, state= cost_fn(betas)
            betas.append(-w * beta)  # this call measures the expectation of the commuter hamiltonian
            energies.append(energy)
            states.append(state)

            ##Stopping criterion
            if threshold>0 and ii>0 and energies[ii-1]-energies[ii]<threshold:
                break
        return betas, energies, states

    
################################################
## Class to run experiments with independent cce
################################################    
class FALQON_independent_cce(FALQON):
    def __init__(self, problem, max_depth, delta_t, dev, epsilon):
        super().__init__(problem, max_depth, delta_t, dev)
        self.epsilon=epsilon
        return 
    
    def _falqon_layer(self, beta_k):
        error_gain=1+2*(np.random.rand()-0.5)*self.epsilon
        beta_k=beta_k*error_gain
        qml.ApproxTimeEvolution(self.cost_h, self.delta_t*error_gain, 1)
        qml.ApproxTimeEvolution(self.driver_h, self.delta_t * beta_k, 1) # 
        return
    

################################################
## Class to run experiments with systematic cce
################################################
class FALQON_systematic_cce(FALQON):
    def __init__(self, problem, max_depth, delta_t, dev, epsilon):
        super().__init__(problem, max_depth, delta_t, dev)
        self.epsilon=np.load(f"./noise_array_{epsilon}.npy")[:max_depth]
        return 
    
    def _falqon_layer(self, beta_k, e_k):
        beta_k=beta_k*(1+e_k.item())
        qml.ApproxTimeEvolution(self.cost_h, self.delta_t, 1)
        qml.ApproxTimeEvolution(self.driver_h, self.delta_t * beta_k, 1) # 
        return
    
    ## Build Ansatz for the circuit (fixed dimension)
    def _build_ansatz(self):
        @qml.simplify
        def ansatz(beta, **kwargs):
            layers = len(beta)
            for w in self.dev.wires:
                qml.Hadamard(wires=w)
            qml.layer(
                self._falqon_layer,
                layers,
                beta,
                self.epsilon[:layers]
            )
        return ansatz
    
class FALQON_accelerated():
    def __init__(self, problem: Problem, max_depth: int, delta_t: float, dev) -> None:
        self.dev=dev
        self.max_depth=max_depth
        self.delta_t=delta_t
        self.cost_h=problem.cost_h
        self.driver_h=problem.driver_h
        self.comm_h=problem.comm_h

    def _falqon_layer(self, beta_k):
        qml.ApproxTimeEvolution(self.cost_h, self.delta_t, 1)
        qml.ApproxTimeEvolution(self.driver_h, self.delta_t * beta_k, 1)
        return
    


        
    ## Build Ansatz for the circuit (fixed dimension)
    def _build_ansatz(self):
        @qml.simplify
        def ansatz(state, beta, **kwargs):
            layers=1
            qml.StatePrep(state, wires=self.dev.wires, normalize=True)
            qml.layer(
                self._falqon_layer,
                layers,
                beta
            )
        return ansatz
    
    def _expval_circuit(self, state, beta):
        ansatz = self._build_ansatz()
        ansatz(state, beta)
        return qml.expval(self.comm_h), qml.expval(self.cost_h), qml.state()
    


    def falqon(self, beta_1, w, threshold=0): ## MODIFIED to add weight to beta
        cost_fn = qml.QNode(self._expval_circuit, self.dev, interface="autograd") # The ansatz + measurement circuit is executable
        betas = [beta_1] # Records each value of beta_k
        energies = [] # Records the value of the cost function at each step
        states = [np.ones((2**len(self.dev.wires)))]
        
        for ii in trange(self.max_depth):
            # Adds a value of beta to the list and evaluates the cost function
            beta, energy, state= cost_fn(states[-1], [betas[-1]])
            betas.append(-w * beta)  # this call measures the expectation of the commuter hamiltonian
            energies.append(energy)
            states.append(state)

            ##Stopping criterion
            if threshold>0 and ii>0 and energies[ii-1]-energies[ii]<threshold:
                break
        return betas, energies, states

class FALQON_accelerated_systematic_cce(FALQON_accelerated):
    def __init__(self, problem, max_depth, delta_t, dev, epsilon):
        super().__init__(problem, max_depth, delta_t, dev)
        self.epsilon=epsilon
        return 
    
    def _falqon_layer(self, beta_k):
        error_gain=1+2*(np.random.rand()-0.5)*self.epsilon
        beta_k=beta_k*error_gain
        qml.ApproxTimeEvolution(self.cost_h, self.delta_t*error_gain, 1)
        qml.ApproxTimeEvolution(self.driver_h, self.delta_t * beta_k, 1) # 
        return
    

#class FALQON_accelerated_systematic_cce(FALQON_accelerated):
#    def __init__(self, problem, max_depth, delta_t, dev, epsilon):
#        super().__init__(problem, max_depth, delta_t, dev)
#        self.epsilon=np.load(f"./noise_array_{epsilon}.npy")[:max_depth]
#        return 
#    
#    def _falqon_layer(self, beta_k, e_k):
#        beta_k=beta_k*(1+e_k.item())
#        qml.ApproxTimeEvolution(self.cost_h, self.delta_t, 1)
#        qml.ApproxTimeEvolution(self.driver_h, self.delta_t * beta_k, 1) # 
#        return
#    
#    def _build_ansatz(self):
#        @qml.simplify
#        def ansatz(state, beta, epsilon, **kwargs):
#            layers=1
#            qml.StatePrep(state, wires=self.dev.wires, normalize=True)
#            qml.layer(
#                self._falqon_layer,
#                layers,
#                beta,
#                epsilon
#            )
#        return ansatz
#    
#    def _expval_circuit(self, state, beta, epsilon):
#        ansatz = self._build_ansatz()
#        ansatz(state, beta, epsilon)
#        return qml.expval(self.comm_h), qml.expval(self.cost_h), qml.state()
#    
#    def falqon(self, beta_1, w, threshold=0): ## MODIFIED to add weight to beta
#        cost_fn = qml.QNode(self._expval_circuit, self.dev, interface="autograd") # The ansatz + measurement circuit is executable
#        betas = [beta_1] # Records each value of beta_k
#        energies = [] # Records the value of the cost function at each step
#        states = [np.ones((2**len(self.dev.wires)))]
#        
#        for ii in trange(self.max_depth):
#            # Adds a value of beta to the list and evaluates the cost function
#            beta, energy, state= cost_fn(states[-1], [betas[-1]], [self.epsilon[ii]])
#            betas.append(-w * beta)  # this call measures the expectation of the commuter hamiltonian
#            energies.append(energy)
#            states.append(state)
#
#            ##Stopping criterion
#            if threshold>0 and ii>0 and (energies[ii-1]-energies[ii])/w <threshold:
#                break
#        return betas, energies, states
