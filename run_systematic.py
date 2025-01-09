from FALQON import FALQON_accelerated_systematic_cce, FALQON_systematic_cce, FALQON_accelerated 
import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import problem
import scipy as sp


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 22


def main():
    ######################################
    # Parameters declaration
    ######################################
    max_depth = 1000
    max_iters=50
    beta_1 = 0.0
    delta_t = 0.05

    ######################################################
    # Load graph
    ######################################################
    n_nodes=8
    min_edge=3
    graph=nx.read_adjlist(f"./graph_regular_{n_nodes}_{min_edge}.graph")

    #####################################################
    # Device initialization
    #####################################################
    dev_nominal= qml.device("default.qubit", wires=graph.nodes, shots=None) # Creates a device for the simulation

    ##########################################
    # Problem initialization
    ##########################################
    maxcut=problem.MaxCut(graph)
    
    #########################################
    # FALQON section
    #########################################
    falqon_ideal=FALQON_accelerated(maxcut, max_depth, delta_t, dev_nominal)
    _, energies_ideal, states_ideal=falqon_ideal.falqon(beta_1, 1.0, 0)
    final_state=states_ideal[-1].numpy()
    final_probabilities=np.array([sp.linalg.norm(x)**2 for x in final_state])
    
    # fig, ax=plt.subplots()
    # plt.xlabel("Solution")
    # plt.ylabel("Probability")
    # ax.grid(which='both')
    # plt.title(f"Probability distribution")
    # ax.bar(range(len(final_probabilities)), final_probabilities)
    # plt.show()
    
    fig, ax=plt.subplots()
    plt.xlabel("Circuit Depth")
    plt.ylabel("Cost Function Value")
    ax.grid(which='both')
    ax.plot(range(len(energies_ideal)+1)[1:], energies_ideal, label="Noiseless evolution", linestyle='-', color='k')
    
    energies_ideal=np.array(energies_ideal)
    errors=np.zeros((3, max_depth))
    linestyles=['--', '-.', ':']
    for ii in range(3):
        np.random.seed(ii)
        epsilon=0.1+0.4*ii
        for _ in range(max_iters):
            falqon_systematic_cce=FALQON_accelerated_systematic_cce(maxcut, max_depth, delta_t, dev_nominal, epsilon)
            _, energies_systematic_cce, _ =falqon_systematic_cce.falqon(beta_1, 1.0, 0)
            energies_systematic_cce=np.array(energies_systematic_cce)
            errors[ii]+=np.absolute((energies_ideal-energies_systematic_cce))
        ax.plot(range(len(energies_ideal)+1)[1:], energies_systematic_cce, label=f"Evolution under systematic CCE, $\\bar\epsilon$={epsilon}", linestyle=linestyles[ii])
    ax.legend()
    plt.show()

    errors=errors/max_iters
    fig, ax=plt.subplots()
    plt.xlabel("Circuit Depth")
    plt.ylabel("Cost Function Error")
    ax.grid(which='both')
    for ii in range(3):
        epsilon=0.1+0.4*ii
        ax.plot(range(len(energies_ideal)+1)[2:], errors[ii,1:], label=f"Systematic error evolution, $\\bar\epsilon$={epsilon}", linestyle=linestyles[ii], linewidth=1.5)
    ax.legend()
    plt.show()
    





if __name__=="__main__":
    main()