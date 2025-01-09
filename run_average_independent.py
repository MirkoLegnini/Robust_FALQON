from FALQON import FALQON_independent_cce, FALQON_accelerated 
import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import problem


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 22


def main():
    ######################################
    # Parameters declaration
    ######################################
    max_depth = 200
    max_iters=50
    lambd= 1.0
    beta_1 = 0.0
    delta_t = 0.05
    epsilon=0.5
    w=1/(2*lambd)
    np.random.seed(0)

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
    # Robust FALQON section
    #########################################
    # Noiseless execution
    falqon_ideal=FALQON_accelerated(maxcut, max_depth, delta_t, dev_nominal)
    _, energies_ideal, _=falqon_ideal.falqon(beta_1, w, 0)
    energies_ideal=np.array(energies_ideal)
    sqrd_errors=np.zeros((max_iters, max_depth))
    # Noisy execution
    for ii in range(max_iters):
        print(f"Iter {ii}")
        falqon_independent_cce=FALQON_independent_cce(maxcut, max_depth, delta_t, dev_nominal, epsilon)
        _, energies_independent_cce, _ =falqon_independent_cce.falqon(beta_1, w, 0)

        sqrd_errors[ii]=(energies_ideal-np.array(energies_independent_cce))**2
    
    # Calc standard deviation for each layer
    standard_deviation=np.sqrt(np.sum(sqrd_errors, axis=0)/max_iters)

    ##########################################
    # Plot results
    ##########################################
    upper_line=(energies_ideal-standard_deviation) 
    upper_line=np.clip(upper_line, -10,0)
    fig, ax=plt.subplots()
    plt.xlabel("Circuit Depth")
    plt.ylabel("Cost Function Value")
    ax.grid(which='both')
    plt.title(f"Independent error, $\\bar\epsilon$={epsilon}")
    ax.plot(range(len(energies_ideal)+1)[1:], np.array(energies_ideal) , label="Robust FALQON, $\lambda$=1.0", linestyle='-', linewidth=3.0)
    ax.fill_between(range(len(energies_independent_cce)+1)[1:], upper_line, (energies_ideal+standard_deviation) , alpha=0.4)

    ########################################
    # FALQON
    ########################################
    lambd=0.5
    w=1/(2*lambd)
    #reset random seed
    np.random.seed(0)
    # Noiseless execution
    falqon_ideal=FALQON_accelerated(maxcut, max_depth, delta_t, dev_nominal)
    _, energies_ideal, _=falqon_ideal.falqon(beta_1, w, 0)
    energies_ideal=np.array(energies_ideal)
    sqrd_errors=np.zeros((max_iters, max_depth))
    
    for ii in range(max_iters):
        print(f"Iter {ii}")
        # Noisy execution
        falqon_independent_cce=FALQON_independent_cce(maxcut, max_depth, delta_t, dev_nominal, epsilon)
        _, energies_independent_cce, _ =falqon_independent_cce.falqon(beta_1, w, 0)

        sqrd_errors[ii]=(energies_ideal-np.array(energies_independent_cce))**2
    
    standard_deviation=np.sqrt(np.sum(sqrd_errors, axis=0)/max_iters)
    upper_line=(energies_ideal-standard_deviation) 
    upper_line=np.clip(upper_line, -10,0)
    lower_line=(energies_ideal+standard_deviation) 
    ax.plot(range(len(energies_ideal)+1)[1:], np.array(energies_ideal) , label="FALQON", linestyle='-.', linewidth=3.0)
    ax.fill_between(range(len(energies_independent_cce)+1)[1:], upper_line, lower_line, alpha=0.2)
    ax.legend()
    plt.show()




if __name__=="__main__":
    main()