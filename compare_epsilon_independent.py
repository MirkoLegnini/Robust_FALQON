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
    lambd= 1.0 
    beta_1 = 0.0
    delta_t = 0.05
    epsilon=np.linspace(0.02,1,20)
    w=1/(2*lambd)

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
    
    robust_final_costs=[]
    standard_final_costs=[]
    # Noiseless execution
    _,  robust_cost_ideal, _= falqon_ideal.falqon(beta_1, w)
    _,  standard_cost_ideal, _= falqon_ideal.falqon(beta_1, 1)
    
    robust_final_cost_ideal= robust_cost_ideal[-1]
    standard_final_cost_ideal= standard_cost_ideal[-1]
    
    for e,ii in zip(epsilon, range(len(epsilon))):
        np.random.seed(0)
        print(f"Iteration {ii+1}, noise_level={e}")
        falqon_independent_cce=FALQON_independent_cce(maxcut, max_depth, delta_t, dev_nominal, e)
        print("\tRobust execution")
        _,  robust_cost, _ =falqon_independent_cce.falqon(beta_1, w, 0)
        robust_final_costs.append( robust_cost[-1])
        print("\tStandard execution")
        _,  standard_cost, _ =falqon_independent_cce.falqon(beta_1, 1, 0)
        standard_final_costs.append( standard_cost[-1])


    ##########################################
    # Plot results
    ##########################################



    fig, ax=plt.subplots()
    ax.semilogy(epsilon, np.abs(np.array(robust_final_costs)-robust_final_cost_ideal), label="Robust FALQON, $\lambda=1$", linewidth=3)
    ax.semilogy(epsilon, np.abs(np.array(standard_final_costs)-standard_final_cost_ideal), label="FALQON", linestyle='-.', linewidth=3)
    plt.xlabel("$\\bar\epsilon$")
    plt.xlim(0, 1.05)
    plt.ylabel("Final cost")
    ax.grid(which='both')
    plt.title(f"Error on final cost for varying $\\bar\epsilon$")
    ax.legend()
    plt.show()



if __name__=="__main__":
    main()