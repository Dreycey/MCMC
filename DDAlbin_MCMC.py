import numpy as np
import math
import random
import matplotlib.pyplot as plt


def binary_entropy(p):
    """
    This method calculates the binary entropy.
    INPUT = state vector
    """
    w = calculate_weight(p)
    if w == 0:
        entropy = 0
    elif w == 1:
        entropy = 0
    else:
        entropy = w * math.log2(1 / w) + (1 - w) * math.log2(1 / (1-w))
    return round(entropy, 4)

def calculate_weight(in_vec):
    """
    This method calculates the weight for a given input state vector
    """
    weight = sum(in_vec) / len(in_vec)
    return round(weight, 4)

def choose_update_vec(curr_state):
    """
    choose either a neighbor or self based on probabilities
    """
    # find the neighbors
    neighbors_array = []
    #print("current state vector")
    #print(curr_state)
    for index in range(len(curr_state)):
        curr_val = curr_state[index]
        next_vec = curr_state.copy()
        if curr_val == 1:
            next_vec[index] = 0
        elif curr_val == 0:
            next_vec[index] = 1
        neighbors_array.append(next_vec)
    # choose a random neighbor (this is okay because all have same degree!)
    next_vec_chosen = random.choice(neighbors_array)
    return next_vec_chosen

def mcmc_hypercube(n, T):
    """
    This method impliments the MCMC method on a particle.
    """
    # save weights
    weight_vec = []
    value_vec = []
    # guess initial state vector
    curr_state = np.zeros((n)) #np.random.randint(0, high=2, size=n)
    # run through the algorithm
    for iteration in range(T):
        # choose an update
        update_vec = choose_update_vec(curr_state)
        val_curr = binary_entropy(curr_state)
        val_update = binary_entropy(update_vec)
        if val_update > val_curr:
            curr_state = update_vec.copy()
        elif val_curr == 0:
            continue
        else:
            prob = val_update / val_curr
            nextstate_index = np.random.choice([0,1], 1, p=[prob,(1-prob)])
            if nextstate_index == 0:
                curr_state = update_vec.copy()
            else:
                curr_state = curr_state.copy()
        # save weights
        weight_vec.append(calculate_weight(curr_state))
        value_vec.append(val_curr)
    # plot hist
    plt.hist(weight_vec, edgecolor="black", color="orange")
    plt.title("MCMC with {n} particles and {T} iterations")
    plt.ylabel("Counts")
    plt.xlabel("Value")
    plt.show()
    # plot value per iteration
    plt.scatter(range(len(value_vec)), value_vec, edgecolor="black", color="orange")
    plt.ylabel("Mapping Functions")
    plt.xlabel("Iterations")
    plt.title("MCMC with {n} particles and {T} iterations")
    plt.show()

# Main function for running the script.
def main():
    mcmc_hypercube(100, 1000)

if __name__ == "__main__":
    main()
