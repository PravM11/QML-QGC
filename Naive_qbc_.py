import qiskit
from qiskit import QuantumCircuit
import numpy as np

# Function to calculate rotation angle
def calc_angle(prob):
    #Calculate the angle using the formula
    return 2 * np.arccos(np.sqrt(prob))

def create_quantum_circuit(P_y0, P_x1_given_y0, P_x1_given_y1):
    # Calculate angles based on the provided probabilities
    theta_y = calc_angle(P_y0)
    theta_x1_y0 = calc_angle(P_x1_given_y0)
    theta_x1_y1 = calc_angle(P_x1_given_y1)
    
    # Initialize the quantum circuit with 4 qubits and 4 classical bits
    circ = QuantumCircuit(4, 4)
    
    # Implementing the gates based on the calculated thetas
    circ.ry(theta_y, 0)  # Encode P(y=0) into qubit 0
    circ.cry(theta_x1_y0, 0, 1)  # If y=0, encode P(x1|y=0)
    circ.x(0)  # Flip y to represent y=1
    circ.cry(theta_x1_y1, 0, 1)  # Now for y=1, encode P(x1|y=1)
    circ.x(0)  # Reset y back
    
    # Measure 
    circ.measure([0, 1], [0, 1])  # Measure both the label and feature qubits
    
    return circ

# Example usage
P_y0 = 0.6
P_x1_given_y0 = 0.7
P_x1_given_y1 = 0.2
circ = create_quantum_circuit(P_y0, P_x1_given_y0, P_x1_given_y1)

# To draw the circuit using Matplotlib, ensure you have matplotlib installed and your environment supports plotting
circ.draw('mpl')
