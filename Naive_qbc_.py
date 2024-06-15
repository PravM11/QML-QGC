from qiskit import QuantumCircuit
import numpy as np

# Function to calculate rotation angle
def calc_angle(prob):
    return 2 * np.arccos(np.sqrt(prob))



def create_quantum_circuit(priors, null_probs_list, positive_probs_list):
    num_classes = len(priors)
    num_class_qubits = int(np.ceil(np.log2(num_classes)))
    n = len(null_probs_list[0])
    
    circ = QuantumCircuit(n + num_class_qubits)
    
    for i in range(num_classes):
        binary_representation = format(i, f'0{num_class_qubits}b')
        theta_y = calc_angle(priors[i])
        for j, bit in enumerate(binary_representation):
            if bit == '1':
                circ.x(j)
        circ.ry(theta_y, num_class_qubits - 1)
        for j, bit in enumerate(binary_representation):
            if bit == '1':
                circ.x(j)

    for i in range(num_classes):
        binary_representation = format(i, f'0{num_class_qubits}b')
        null_rotations = [calc_angle(conditional) for conditional in null_probs_list[i]]
        pos_rotations = [calc_angle(conditional) for conditional in positive_probs_list[i]]

        for j, bit in enumerate(binary_representation):
            if bit == '1':
                circ.x(j)
        for k in range(n):
            circ.cry(null_rotations[k], num_class_qubits - 1, num_class_qubits + k)
        circ.x(num_class_qubits - 1)
        for k in range(n):
            circ.cry(pos_rotations[k], num_class_qubits - 1, num_class_qubits + k)
        circ.x(num_class_qubits - 1)
        for j, bit in enumerate(binary_representation):
            if bit == '1':
                circ.x(j)
    
    circ.measure_all()
    circ.draw('mpl')
    return circ
