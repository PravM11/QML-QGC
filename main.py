import Naive_qbc_ as nqbc
from processed_dataset import BinaryPreprocess
from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

import torch
import numpy as np
from PIL import Image

null_class = 0
alt_class = 1

if __name__ == "__main__":
    preproc = BinaryPreprocess([null_class, alt_class])

    #null prior, null probs, positive probs
    naive_qbc = nqbc.create_quantum_circuit(preproc.prior, preproc.null_probs.tolist(), preproc.positive_probs.tolist())    
    img = torch.from_numpy(np.array(Image.open('MNIST_inference.png').convert('L')))
    binarized_features = preproc.inference_features(img).squeeze()
    print(binarized_features)

    pre_circ = QuantumCircuit(len(preproc.null_probs)+1)
    for i in range(len(binarized_features)):
        if binarized_features[i].item() == 1:
            pre_circ.x(i+1)

    composite = pre_circ.compose(naive_qbc)
    # composite.draw('mpl')
    simulator = AerSimulator()
    shots = 1024
    
    composite = transpile(composite, simulator)
    result = simulator.run(composite, shots=shots).result()
    counts = result.get_counts(composite)

    print(counts)
    matched_counts = {}
    for output in counts.keys():
        for i in range(len(binarized_features)):
            n = len(binarized_features)-1 #Output is little-endian
            if float(output[i]) != binarized_features[n-i].item():
                continue
            matched_counts[output[-1]] = counts[output]

    plot_histogram(matched_counts, title='Output Counts')
    plt.show()
