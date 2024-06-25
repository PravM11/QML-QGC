import Naive_qbc_ as nqbc
from processed_dataset import BinaryPreprocess
from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

import torch
import torchvision
import numpy as np
from PIL import Image

null_class = 0
alt_class = 1

class classicalBayes:
    def __init__(self, priors, probs):
        self.priors = priors
        self.probs = probs
    
    def inference(self, sample):
        scores = self.priors

        for i in range(len(sample)):
            if sample[i]:
                scores *= self.probs
            else:
                scores *= (1 - self.probs)
        
        return scores.argmax()

if __name__ == "__main__":
    preproc = BinaryPreprocess([null_class, alt_class])

    #null prior, null probs, positive probs
    naive_qbc = nqbc.create_quantum_circuit(preproc.prior, preproc.null_probs.tolist(), preproc.positive_probs.tolist())    
    classical_bayes = classicalBayes(preproc.priors, preproc.probs)
    
    val_set = torchvision.datasets.MNIST(root='data', train=False)

    total = 0
    classical_correct = 0
    quantum_correct = 0
    for (img, label) in val_set:
        if label == 0 or label == 1:
            #print(f"Example {total+1}")

            img = torch.from_numpy(np.array(img.convert('L')))
            binarized_features = preproc.inference_features(img).squeeze()
            # print(binarized_features)

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

            #print(counts)
            matched_counts = {}
            for output in counts.keys():
                for i in range(len(binarized_features)):
                    n = len(binarized_features)-1 #Output is little-endian
                    if float(output[i]) != binarized_features[n-i].item():
                        continue
                    matched_counts[output[-1]] = counts[output]

            quantum_pred= 1 if matched_counts['1'] > matched_counts['0'] else 0
            if quantum_pred == label:
                quantum_correct += 1
            if classical_bayes.inference(binarized_features) == label:
                classical_correct += 1
            total += 1

            # plot_histogram(matched_counts, title='Output Counts')
            # plt.show()
    
    print(f"Classical Accuracy: {classical_correct / total}")
    print(f"Quantum Accuracy: {quantum_correct / total}")