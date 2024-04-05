import Naive_qbc_ as nqbc
from processed_dataset import BinaryPreprocess
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram

from PIL import Image

null_class = 0
alt_class = 1

if __name__ == "__main__":
    preproc = BinaryPreprocess((null_class, alt_class))

    #null prior, null probs, positive probs
    naive_qbc = nqbc.create_quantum_circuit(preproc.prior, preproc.null_probs, preproc.positive_probs)

    naive_qbc.draw('mpl')

    #LOAD A 28x28 PIXEL IMAGE
    img = Image.open('/Users/pravinmahendran/Documents/GitHub/QML-QGC/MNIST_57_0.png')
    binarized_features = preproc.inference_features(img)
    print(binarized_features)

    #RUN QUANTUM CIRCUIT WITH BINARIZED FEATURES AS INPUT
    backend = Aer.get_backend('qasm_simulator')
    shots = 1024
    job = transpile(naive_qbc, backend, shots=shots)
    

    result = job.result()
    counts = result.get_counts(naive_qbc)
    print(counts)
