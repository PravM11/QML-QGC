import Naive_qbc_
from processed_dataset import BinaryPreprocess

null_class = 0
alt_class = 1

if __name__ == "__main__":
    preproc = BinaryPreprocess((null_class, alt_class))

    #null prior, null probs, positive probs
    naive_qbc = Naive_qbc_.create_quantum_circuit(preproc.prior, preproc.null_probs, preproc.positive_probs)

    naive_qbc.draw('mpl')

    #LOAD A 28x28 PIXEL IMAGE
    img = Image.open('path/to/image.png')
    binarized_features = preproc.inference_features(img)

    #RUN QUANTUM CIRCUIT WITH BINARIZED FEATURES AS INPUT
