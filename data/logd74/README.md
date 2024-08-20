# Data

## Lipophilicity (logd74)

The script `featurize_logd74.py` reads in the `logd74.tsv` file and calculates the Morgan fingerprints of each molecule.

It outputs two files: `fingerprints.npy`, a numpy array of dimension (1130, 2048) containing the 2048 bit Morgan Fingerprint for each molecule, and `logd74s.npy`, a numpy array of dimension (1130) containing the logd74 value (liphophilicity measure) for each molecule.