import csv
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

class MolSmiles:
    def __init__(self, smiles: str, logd74: float, fingerprint_nbits=2048):
        self.smiles = smiles
        self.logd74 = logd74
        self.fingerprint_nbits = fingerprint_nbits

    def set_morgan_fingerprint(self):
        mol = Chem.MolFromSmiles(self.smiles)

        fingerprint_obj = AllChem.GetMorganFingerprintAsBitVect(
            mol, 
            radius=2, 
            nBits=self.fingerprint_nbits
        ) 

        # Convert the fingerprint to a numpy array
        self.fingerprint = np.zeros((self.fingerprint_nbits), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(fingerprint_obj, self.fingerprint)

def get_mols_from_tsv(tsv_path, fingerprint_nbits):
    molecules = []
    with open(tsv_path, mode='r', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')

        # skip header line
        next(tsv_reader)
        
        for molecule in tsv_reader:
            smiles = molecule[1]
            logd74 = molecule[2]
            molecules.append(MolSmiles(smiles, logd74, fingerprint_nbits))

    return molecules

def set_morgan_fingerprints(molecules: list[MolSmiles]):
    for molecule in molecules:
        molecule.set_morgan_fingerprint()

# this script reads in a file 'logd74.tsv', featurizes it, and
# outputs the featurized representation in numpy format 'logd74.npy'
def main():
    fingerprint_nbits = 2048

    molecules = get_mols_from_tsv('data/logd74.tsv', fingerprint_nbits)
    set_morgan_fingerprints(molecules)

    num_molecules = len(molecules)
    
    fingerpints = np.zeros((num_molecules, fingerprint_nbits), dtype=np.int8)
    logd74s = np.zeros((num_molecules), dtype=np.float32)
    for i, mol in enumerate(molecules):
        fingerpints[i] = mol.fingerprint
        logd74s[i] = mol.logd74

    fingprints_path = 'data/fingerprints.npy'
    np.save(fingprints_path, fingerpints)

    logd74s_path = 'data/logd74s.npy'
    np.save(logd74s_path, logd74s)

if __name__=="__main__":
    main()