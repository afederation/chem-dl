import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import pubchempy as pcp

data = pd.read_csv('../data/raw/csv/CID-malaria.csv') #csv file containing compound ID, properties and malaria acivity
cid_list = list(data.cid)

def pull_smiles(cid_list):
    '''
    Use pcp package to pull SMILES string from the pubchem API
    '''
    smiles = [x['CanonicalSMILES'] for
              x in pcp.get_properties('CanonicalSMILES', cid_list)]

    return smiles
    
def sub_smiles(smiles_list, length=120):
    '''
    Replace two-letter atom codes with one-letter codes
    '''
    sub_dict = {
        'Cl' : 'R',
        'Br' : 'M',
        'Ca' : 'A',
        'Be' : 'E',
        'Na' : 'X',
        'Li' : 'L'
    }
    smiles_sub = []
    
    for atom in sub_dict:
        for s in smiles:
            if len(s) > length: continue
            smile = s.replace(atom, sub_dict[atom])
            while len(smile) < length:
                smile += ' '
            smiles_sub.append(smile)

    return smiles_sub

def find_smiles_bank(smiles_list):
    '''
    From a list of SMILES strings
    Return all possible characters in a list
    '''

    bank = []

    for s in smiles_list:
        for char in set(s):
            if char not in bank:
                bank.append(char)
    bank = sorted(bank)
    return bank
    
def smiles_vectorizer(smiles_list):
    vector = [[0 if symbol != char else 1 for symbol in bank]
              for char in s]
    return vector


def smiles_decoder(tensor, bank):
    '''
    Inputs a tensor containing one-hot encoded of SMILES strings
    Returns a SMILES string
    '''
    smiles = ''
    for vector in tensor:
        npv = vector.numpy()
        idx = np.where(npv == 1)[0][0]
        smiles += bank[idx]
    return smiles


def smiles_list_to_tensor(smiles_list):
    smiles_tensor = []
    for s in smiles:
        smiles_tensor.append(smiles_vectorizer(s, bank))
        smiles_tensor = torch.tensor(smiles_tensor, dtype=torch.long) #need long for cross entropy loss
    torch.save(smiles_tensor, 'smiles_tensor.pkl')
