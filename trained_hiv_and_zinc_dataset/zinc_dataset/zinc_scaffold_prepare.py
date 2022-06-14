import pandas as pd
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import FastFindRings
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import json


# copy from DeepChem. MoleculeNet
def count_and_log(message, i, total, log_every_n):
    """Print a message to reflect the progress of processing once a while.

    Parameters
    ----------
    message : str
        Message to print.
    i : int
        Current index.
    total : int
        Total count.
    log_every_n : None or int
        Molecule related computation can take a long time for a large dataset and we want
        to learn the progress of processing. This can be done by printing a message whenever
        a batch of ``log_every_n`` molecules have been processed. If None, no messages will
        be printed.
    """
    if (log_every_n is not None) and ((i + 1) % log_every_n == 0):
        print('{} {:d}/{:d}'.format(message, i + 1, total))


def prepare_mols(dataset, mols, sanitize, log_every_n=1000):
    """Prepare RDKit molecule instances.

    Parameters
    ----------
    dataset
        We assume ``len(dataset)`` gives the size for the dataset, ``dataset[i]``
        gives the ith datapoint and ``dataset.smiles[i]`` gives the SMILES for the
        ith datapoint.
    mols : None or list of rdkit.Chem.rdchem.Mol
        None or pre-computed RDKit molecule instances. If not None, we expect a
        one-on-one correspondence between ``dataset.smiles`` and ``mols``, i.e.
        ``mols[i]`` corresponds to ``dataset.smiles[i]``.
    sanitize : bool
        This argument only comes into effect when ``mols`` is None and decides whether
        sanitization is performed in initializing RDKit molecule instances. See
        https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
    log_every_n : None or int
        Molecule related computation can take a long time for a large dataset and we want
        to learn the progress of processing. This can be done by printing a message whenever
        a batch of ``log_every_n`` molecules have been processed. If None, no messages will
        be printed. Default to 1000.

    Returns
    -------
    mols : list of rdkit.Chem.rdchem.Mol
        RDkit molecule instances where there is a one-on-one correspondence between
        ``dataset.smiles`` and ``mols``, i.e. ``mols[i]`` corresponds to ``dataset.smiles[i]``.
    """
    if mols is not None:
        # Sanity check
        assert len(mols) == len(dataset), \
            'Expect mols to be of the same size as that of the dataset, ' \
            'got {:d} and {:d}'.format(len(mols), len(dataset))
    else:
        if log_every_n is not None:
            print('Start initializing RDKit molecule instances...')
        mols = []
        for i, s in enumerate(dataset.smiles):
            count_and_log('Creating RDKit molecule instance',
                          i, len(dataset.smiles), log_every_n)
            mols.append(Chem.MolFromSmiles(s, sanitize=sanitize))

    return mols


class ScaffoldSplitter(object):
    @staticmethod
    def get_ordered_scaffold_sets(dataset, include_chirality=False, log_every_n=1000):

        molecules = prepare_mols(dataset, mols=None, sanitize=True)

        if log_every_n is not None:
            print('Start computing Bemis-Murcko scaffolds.')
        scaffolds = defaultdict(list)
        for i, mol in enumerate(molecules):
            count_and_log('Computing Bemis-Murcko for compound',
                          i, len(molecules), log_every_n)
            # For mols that have not been sanitized, we need to compute their ring information
            try:
                FastFindRings(mol)
                mol_scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=include_chirality)
                # Group molecules that have the same scaffold
                scaffolds[mol_scaffold].append(i)
            except:
                print('Failed to compute the scaffold for molecule {:d} '
                      'and it will be excluded.'.format(i + 1))

        # Order groups of molecules by first comparing the size of groups
        # and then the index of the first compound in the group.
        scaffold_sets = {
            scaffold: scaffold_set for (scaffold, scaffold_set) in sorted(
                scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        }

        return scaffold_sets





# def get_rings(s):
#     try:
#         mol = Chem.MolFromSmiles(s)
#     except:
#         return False
#     ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
#     # return len(ssr) <= 3
#     return len(ssr) >= 5
#
#
# t = pd.read_csv('./zinc_smiles_total.csv',header=None)
# t.rename(columns={0:'smiles'},inplace=True)
# t['rings'] = t['smiles'].apply(get_rings)
# data = t[t['rings']]
# data.to_csv('./zinc_smiles_less5.csv',index=False)



data = pd.read_csv('./zinc_smiles_less5.csv')
if not os.path.exists('./ScaffoldSet.json'):
    scaffold_set = ScaffoldSplitter.get_ordered_scaffold_sets(data)
    with open('./ScaffoldSet.json','w') as f:
        json.dump(scaffold_set,f)
else:
    with open('./ScaffoldSet.json','r') as f:
        scaffold_set = json.load(f)

for smi in list(scaffold_set.keys())[:10]:
    print(len(scaffold_set[smi]))
print()
smiles = []
labels = []
scaffolds = []
UseScafList = list(scaffold_set.keys())[:10]
for class_, scaffold in enumerate(UseScafList):
    scaffolds.extend([scaffold]*500)
    labels.extend([class_]*500)
    smiles.extend(scaffold_set[scaffold][:500])

assert len(scaffolds) == len(labels) == len(smiles)

smile_pd = data.iloc[smiles].reset_index().drop(columns={'index','rings'})
final = pd.concat([smile_pd,pd.DataFrame(data=scaffolds,columns=['scaffolds']),
                   pd.DataFrame(data=labels,columns=['label'])],axis=1)
final.to_csv('./smi_scaffold_less5_tmp.csv',index=False)




# # get the scaffold smi with less than rings. total num is 5k
# import pandas as pd
# data = pd.read_csv('./dataset/scaffold_embedding/smi_scaffold_less5_10k.csv')
# pd_data = []
# for i in range(10):
#     flag = (data['label'] == i)
#     tmp = data[flag].sample(frac=0.5,random_state=42)
#     pd_data.append(tmp)
#
# final = pd.concat(pd_data,axis=0)
# final.to_csv('./smi_scaffold_less5_5k.csv',index=False)