# import pandas as pd
# import os
#
# files = os.listdir('./dataset/molnet')
# all = set()
# for file in ('./dataset/molnet/{}'.format(f) for f in files):
#     d = pd.read_csv(file)
#     try:
#         t = d['Smiles'].to_list()
#     except:
#         t = d['smiles'].to_list()
#     for ss in t:
#         all.add(ss)
#
# print(len(all))
# all = list(all)
# data = pd.DataFrame(data=all,columns=['smiles'])
# data = data.dropna(axis=0)
# print(data.head())
# data.to_csv('./dataset/tree_prepare.csv',index=False)

from GNN_utils import mol_tree
from GNN_utils import chemutils
import pandas as pd
import os
files = os.listdir('./dataset/chembl_raw')
all = set()
for file in ('./dataset/chembl_raw/{}'.format(f) for f in files):
    if 'cancer' in file:
        continue
    d = pd.read_csv(file)
    try:
        t = d['Smiles'].to_list()
    except:
        t = d['smiles'].to_list()
    for ss in t:
        all.add(ss)

print(len(all))
all = list(all)
data = pd.DataFrame(data=all,columns=['smiles'])
data = data.dropna(axis=0)
print(data.head())


# data = pd.read_csv('./dataset/chembl_raw/cancer_all.csv')
# data = data.dropna()
try:
    data.rename(inplace=True,columns={'smiles':'Smiles'})
except:
    pass

def get_valid(x):
    return chemutils.get_mol(x) is not None
data['valid'] = data['Smiles'].apply(func=get_valid)
print(len(data))
data = data[data['valid']==True]
print(len(data))
mol_tree.get_Vocab_df(data_name='chembl',df=data)


