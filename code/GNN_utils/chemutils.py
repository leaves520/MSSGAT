import time
import numpy as np
import rdkit.Chem as Chem
import tqdm
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.ML.Descriptors import MoleculeDescriptors
try:
    from IPython.display import SVG
    from cairosvg import svg2png,svg2pdf
except:
    pass

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from dgl import unbatch
from dgllife.utils import CanonicalAtomFeaturizer,AttentiveFPAtomFeaturizer
from dgllife.utils import CanonicalBondFeaturizer


# get node_feature from dgllife
def get_dgl_node_feature(mol):
    '''
    """A default featurizer for atoms.

    The atom features include:
    * **One hot encoding of the atom type**. The supported atom types include
      ``C``, ``N``, ``O``, ``S``, ``F``, ``Si``, ``P``, ``Cl``, ``Br``, ``Mg``,
      ``Na``, ``Ca``, ``Fe``, ``As``, ``Al``, ``I``, ``B``, ``V``, ``K``, ``Tl``,
      ``Yb``, ``Sb``, ``Sn``, ``Ag``, ``Pd``, ``Co``, ``Se``, ``Ti``, ``Zn``,
      ``H``, ``Li``, ``Ge``, ``Cu``, ``Au``, ``Ni``, ``Cd``, ``In``, ``Mn``, ``Zr``,
      ``Cr``, ``Pt``, ``Hg``, ``Pb``.
    * **One hot encoding of the atom degree**. The supported possibilities
      include ``0 - 10``.
    * **One hot encoding of the number of implicit Hs on the atom**. The supported
      possibilities include ``0 - 6``.
    * **Formal charge of the atom**.
    * **Number of radical electrons of the atom**.
    * **One hot encoding of the atom hybridization**. The supported possibilities include
      ``SP``, ``SP2``, ``SP3``, ``SP3D``, ``SP3D2``.
    * **Whether the atom is aromatic**.
    * **One hot encoding of the number of total Hs on the atom**. The supported possibilities
      include ``0 - 4``.
    '''

    atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
    feats = atom_featurizer(mol)

    return feats['h']


def get_dgl_bond_feature(mol):
    # 13 size
    bond_featurizer = CanonicalBondFeaturizer(bond_data_field='feat', self_loop=True)
    feats = bond_featurizer(mol)
    return feats['feat']



'''get molecule node features for graph (three function below)'''
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs that not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom,bool_id_feat=False,explicit_H=False,use_chirality=True):
    if bool_id_feat:
        pass
    else:
        results = one_of_k_encoding_unk(
          atom.GetSymbol(),
          [
            'B',
            'C',
            'N',
            'O',
            'F',
            'Si',
            'P',
            'S',
            'Cl',
            'As',
            'Se',
            'Br',
            'Te',
            'I',
            'At',
            'other'
          ]) + one_of_k_encoding(atom.GetDegree(),
                                 [0, 1, 2, 3, 4, 5, 6 ,7, 8 , 9 , 10]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2,'other'
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False
                                     ] + [atom.HasProp('_ChiralityPossible')]

        return np.array(results)


def bond_features(bond,self_loop=False):
    # for bond object type..: Chem.rdchem.Bond
    feature = one_of_k_encoding(bond.GetBondType(),[Chem.rdchem.BondType.SINGLE,Chem.rdchem.BondType.DOUBLE,Chem.rdchem.BondType.TRIPLE,Chem.rdchem.BondType.AROMATIC]) + \
        [bond.GetIsConjugated()] + [bond.IsInRing()] + one_of_k_encoding(bond.GetStereo(),[Chem.rdchem.BondStereo.STEREONONE,Chem.rdchem.BondStereo.STEREOANY,Chem.rdchem.BondStereo.STEREOZ,Chem.rdchem.BondStereo.STEREOE])

    if self_loop:
        feature += [False]  # indicate index: self loops or not
    return np.array(feature).astype(float)



def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def sanitize(mol): 
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol


def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol


def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)  # We assume this is not None
    return new_mol


def get_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 1, nBits=512)
    npfp = np.array(list(fp.ToBitString())).astype('float')
    return npfp


def get_descriptors(smiles):
    mol = get_mol(smiles)
    des_list = ['MolLogP','NumHAcceptors','NumHeteroatoms','NumHDonors','MolWt','NumRotatableBonds',
                'RingCount','Ipc','HallKierAlpha','NumValenceElectrons','NumSaturatedRings','NumAliphaticRings','NumAromaticRings']
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
    result = np.array(calculator.CalcDescriptors(mol))
    result[np.isnan(result)] = 0
    return result.astype("float")


# highlight the crucial atoms-bonds
class highlight_mol(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __call__(self, iter_nums, output_batch, input_batch, RGB=(236 / 256., 173 / 256., 158 / 256.), radius=0.5,
                  size=(400, 200)):

        print('=========highlight crucial cluster============')
        output_batch = unbatch(output_batch)
        output_batch_size = len(output_batch)
        for idx, chem in enumerate(tqdm.tqdm(output_batch)):
            try:
                bonds = set()
                max_eid = chem.edata['e'].argmax()
                u, v = chem.find_edges(max_eid)
                mol = input_batch[idx].mol
                clique = input_batch[idx].nodes_dict[int(u)]['clique']
                clique2 = input_batch[idx].nodes_dict[int(v)]['clique']

                for ix in range(len(clique)):
                    if (ix < len(clique) - 1):
                        bonds.add(mol.GetBondBetweenAtoms(clique[ix], clique[ix + 1]).GetIdx())
                #     elif (ix == len(clique) - 1 and len(clique) > 1):
                #         bonds.add(mol.GetBondBetweenAtoms(clique[0], clique[-1]).GetIdx())
                #
                # for ix in range(len(clique2)):
                #     if (ix < len(clique2) - 1):
                #         bonds.add(mol.GetBondBetweenAtoms(clique2[ix], clique2[ix + 1]).GetIdx())
                #     elif (ix == len(clique2) - 1 and len(clique2) > 1):
                #         bonds.add(mol.GetBondBetweenAtoms(clique2[0], clique2[-1]).GetIdx())
            except:
                pass
            # clique = clique + clique2
            self.highlight_mol(mol, clique, bonds, c=RGB, r=radius, s=size, output_name='mol_' + str(idx + 1) + str(iter_nums))


            # time.sleep(0.01)
            # print('\r{}/{}th to highlight..'.format(idx + 1, output_batch_size), end='')

    def highlight_mol(self, mol, atoms_id=None, bonds_id=None, c=(1, 0, 0), r=0.5, s=(400, 200), output_name='test'):
        atom_hilights = {}
        bond_hilights = {}
        radii = {}

        if atoms_id:
            for atom in atoms_id:
                atom_hilights[int(atom)] = c
                radii[int(atom)] = r

        if bonds_id:
            for bond in bonds_id:
                bond_hilights[int(bond)] = c

        self.generate_image(mol, list(atom_hilights.keys()), list(bond_hilights.keys()),
                       atom_hilights, bond_hilights, radii, s, output_name + '.pdf', False)

    def generate_image(self, mol, highlight_atoms, highlight_bonds, atomColors, bondColors, radii, size, output,
                       isNumber=False):

        if self.verbose:
            print('\thighlight_atoms_id:', highlight_atoms)
            print('\thighlight_bonds_id:', highlight_bonds)
            print('\tatoms_colors:', atomColors)
            print('\tbonds_colors:', bondColors)

        view = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        tm = rdMolDraw2D.PrepareMolForDrawing(mol)

        option = view.drawOptions()
        if isNumber:
            for atom in mol.GetAtoms():
                option.atomLabels[atom.GetIdx()] = atom.GetSymbol() + str(atom.GetIdx() + 1)

        view.DrawMolecule(tm, highlightAtoms=highlight_atoms, highlightBonds=highlight_bonds,
                          highlightAtomColors=atomColors, highlightBondColors=bondColors, highlightAtomRadii=radii)
        view.FinishDrawing()

        svg = view.GetDrawingText()
        SVG(svg.replace('svg:', ''))

        svg2pdf(bytestring=svg, write_to='./mol_picture/' + output)  # svg to png format


def tree_decomp_old(mol):

    MST_MAX_WEIGHT = 100  
    n_atoms = mol.GetNumAtoms() 
    if n_atoms == 1:  
        return [[0]], []

    # step 1 :
    cliques = [] 
    for bond in mol.GetBonds(): 
        a1 = bond.GetBeginAtom().GetIdx() 
        a2 = bond.GetEndAtom().GetIdx()  
        if not bond.IsInRing(): 
            cliques.append([a1, a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)] 
    cliques.extend(ssr)  

    nei_list = [[] for i in range(n_atoms)]  
    for i in range(len(cliques)):  
        for atom in cliques[i]: 
            nei_list[atom].append(i) 


    for i in range(len(cliques)): 
        if len(cliques[i]) <= 2:  
            continue
        for atom in cliques[i]:  
            for j in nei_list[atom]: 
                if i >= j or len(cliques[j]) <= 2: 
                    continue
                inter = set(cliques[i]) & set(cliques[j])  
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []

    cliques = [c for c in cliques if len(c) > 0] 
    nei_list = [[] for i in range(n_atoms)] 
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)


    edges = defaultdict(int)
    for atom in range(n_atoms): 
        if len(nei_list[atom]) <= 1: 
            continue
        cnei = nei_list[atom]  
        bonds = [c for c in cnei if len(cliques[c]) == 2] 
        rings = [c for c in cnei if len(cliques[c]) > 4]  
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2): 
            cliques.append([atom])  
            c2 = len(cliques) - 1
            for c1 in cnei: 
                edges[(c1, c2)] = 1
        elif len(rings) > 2:  
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei: 
                edges[(c1, c2)] = MST_MAX_WEIGHT - 1
        else: 
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1, c2)] < len(inter):
                        edges[(c1, c2)] = len(inter)  

    edges = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()]


    if len(edges) == 0: 
        return cliques, edges

    row, col, data = list(zip(*edges))
    n_clique = len(cliques)
    clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
    junc_tree = minimum_spanning_tree(clique_graph)
    row, col = junc_tree.nonzero()
    edges = [(row[i], col[i]) for i in range(len(row))]

    return (cliques, edges)


def tree_decomp(mol):

    MST_MAX_WEIGHT = 100  
    n_atoms = mol.GetNumAtoms() 
    if n_atoms == 1:  
        return [[0]], []

    cliques = [] 
    for bond in mol.GetBonds(): 
        a1 = bond.GetBeginAtom().GetIdx() 
        a2 = bond.GetEndAtom().GetIdx() 
        if not bond.IsInRing(): 
            cliques.append([a1, a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]  
    cliques.extend(ssr)  

    nei_list = [[] for i in range(n_atoms)]  
    for i in range(len(cliques)):  
        for atom in cliques[i]: 
            nei_list[atom].append(i) 


    edges = defaultdict(int)
    for atom in range(n_atoms):  
        if len(nei_list[atom]) <= 1:
            continue
        cnei = nei_list[atom]  
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]  
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2): 
            cliques.append([atom])  
            c2 = len(cliques) - 1
            for c1 in cnei:  
                edges[(c1, c2)] = 1
        elif len(rings) > 2:  
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei: 
                edges[(c1, c2)] = MST_MAX_WEIGHT - 1
        else: 
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1, c2)] < len(inter):
                        edges[(c1, c2)] = len(inter)  

    edges = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()] 


    if len(edges) == 0: 
        return cliques, edges

    row, col, data = list(zip(*edges))
    n_clique = len(cliques)
    clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
    junc_tree = minimum_spanning_tree(clique_graph)
    row, col = junc_tree.nonzero()
    edges = [(row[i], col[i]) for i in range(len(row))]

    return (cliques, edges)



# for test
if __name__ == '__main__':
    m1 = Chem.MolFromSmiles('C1C2C3(CC1)C(CC2)CCC3') # 三环[6.3.0.01,5]十一烷
    m2 = Chem.MolFromSmiles('C1=CC=CC=C1C(=O)O')   # 苯甲酸
    m3 = Chem.MolFromSmiles('C1CCC(CC1)(C)C') # 1,1-二甲基环己烷
    m4 = Chem.MolFromSmiles('C1CC2CC1C(=O)CC2=O') # 二环[3.2.1]辛烷-2,4-二酮 （含桥）
    m5 = Chem.MolFromSmiles('CCCC(=O)O') # 丁酸
    tree_decomp(m2)
