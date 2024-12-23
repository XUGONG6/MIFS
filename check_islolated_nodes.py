from rdkit import Chem
from rdkit.Chem import AllChem
def check_isolated_nodes(smile):
    mol = Chem.MolFromSmiles(smile)
    print(mol)
    atom_list = [atom.GetIdx() for atom in mol.GetAtoms()]
    bond_list = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    isolated_nodes = set(atom_list) - set(sum(bond_list, ()))

    if isolated_nodes:
        return 1
    else:
        return 0
