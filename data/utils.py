import networkx as nx

def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum())
    for bond in mol.GetBonds():
        # G.add_edge(bond.GetBeginAtomIdx(),
        #            bond.GetEndAtomIdx(),
        #            bond_type=bond.GetBondType())
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx())
    return G
