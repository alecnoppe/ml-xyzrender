"""Bond inference strategy adapted from @ehoogeboom's implementation https://github.com/ehoogeboom/e3_diffusion_for_molecules/]"""
from typing import Tuple, Type, List

import networkx as nx
import numpy as np

# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
ORDER1_BONDS = {
    'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
        'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
        'Cl': 127, 'Br': 141, 'I': 161},
    'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
        'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
        'I': 214},
    'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
        'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
    'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
        'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
        'I': 194},
    'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
        'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
        'I': 187},
    'B': {'H':  119, 'Cl': 175},
    'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
            'F': 160, 'Cl': 202, 'Br': 215, 'I': 243 },
    'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
            'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
            'Br': 214},
    'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
        'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
        'I': 234},
    'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
            'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
    'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
        'S': 210, 'F': 156, 'N': 177, 'Br': 222},
    'I': {'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
        'S': 234, 'F': 187, 'I': 266},
    'As': {'H': 152}
}


ORDER2_BONDS = {
    'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
    'N': {'C': 129, 'N': 125, 'O': 121},
    'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
    'P': {'O': 150, 'S': 186},
    'S': {'P': 186}
}


ORDER3_BONDS = {
    'C': {'C': 120, 'N': 116, 'O': 113},
    'N': {'C': 116, 'N': 110},
    'O': {'C': 113}
}


# Following the Equivariant Diffusion Model implementation, we introduce slight margins (fudge factors)
# optimized on the stability/validity of the QM9 training dataset.
ORDER1_MARGIN, ORDER2_MARGIN, ORDER3_MARGIN = 10, 5, 3
# ORDER1_MARGIN, ORDER2_MARGIN, ORDER3_MARGIN = 0,  0, 0


ATOMIC_NUM_LOOKUP = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "As": 33,
    "Br": 35,
    "I": 53
}


VALENCE_LOOKUP = {
    "H": 1,
    "B": 3,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "Si": 4,
    "P": 3,   # often 3 or 5; 3 is typical for valence models
    "S": 2,   # often 2, 4, 6; 2 is typical for valence models
    "Cl": 1,
    "As": 3,  # often 3 or 5
    "Br": 1,
    "I": 1
}


def get_bond_order(source_atom_type:str, target_atom_type:str, distance:float) -> int:
    """Heuristic distance-based method for determining whether a bond exists between two atoms.
    
    Uses the distance threshold table above as follows:
    - Check if ('source atom-type', 'target atom-type') pair exists in the threshold table
    - If so, check if the distance between 'source atom' and 'target atom' is less than the threshold
    - If so, check the highest bond-order (max 3) for which the distance is still less than the threshold.
    - Return this bond-order (as integer), or 0 if there is no bond between these atoms.

    Args:
        source_atom_type: Atom type of the first atom
        target_atom_type: Atom type of the second atom
        distance: Distance between these atoms

    Returns:
        Bond-order of the bond between the source, target atoms - or 0 if there is no bond.
    """
    # Shorten var. names for convenience
    src = source_atom_type
    tgt = target_atom_type
    # Scale to match distances described in tables.
    distance = 100 * distance

    # Check if the src,tgt pair exists in each dict, if not set the threshold equal to -1 >:)
    order_1_threshold = ORDER1_BONDS[src][tgt] + ORDER1_MARGIN if ORDER1_BONDS.get(src) and \
        ORDER1_BONDS.get(src).get(tgt) else -1
    order_2_threshold = ORDER2_BONDS[src][tgt] + ORDER2_MARGIN if ORDER2_BONDS.get(src) and \
        ORDER2_BONDS.get(src).get(tgt) else -1
    order_3_threshold = ORDER3_BONDS[src][tgt] + ORDER3_MARGIN if ORDER3_BONDS.get(src) and \
        ORDER3_BONDS.get(src).get(tgt) else -1

    # Check the minimum threshold that the distance falls under, if any.
    if distance < order_3_threshold:
        return 3
    elif distance < order_2_threshold:
        return 2
    elif distance < order_1_threshold:
        return 1
    else:
        return 0
    

XYZ = Tuple[str, Tuple[float, float, float]]
def build_distance_based_graph(xyz_tuples:List[XYZ], **kwargs) -> nx.Graph:
    """Build a networkx.Graph instance based on a list of xyz tuples. Uses the interatomic distance threshold heuristic 
    defined above, as is common in ML papers. When using QM9 in the 'vanilla' xyzrender, many molecules look quite wonky.

    NOTE: for compatibility, this uses the same graph attributes as xyzrender, though many are unused in QM9/GEOM-DRUGS.

    Args:
        xyz_tuples: List of (atom type, (x, y, z)) tuples.

    Returns:
        networkx Graph instance with atoms and bonds.
    """
    # Iterate over all atoms and add them to the networkx Graph
    G = nx.Graph()
    for i, atom in enumerate(xyz_tuples): 
        G.add_node(i,
                   symbol=atom[0],
                   atomic_number=ATOMIC_NUM_LOOKUP[atom[0]], 
                   formal_charge=0, # NOTE: Assuming this is always 0
                   valence=VALENCE_LOOKUP[atom[0]],
                   position=np.float64(atom[1]),
                   metal_valence=0 # NOTE: Assuming this is always 0
                   )
        
    # Iterate over all (undirected) atom pairs and connect them with bonds if the distance between them is less than
    # some threshold specified in the terrible tables above.
    for i, atom in enumerate(xyz_tuples):
        for k, other_atom in enumerate(xyz_tuples[i+1:]):
            j = i + k + 1
            dist = np.linalg.norm(G.nodes[i]["position"] - G.nodes[j]["position"])
            bo = get_bond_order(atom[0], other_atom[0], dist)
            if bo > 0:
                # NOTE: Assumes metal_coord is False
                G.add_edge(i, j, bond_order=bo, distance=dist, metal_coord=False, bond_type=(atom[0], other_atom[0]))
                
    return G