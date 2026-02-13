import json
import numpy as np
import fastjet as fj
from tqdm import tqdm
import torch
from torch_geometric.data import Data


def delta_R(p1, p2):
    
    dphi = np.arctan2(
        np.sin(p1.phi() - p2.phi()),
        np.cos(p1.phi() - p2.phi())
    )

    deta = p1.eta() - p2.eta()

    return np.sqrt(deta**2 + dphi**2)


def build_jet(particles, R=10):
    fj_particles = [
        fj.PseudoJet(p["px"], p["py"], p["pz"], p["E"])
        for p in particles
    ]

    jet_def = fj.JetDefinition(fj.cambridge_algorithm, R)
    jets = fj.sorted_by_pt(jet_def(fj_particles))

    if len(jets) == 0:
        return None

    return jets[0]  # leading jet




def build_full_lund_tree(jet):

    children = {}
    node_type = {}
    lund_features = {}
    hyperedges = []

    node_id = 0

    def recurse(j):
        nonlocal node_id

        j1, j2 = fj.PseudoJet(), fj.PseudoJet()
        
        # particle (leaf)
        if not j.has_parents(j1, j2):
            pid = node_id
            node_type[pid] = "particle"
            node_id += 1
            return pid

        # declustering node
        my_id = node_id
        node_type[my_id] = "declustering"
        node_id += 1

        # order by pT
        if j2.pt() > j1.pt():
            j1, j2 = j2, j1

        pt1, pt2 = j1.pt(), j2.pt()
        z = pt2 / (pt1 + pt2)
        dR = delta_R(j1, j2)
        kt = z * (pt1 + pt2) * dR
        lnm = np.float32(0.5 * np.log(abs((j1 + j2).m2())))
        
        try:
            psi = np.float32(np.arctan2((j1.rap() - j2.rap()) , (j1.phi() - j2.phi())))
        except ZeroDivisionError:
            psi = 0

        lund_features[my_id] = [
            np.log(z), np.log(1.0 / dR), np.log(kt), lnm, psi
        ]

        left = recurse(j1)
        right = recurse(j2)

        children[my_id] = (left, right)
        hyperedges.append((my_id, left, right))

        return my_id

    root = recurse(jet)
    return root, children, node_type, lund_features, hyperedges


def build_pyg_hypergraph(root, children, node_type, lund_features, hyperedges, label):

    num_nodes = max(
        max(p, l, r) for (p, l, r) in hyperedges
    ) + 1

    x = torch.zeros((num_nodes, 5), dtype=torch.float)

    for nid, feats in lund_features.items():
        x[nid] = torch.tensor(feats, dtype=torch.float)

    incidence = []
    for h_id, (p, l, r) in enumerate(hyperedges):
        incidence.append([p, h_id])
        incidence.append([l, h_id])
        incidence.append([r, h_id])

    hyperedge_index = torch.tensor(
        incidence, dtype=torch.long
    ).t().contiguous()

    y = torch.tensor(label, dtype=torch.long)

    return Data(
        x=x,
        hyperedge_index=hyperedge_index,
        y=y
    )


def jet_json_to_hypergraph(particles, label, R=0.8):
    jet = build_jet(particles, R)

    root, children, node_type, lund_features, hyperedges = \
        build_full_lund_tree(jet)

    return build_pyg_hypergraph(
        root,
        children,
        node_type,
        lund_features,
        hyperedges,
        label
    )

def load_json_file(filename, label):
    graphs = []

    with open(filename, "r") as f:
        for line in tqdm(f, desc=f"Loading {filename}"):
            particles = json.loads(line)
            data = jet_json_to_hypergraph(particles, label)
            graphs.append(data)

    return graphs

def get_descendants(node, children):

    daughters = []
    granddaughters = []

    if node not in children:
        return daughters, granddaughters

    d1, d2 = children[node]
    daughters.extend([d1, d2])

    for d in (d1, d2):
        if d in children:
            gd1, gd2 = children[d]
            granddaughters.extend([gd1, gd2])

    return daughters, granddaughters

def build_extended_hyperedges(children, node_type):
    hyperedges = []

    for node, ntype in node_type.items():

        if ntype != "declustering":
            continue

        daughters, granddaughters = get_descendants(node, children)

        members = [node] + daughters + granddaughters

        members = list(set(members))

        hyperedges.append(members)

    return hyperedges

def build_hyperedge_index_extended(hyperedges):
    incidence = []

    for h_id, nodes in enumerate(hyperedges):
        for n in nodes:
            incidence.append([n, h_id])

    return torch.tensor(
        incidence, dtype=torch.long
    ).t().contiguous()

def build_pyg_hypergraph_extended(
    node_type,
    lund_features,
    hyperedges,
    label
):
    num_nodes = len(node_type)

    x = torch.zeros((num_nodes, 5), dtype=torch.float)

    for nid, feats in lund_features.items():
        x[nid] = torch.tensor(feats, dtype=torch.float)

    hyperedge_index = build_hyperedge_index_extended(hyperedges)

    return Data(
        x=x,
        hyperedge_index=hyperedge_index,
        y=torch.tensor(label, dtype=torch.long)
    )

def jet_json_to_extended_hypergraph(particles, label, R=0.8):
    jet = build_jet(particles, R)

    root, children, node_type, lund_features, hyperedges = \
        build_full_lund_tree(jet)

    hyperedges = build_extended_hyperedges(children, node_type)

    return build_pyg_hypergraph_extended(
        node_type,
        lund_features,
        hyperedges,
        label
    )

def load_extended_json_file(filename, label):
    graphs = []

    with open(filename, "r") as f:
        for line in tqdm(f, desc=f"Loading {filename}"):
            particles = json.loads(line)
            data = jet_json_to_extended_hypergraph(particles, label)
            graphs.append(data)

    return graphs
