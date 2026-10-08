import os

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Geometry import Point3D
from rdkit import RDLogger
import imageio
import networkx as nx
import numpy as np
import rdkit.Chem
import matplotlib.pyplot as plt


def fiedler_colors(graph):
    """Node colours from the second eigenvector of the normalized Laplacian, in graph.nodes()
    order, with a symmetric (vmin, vmax) for a diverging cmap.
    For tiny graphs (e.g., 1 node), fall back to constant colors."""
    node_colors = None
    vmin, vmax = None, None
    if graph.number_of_nodes() >= 2:
        try:
            _, U = np.linalg.eigh(nx.normalized_laplacian_matrix(graph).toarray())
            if U.ndim == 2 and U.shape[1] > 1:
                node_colors = U[:, 1]
                cur_min, cur_max = np.min(node_colors), np.max(node_colors)
                m = max(np.abs(cur_min), cur_max)
                vmin, vmax = -m, m
        except Exception:
            node_colors = None
    if node_colors is None:
        node_colors = np.zeros(graph.number_of_nodes(), dtype=float)
        vmin, vmax = -1.0, 1.0
    return node_colors, vmin, vmax


class MolecularVisualization:
    def __init__(self, remove_h, dataset_infos):
        self.remove_h = remove_h
        self.dataset_infos = dataset_infos

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """
        Convert graphs to rdkit molecules
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        # dictionary to map integer value to the char of atom
        atom_decoder = self.dataset_infos.atom_decoder

        # create empty editable mol object
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        n_types = len(atom_decoder)
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            idx = int(node_list[i])
            if idx >= n_types:
                raise ValueError(
                    f"Node index {idx} out of range for atom_decoder (len={n_types}). "
                    f"Check that model.node_decoder.node_classes == len(dataset_infos.atom_decoder)."
                )
            a = Chem.Atom(atom_decoder[idx])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                # only traverse half the symmetric matrix
                if iy <= ix:
                    continue
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    continue
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        try:
            mol = mol.GetMol()
        except rdkit.Chem.KekulizeException:
            print("Can't kekulize molecule")
            mol = None
        return mol

    def visualize(self, path: str, molecules: list, num_molecules_to_visualize: int):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        print(f"Visualizing {num_molecules_to_visualize} of {len(molecules)}")
        if num_molecules_to_visualize > len(molecules):
            print(f"Shortening to {len(molecules)}")
            num_molecules_to_visualize = len(molecules)

        # RDKit has no batched decode from dense (bs, n, n) tensors -- mol_from_graphs
        # necessarily builds one RWMol per sample. But the *rendering* doesn't have to
        # be N separate Draw.MolToFile calls: collect the valid mols here and render
        # them in one Draw.MolsToGridImage call below, matching how sampling itself
        # already processes the whole batch at once.
        valid_mols, valid_idx = [], []
        for i in range(num_molecules_to_visualize):
            file_path = os.path.join(path, "molecule_{}.png".format(i))
            mol = self.mol_from_graphs(molecules[i][0].numpy(), molecules[i][1].numpy())
            if mol is None:
                log_path = os.path.join(path, "visualization_log.txt")
                with open(log_path, "a") as f:
                    f.write(f"Molecule {i}: Invalid molecule (None)\n")
                continue
            if mol.GetNumAtoms() == 0:
                log_path = os.path.join(path, "visualization_log.txt")
                with open(log_path, "a") as f:
                    f.write(f"Molecule {i}: Molecule has no atoms\n")
                continue
            try:
                Draw.MolToFile(mol, file_path)
                valid_mols.append(mol)
                valid_idx.append(i)
            except (rdkit.Chem.KekulizeException, RuntimeError) as e:
                log_path = os.path.join(path, "visualization_log.txt")
                with open(log_path, "a") as f:
                    f.write(f"Molecule {i}: Failed to draw molecule - {str(e)}\n")

        if valid_mols:
            try:
                grid = Draw.MolsToGridImage(
                    valid_mols,
                    molsPerRow=10,
                    subImgSize=(200, 200),
                    legends=[f"molecule_{i}" for i in valid_idx],
                )
                grid.save(os.path.join(path, "molecules_grid.png"))
            except (rdkit.Chem.KekulizeException, RuntimeError) as e:
                log_path = os.path.join(path, "visualization_log.txt")
                with open(log_path, "a") as f:
                    f.write(f"Grid image creation failed: {str(e)}\n")

    def visualize_chain(self, path, nodes_list, adjacency_matrix, times, trainer=None):
        RDLogger.DisableLog("rdApp.*")
        # convert graphs to the rdkit molecules
        mols = [
            self.mol_from_graphs(nodes_list[i], adjacency_matrix[i])
            for i in range(nodes_list.shape[0])
        ]

        # Initialize if any visualization happened
        visualization_success = False

        # find the coordinates of atoms in the final molecule
        final_molecule = mols[-1]
        coords = []
        final_valid = True
        if final_molecule is None or final_molecule.GetNumAtoms() == 0:
            final_valid = False
            log_path = os.path.join(path, "visualization_log.txt")
            with open(log_path, "a") as f:
                f.write("Final molecule is invalid or has no atoms; skipping coordinate calculation and gif creation.\n")
        else:
            try:
                AllChem.Compute2DCoords(final_molecule)
                coords = []
                for i, atom in enumerate(final_molecule.GetAtoms()):
                    positions = final_molecule.GetConformer().GetAtomPosition(i)
                    coords.append((positions.x, positions.y, positions.z))
            except Exception as e:
                final_valid = False
                log_path = os.path.join(path, "visualization_log.txt")
                with open(log_path, "a") as f:
                    f.write(f"Failed to compute 2D coordinates for final molecule: {e}; skipping gif.\n")

        # align all the molecules if coordinates are available
        if final_valid and coords:
            for i, mol in enumerate(mols):
                if mol is not None and mol.GetNumAtoms() > 0:
                    try:
                        AllChem.Compute2DCoords(mol)
                        conf = mol.GetConformer()
                        for j, atom in enumerate(mol.GetAtoms()):
                            if j < len(coords):
                                x, y, z = coords[j]
                                conf.SetAtomPosition(j, Point3D(x, y, z))
                    except Exception as e:
                        log_path = os.path.join(path, "visualization_log.txt")
                        with open(log_path, "a") as f:
                            f.write(f"Failed to align molecule at frame {i}: {e}\n")

        # draw gif
        save_paths = []
        num_frams = nodes_list.shape[0]

        for frame in range(num_frams):
            file_name = os.path.join(path, "fram_{}.png".format(frame))
            if mols[frame] is None:
                log_path = os.path.join(path, "visualization_log.txt")
                with open(log_path, "a") as f:
                    f.write(f"Frame {frame} (t={times[frame]:.2f}): Invalid molecule (None)\n")
                continue
            if mols[frame].GetNumAtoms() == 0:
                log_path = os.path.join(path, "visualization_log.txt")
                with open(log_path, "a") as f:
                    f.write(f"Frame {frame} (t={times[frame]:.2f}): Molecule has no atoms\n")
                continue
            try:
                Draw.MolToFile(
                    mols[frame],
                    file_name,
                    size=(300, 300),
                    legend=f"t =  {times[frame]:.2f}",
                )
                save_paths.append(file_name)
                visualization_success = True
            except (rdkit.Chem.KekulizeException, RuntimeError) as e:
                log_path = os.path.join(path, "visualization_log.txt")
                with open(log_path, "a") as f:
                    f.write(f"Frame {frame} (t={times[frame]:.2f}): Failed to draw molecule - {str(e)}\n")
                continue

        if save_paths and visualization_success:
            try:
                imgs = [imageio.imread(fn) for fn in save_paths]
                gif_path = os.path.join(
                    os.path.dirname(path), "{}.gif".format(path.split("/")[-1])
                )
                imgs.extend([imgs[-1]] * 10)
                imageio.mimsave(gif_path, imgs, subrectangles=True, duration=200)
            except Exception as e:
                log_path = os.path.join(path, "visualization_log.txt")
                with open(log_path, "a") as f:
                    f.write(f"GIF creation failed: {str(e)}\n")

        # draw grid image
        try:
            img = Draw.MolsToGridImage(mols, molsPerRow=10, subImgSize=(200, 200))
            img.save(
                os.path.join(path, "{}_grid_image.png".format(path.split("/")[-1]))
            )
        except (Chem.rdchem.KekulizeException, RuntimeError) as e:
            log_path = os.path.join(path, "visualization_log.txt")
            with open(log_path, "a") as f:
                f.write(f"Grid image creation failed: {str(e)}\n")
        return mols


class NonMolecularVisualization:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def to_networkx(self, node_list, adjacency_matrix):
        """
        Convert graphs to networkx graphs
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        graph = nx.Graph()

        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            graph.add_node(i, number=i, symbol=node_list[i], color_val=node_list[i])

        rows, cols = np.where(adjacency_matrix >= 1)
        edges = zip(rows.tolist(), cols.tolist())
        for edge in edges:
            edge_type = adjacency_matrix[edge[0]][edge[1]]
            graph.add_edge(
                edge[0], edge[1], color=float(edge_type), weight=3 * edge_type
            )
        graph.remove_edges_from(nx.selfloop_edges(graph))

        return graph

    def visualize_non_molecule(
        self,
        graph,
        pos,
        path,
        iterations=100,
        node_size=100,
        largest_component=False,
        time=None,
    ):
        if largest_component:
            CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            graph = CGs[0]

        # Plot the graph structure with colors
        if pos is None:
            pos = nx.spring_layout(graph, iterations=iterations)

        node_colors, vmin, vmax = fiedler_colors(graph)

        plt.figure()
        nx.draw(
            graph,
            pos,
            font_size=5,
            node_size=node_size,
            with_labels=False,
            node_color=node_colors,
            cmap=plt.cm.coolwarm,
            vmin=vmin,
            vmax=vmax,
            edge_color="grey",
        )
        if time is not None:
            plt.text(
                0.5,
                0.05,  # place below the graph
                f"t = {time:.2f}",
                ha="center",
                va="center",
                transform=plt.gcf().transFigure,
                fontsize=16,
            )

        plt.tight_layout()
        plt.savefig(path)
        plt.close("all")

    def visualize(self, path: str, graphs: list, num_graphs_to_visualize: int):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, "graph_{}.png".format(i))
            graph = self.to_networkx(graphs[i][0].numpy(), graphs[i][1].numpy())
            self.visualize_non_molecule(graph=graph, pos=None, path=file_path)

    def visualize_chain(self, path, nodes_list, adjacency_matrix, times):

        graphs = []
        for i in range(nodes_list.shape[0]):
            graphs.append(self.to_networkx(nodes_list[i], adjacency_matrix[i]))

        # find the coordinates of atoms in the final molecule
        final_graph = graphs[-1]
        final_pos = nx.spring_layout(final_graph, seed=0)

        # draw gif
        save_paths = []
        num_frams = nodes_list.shape[0]

        for frame in range(num_frams):
            file_name = os.path.join(path, "fram_{}.png".format(frame))
            self.visualize_non_molecule(
                graph=graphs[frame],
                pos=final_pos,
                path=file_name,
                time=times[frame],
            )
            save_paths.append(file_name)

        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(
            os.path.dirname(path), "{}.gif".format(path.split("/")[-1])
        )
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=200)
