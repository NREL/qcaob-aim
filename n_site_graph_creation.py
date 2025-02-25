from enum import Enum
from networkx import grid_graph
from dataclasses import dataclass, field
import networkx as nx
import matplotlib.pyplot as plt


# Create some canonical test cases to test graph creation for n-site cases
# We only accept symmetric cases that have nearest neighbor connectivity on a grid - this is enforced by assertions \
# in create_connected_graphs()

@dataclass(order=True, frozen=False)  # Enabling to sort by number of sites
class NSiteModelIdx:
    # NOTE: idx should be assigned the index as assigned w/vacancies included - zero indexed
    sort_index: int = field(init=False, repr=False)
    vacancy_sites_idx: list[int]  # Can be an empty list
    bath_sites_idx: list[int]
    impurity_sites_idx: list[int]
    grid_dims: tuple[int, int]

    # TODO: change all these assignments with super such that we can freeze the dataclass
    def __post_init__(self):
        self.sort_index = (len(self.bath_sites_idx) + len(self.impurity_sites_idx)) // 2
        self.n_vacancy_sites = len(self.vacancy_sites_idx)
        self.n_bath_sites = len(self.bath_sites_idx) // 2  # Do this to follow standard literature nomenclature
        self.n_impurity_sites = len(self.impurity_sites_idx) // 2
        self.two_n_bath_sites = len(self.bath_sites_idx)
        self.two_n_impurity_sites = len(self.impurity_sites_idx)
        # Total size is the number of total sites required (assuming symmetry) to build the model (unaffected by grid dims)
        self.total_size = self.two_n_bath_sites + self.two_n_impurity_sites + self.n_vacancy_sites
        self.model_name = f'{self.sort_index} Site ({self.n_bath_sites}-Bath, {self.n_impurity_sites}-Impurity) AIM'

#@profile
def create_connected_graphs(n_site_model_idx: NSiteModelIdx, show_full_plot: bool = False, show_sub_graphs: bool = False) -> dict[str, nx.Graph]:
    """
    Given zero-indexed integer lists of bath_sites_idx, impurity_sites_idx, vacancy_sites_idx and a tuple of grid_dims (height, width), return 4 NetworkX Graphs:
    the total graph, up_site graph with connections, the down_site graph with connections, and a graph that has the connectivity at the interface.
    Note: Graphs are generated based on the assumption that for each node in the full graph there is only nearest neighbor connectivity.
    Note: n_sites is the total number of qubits (including up and down spin sites).  CHANGE THIS - SITES SHOULD BE HALF
    Note: The index labeling convention is as follows: indices start at the top left and traverse across a row to the right, starting again at the next
    highest row and so on. In other words it follows matrix ordering where you start with the lowest index row and traverse the column.
    Note: The graphs created by this function have nodes with helpful attributes that can be accessed including cartesian coordiantes - for graph plotting,
    matrix coordinates - if the node was an entry in a matrix this is the ordering, type - impurity (I) or bath (B), color - red for impurity and blue for bath,
    and spin - unicode up-arrow for spin up and unicode down arrow for spin down.
    """

    # TODO: when going to production, change assertion to a try/except block
    # Create the full graph
    gr = grid_graph(dim=n_site_model_idx.grid_dims)
    total_number_of_grid_nodes = gr.number_of_nodes()
    # Ensure there are no repeated indices from the model
    for impurity_idx, bath_idx in zip(n_site_model_idx.impurity_sites_idx, n_site_model_idx.bath_sites_idx):
        assert impurity_idx not in n_site_model_idx.vacancy_sites_idx, f'Repeated indices in NSiteModelIdx for impurity idx: {impurity_idx} and index in vacancy sites.'
        assert bath_idx not in n_site_model_idx.vacancy_sites_idx, f'Repeated indices in NSiteModelIdx for bath idx: {bath_idx} and index in vacancy sites.'
        assert impurity_idx != bath_idx, f'Repeated indices in NSiteModelIdx for impurity idx: {impurity_idx} and bath idx: {bath_idx}.'

    # Ensure that the total number of nodes that NetworkX will create is exactly the sum of all the lengths of lists we provide:
    assert total_number_of_grid_nodes == n_site_model_idx.total_size, f'Grid of size {total_number_of_grid_nodes} with dimension {n_site_model_idx.grid_dims} does not match commanded model from indicies of size {n_site_model_idx.total_size}.'

    # Save the original coordinates in an attribute for plotting later as consumed by NetworkX
    for node in gr.nodes():
        gr.nodes[node]['cartesian_coordinates'] = node

    # Convert cartesian coordinate nodes to matrix coordinates
    # Convert (x,y) to (r,c) coordinate system - translate all down by max(y), flip x/y assignments, make y negative: (r,c) = (-(y-max(y_vec)),x)
    cartesian_nodes = gr.nodes()
    y_max = max([n[1] for n in cartesian_nodes])  # Grab the max of the y cartesian coordinates
    convert_coordinates = lambda x, y, y_max: (-(y - y_max), x)
    mapping_cartesian_to_matrix = {cart_coord: convert_coordinates(cart_coord[0], cart_coord[1], y_max) for cart_coord
                                   in cartesian_nodes}
    gr = nx.relabel_nodes(gr, mapping_cartesian_to_matrix, copy=True)

    # Stash the matrix coordinates in nodes' attributes before re-indexing
    for node in gr.nodes():
        gr.nodes[node]['matrix_coordinates'] = node

    # Sort the matrix coordinates by row first then column
    matrix_coord_list = list(gr.nodes())
    matrix_coords_sorted = sorted(matrix_coord_list, key=lambda k: [k[0], k[1]])

    # Create mapping from matrix coordinates to indices
    mapping_matrix_to_index = {matrix_coord: idx for idx, matrix_coord in enumerate(matrix_coords_sorted)}
    gr = nx.relabel_nodes(gr, mapping_matrix_to_index, copy=True)

    # Create attributes for nodes (label of node, color) with bath sites, impurity sites, spin up and down registers:
    bath_graph = nx.Graph()
    impurity_graph = nx.Graph()
    vacancy_graph = nx.Graph()
    impurity_graph.add_nodes_from(n_site_model_idx.impurity_sites_idx)
    bath_graph.add_nodes_from(
        n_site_model_idx.bath_sites_idx)  # note you can add variable num of attribute names from this as well
    vacancy_graph.add_nodes_from(n_site_model_idx.vacancy_sites_idx)

    for imp_node in impurity_graph.nodes():
        gr.nodes[imp_node]['type'] = 'I'
        gr.nodes[imp_node]['color'] = 'tab:red'

    for b_node in bath_graph.nodes():
        gr.nodes[b_node]['type'] = 'B'
        gr.nodes[b_node]['color'] = 'tab:blue'

    for v_node in vacancy_graph.nodes():
        gr.nodes[v_node]['type'] = 'V'
        gr.nodes[v_node]['color'] = 'tab:black'

    # For each node we want to save the r,c coordinates and type - modify using ((r_max-r),c) and check that a node both has this coordinate and that it has the right type
    # We use this to check - if there is a cooresponding node such that symmetry is preserved, and that it has the right type
    r_max = max([r[0] for r in matrix_coord_list])
    for node in gr.nodes():
        matrix_coordinates = gr.nodes[node]['matrix_coordinates']
        current_type = gr.nodes[node]['type']
        corresponding_matrix_coordinates = (r_max - matrix_coordinates[0], matrix_coordinates[1])

        # This will return the node index corresponding to the symmetric element
        corresponding_node_index = [n for n, attributes in gr.nodes(data=True) if
                                    attributes['matrix_coordinates'] == corresponding_matrix_coordinates][0]
        assert corresponding_node_index is not None, f'No symmetric element for node at matrix coordinates {matrix_coordinates} of any type.'
        corresponding_type = gr.nodes[corresponding_node_index]['type']

        # Check that this node has the same label as the current node we're on for symmetry
        assert corresponding_type == current_type, f'Node at {matrix_coordinates} with type: {current_type} does not have the same type as corresponding node at {corresponding_matrix_coordinates} with type: {corresponding_type} - graph is not symmetric'

    # Take out the vacant sites before final re-indexing occurs:
    gr.remove_nodes_from(n_site_model_idx.vacancy_sites_idx)

    # Re-index so we have just the occupied sites with indices
    index_list = list(gr.nodes())
    index_list_sorted = sorted(index_list)
    mapping_old_to_new_index = {index: idx for idx, index in enumerate(index_list_sorted, start=0)}
    gr = nx.relabel_nodes(gr, mapping_old_to_new_index, copy=True)

    # Partition graphs into up/down spin registers based on location in graph, and stitch graph
    interface_idx = len(gr.nodes()) // 2

    # Grab the highest row value for up register, lowest row value for down register
    up_register_interface_idx = gr.nodes()[interface_idx - 1]['matrix_coordinates'][
        0]  # Grab the corresponding row from the tuple
    down_register_interface_idx = gr.nodes()[interface_idx]['matrix_coordinates'][0]

    # This gets around using deepcopy since it is computationally intensive
    up_register_graph = nx.Graph()
    up_register_graph.add_nodes_from(gr.nodes(data=True))
    up_register_graph.add_edges_from(gr.edges())
    down_register_graph = nx.Graph()
    down_register_graph.add_nodes_from(gr.nodes(data=True))
    down_register_graph.add_edges_from(gr.edges())
    stitch_graph = nx.Graph()
    stitch_graph.add_nodes_from(gr.nodes(data=True))
    stitch_graph.add_edges_from(gr.edges())

    # Construct up, down, stitch graphs by comparing the row indexes of the matrix coordinates
    for node in gr.nodes():
        if gr.nodes()[node]['matrix_coordinates'][0] <= up_register_interface_idx:
            # Remove node from down register and add up arrow in the original graph (doing this way to preserve attributes of original graph)
            gr.nodes[node]['spin'] = "\u2191"  # unicode up arrow (2193 is down)
            down_register_graph.remove_node(node)
        if gr.nodes()[node]['matrix_coordinates'][0] >= down_register_interface_idx:
            # Remove node from up register and add down arrow in the original graph
            up_register_graph.remove_node(node)
            gr.nodes[node]['spin'] = "\u2193"
        if gr.nodes()[node]['matrix_coordinates'][0] not in [down_register_interface_idx, up_register_interface_idx]:
            stitch_graph.remove_node(node)

    # Remove edges of nearest neighbors amongst up and down registers here
    for u, v in stitch_graph.edges():
        if stitch_graph.nodes()[u]['matrix_coordinates'][0] == stitch_graph.nodes()[v]['matrix_coordinates'][0]:
            stitch_graph.remove_edge(u, v)

    total_graph = gr
    # Use dict comprehension to build up labels, ({key:value for vars in iterable})
    # label_dict = {node:f"{node}, {G.nodes()[node]['label']}" for node in G.nodes()} labels with cartesian coordinates and imp/bath
    label_dict = {node: f"{gr.nodes()[node]['type']}, {node}, {gr.nodes()[node]['spin']} " for node in
                  gr.nodes()}  # Include arrow labels
    node_positions = nx.get_node_attributes(gr, 'cartesian_coordinates')
    color_map_full = [gr.nodes[node]['color'] for node in gr.nodes()]
    color_map_up = [up_register_graph.nodes[node]['color'] for node in up_register_graph.nodes()]
    color_map_down = [down_register_graph.nodes[node]['color'] for node in down_register_graph.nodes()]
    color_map_stitch = [stitch_graph.nodes[node]['color'] for node in stitch_graph.nodes()]
    # TODO: Color edges and add interaction term labeling

    # draw graphs
    if show_full_plot:
        plt.figure(figsize=(12, 12))
        #plt.title(f'{n_site_model_idx.model_name} Graph', fontsize = 20)
        n_imp = int(len(n_site_model_idx.impurity_sites_idx)/2)
        n_bath = int(len(n_site_model_idx.bath_sites_idx)/2)
        n_sites = int(gr.number_of_nodes()/2)
        plt.title(f'{n_sites} Site AIM Graph ($ N_{{bath}}= {n_bath},N_{{imp.}}= {n_imp} $)', fontsize = 20)
        nx.draw(gr, node_positions, node_size=8000, labels=label_dict, with_labels=True, node_color=color_map_full,
                font_color="whitesmoke", font_size=20)
        #plt.show()

    if show_sub_graphs:
        fig = plt.figure(figsize=(15, 10))
        ax_up = fig.add_subplot(2, 2, 1)  # nrows, ncolumns, plot number
        nx.draw(up_register_graph, node_positions, node_size=1250, labels=label_dict, with_labels=True,
                node_color=color_map_up, font_color="whitesmoke", font_size=6)
        ax_up.set_title('Spin Up Register ')

        ax_down = fig.add_subplot(2, 2, 3)
        nx.draw(down_register_graph, node_positions, node_size=1250, labels=label_dict, with_labels=True,
                node_color=color_map_down, font_color="whitesmoke", font_size=6)
        ax_down.set_title('Spin Down Register ')

        ax_stitch = fig.add_subplot(4, 2, (4, 6))
        nx.draw(stitch_graph, node_positions, node_size=1250, labels=label_dict, with_labels=True,
                node_color=color_map_stitch, font_color="whitesmoke", font_size=6)
        ax_stitch.set_title('Stitch Graph Register')

    # plt.gcf().set_dpi(300)
        # plt.show()
    # plt.savefig(f'ansatz_graphs/{int(gr.number_of_nodes()/2)}_site.pdf')

    return {"graph_full": total_graph, "graph_up": up_register_graph,
            "graph_down": down_register_graph, "graph_stitch": stitch_graph}


# Create an enum that holds all the different allowable types of models
# - allow them to be indexed in a way that on calling funcs in dmft.py you can access the corresponding model
class AIMSiteModelsEnum(Enum):
    # We want to bind the member names to a value (n_bath, n_imp) and the NSiteModelIdx
    # This enum will be used for now to enforce canonical n-site aim cases while preserving most of the workflow in
    # dmft.py (passing in n to aim.n_site_up_down_symmetric_example() and impurity_orbital to calculate_gf)
    ONE_BATH_ONE_IMP = (1, 1)
    TWO_BATH_ONE_IMP = (2, 1)
    THREE_BATH_ONE_IMP = (3, 1)
    FOUR_BATH_ONE_IMP = (4, 1)
    FIVE_BATH_ONE_IMP = (5, 1)
    FIVE_BATH_TWO_IMP = (5, 2)
    SIX_BATH_ONE_IMP = (6, 1)
    SEVEN_BATH_ONE_IMP = (7, 1)
    EIGHT_BATH_ONE_IMP = (8, 1)
    NINE_BATH_ONE_IMP = (9, 1)
    TEN_BATH_ONE_IMP = (10, 1)
    # Add your own model as a member to this enum here:

    def __str__(self):
        return f"AIM {self.value[0] + self.value[1]}-Site Model Idx Enum - {self.value[0]} Bath Sites, {self.value[1]} Impurity Sites"

    def create_n_site_model_idx(self):
        # Check the type of enum - assign indexes based on self.value
        if self.value == (1, 1):
            vacancy_sites_idx = []
            bath_sites_idx = [1, 3]
            impurity_sites_idx = [0, 2]
            grid_dims = (2, 2)
        elif self.value == (2, 1):
            vacancy_sites_idx = []
            bath_sites_idx = [0, 2, 3, 5]
            impurity_sites_idx = [1, 4]
            grid_dims = (2, 3)
        elif self.value == (3, 1):
            vacancy_sites_idx = [0, 2, 9, 11]
            bath_sites_idx = [1, 3, 5, 6, 8, 10]
            impurity_sites_idx = [4, 7]
            grid_dims = (4, 3)
        elif self.value == (4, 1):
            vacancy_sites_idx = [0, 9]
            bath_sites_idx = [1, 2, 3, 5, 6, 8, 10, 11]
            impurity_sites_idx = [4, 7]
            grid_dims = (4, 3)
        elif self.value == (5, 1):
            vacancy_sites_idx = []
            bath_sites_idx = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11]
            impurity_sites_idx = [4, 7]
            grid_dims = (4, 3)
        elif self.value == (5, 2):
            vacancy_sites_idx = [0, 4,
                                 9,
                                 14,
                                 15, 19]
            bath_sites_idx = [1, 2, 3,
                              5, 8,
                              10, 13,
                              16, 17, 18]
            impurity_sites_idx = [6, 7, 11, 12]
            grid_dims = (4, 5)
        elif self.value == (6, 1):
            vacancy_sites_idx = [0, 4,
                                 9,
                                 14,
                                 15, 19]
            bath_sites_idx = [1, 2, 3,
                              5, 6, 8,
                              10, 11, 13,
                              16, 17, 18]
            impurity_sites_idx = [7, 12]
            grid_dims = (4, 5)
        elif self.value == (7, 1):
            vacancy_sites_idx = [0, 4,
                                 15, 19]
            bath_sites_idx = [1, 2, 3,
                              5, 6, 8, 9,
                              10, 11, 13, 14,
                              16, 17, 18]
            impurity_sites_idx = [7, 12]
            grid_dims = (4, 5)
        elif self.value == (8, 1):
            vacancy_sites_idx = [0, 1, 3, 4,
                                 5, 9,
                                 20, 24,
                                 25, 26, 28, 29]
            bath_sites_idx = [2,
                              6, 7, 8,
                              10, 11, 13, 14,
                              15, 16, 18, 19,
                              21, 22, 23,
                              27
                              ]
            impurity_sites_idx = [12, 17]
            grid_dims = (6, 5)
        elif self.value == (9, 1):
            vacancy_sites_idx = [0, 3, 4,
                                 5, 9,
                                 20, 24,
                                 25, 28, 29]
            bath_sites_idx = [1, 2,
                              6, 7, 8,
                              10, 11, 13, 14,
                              15, 16, 18, 19,
                              21, 22, 23,
                              26, 27
                              ]
            impurity_sites_idx = [12, 17]
            grid_dims = (6, 5)

        elif self.value == (10, 1):
            vacancy_sites_idx = [0, 4,
                                 5, 9,
                                 20, 24,
                                 25, 29]
            bath_sites_idx = [1, 2, 3,
                              6, 7, 8,
                              10, 11, 13, 14,
                              15, 16, 18, 19,
                              21, 22, 23,
                              26, 27, 28
                              ]
            impurity_sites_idx = [12, 17]
            grid_dims = (6, 5)
        n_site_model_idx = NSiteModelIdx(vacancy_sites_idx=vacancy_sites_idx,
                                         bath_sites_idx=bath_sites_idx,
                                         impurity_sites_idx=impurity_sites_idx,
                                         grid_dims=grid_dims)
        return n_site_model_idx


def main():
    # Use enum to call on a canonical case, retrieve associated NSiteModelIdx, create connected graphs
    n_bath_n_imp_tup = (5, 2)
    print(AIMSiteModelsEnum(n_bath_n_imp_tup))
    #Create graphs using NSiteModelIdx - replace the first arg with the desired aim enum member
    connected_graphs = create_connected_graphs(AIMSiteModelsEnum(n_bath_n_imp_tup).create_n_site_model_idx(),
                                         show_full_plot=False,
                                         show_sub_graphs=False)
    print(type(connected_graphs))
    edges = connected_graphs['graph_full'].number_of_edges()
    print(f'edges: {edges}')

    # TODO: Turn these into Pytest fixtures
    # Call on all canonical cases:
    # for n_site_model_idx_member in AIMSiteModelsEnum:
    #     print(f'Enum Member Name: {n_site_model_idx_member.name} - (Bath, Imp): {n_site_model_idx_member.value}')
    #     _, _, _, _ = create_connected_graphs(n_site_model_idx_member.create_n_site_model_idx(), True, False)

    # Testing that assertions are thrown for malformed models:

    # Throw error for non-symmetric creation:
    # error_one_bath_one_imp_idx = NSiteModelIdx(vacancy_sites_idx=[], bath_sites_idx=[0, 3], impurity_sites_idx=[1, 2],
    #                                            grid_dims=(2, 2))
    # u, d, s = create_connected_graphs(error_one_bath_one_imp_idx, show_full_plot=False, show_sub_graphs=False)

    # Throw grid dim error:
    # error_one_bath_one_imp_idx = NSiteModelIdx(vacancy_sites_idx=[], bath_sites_idx=[0, 3], impurity_sites_idx=[1, 2],
    #                                          grid_dims=(3, 2))
    # u, d, s = create_connected_graphs(error_one_bath_one_imp_idx, show_full_plot=False, show_sub_graphs=False)


if __name__ == "__main__":
    main()
