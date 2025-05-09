import torch
import torch.nn as nn

from .disgraphaf import DisGraphAF


class GraphFlowModel(nn.Module):
    def __init__(self, model_conf_dict):
        super(GraphFlowModel, self).__init__()
        self.max_size = model_conf_dict['max_size']
        self.edge_unroll = model_conf_dict['edge_unroll']
        self.node_dim = model_conf_dict['node_dim']
        self.bond_dim = model_conf_dict['bond_dim']

        node_masks, adj_masks, link_prediction_index, self.flow_core_edge_masks = self.initialize_masks(
            max_node_unroll=self.max_size, max_edge_unroll=self.edge_unroll)

        self.latent_step = node_masks.size(
            0)  # (max_size) + (max_edge_unroll - 1) / 2 * max_edge_unroll + (max_size - max_edge_unroll) * max_edge_unroll
        self.latent_node_length = self.max_size * self.node_dim
        self.latent_edge_length = (self.latent_step - self.max_size) * self.bond_dim

        self.dp = model_conf_dict['use_gpu']

        node_base_log_probs = torch.randn(self.max_size, self.node_dim)
        edge_base_log_probs = torch.randn(self.latent_step - self.max_size, self.bond_dim)
        self.flow_core = DisGraphAF(node_masks, adj_masks, link_prediction_index,
                                    num_flow_layer=model_conf_dict['num_flow_layer'], graph_size=self.max_size,
                                    num_node_type=self.node_dim, num_edge_type=self.bond_dim,
                                    num_rgcn_layer=model_conf_dict['num_rgcn_layer'],
                                    nhid=model_conf_dict['nhid'], nout=model_conf_dict['nout'])
        if self.dp:
            self.flow_core = nn.DataParallel(self.flow_core)
            self.node_base_log_probs = nn.Parameter(node_base_log_probs.cuda(), requires_grad=True)
            self.edge_base_log_probs = nn.Parameter(edge_base_log_probs.cuda(), requires_grad=True)
        else:
            self.node_base_log_probs = nn.Parameter(node_base_log_probs, requires_grad=True)
            self.edge_base_log_probs = nn.Parameter(edge_base_log_probs, requires_grad=True)

    def forward(self, inp_node_features, inp_adj_features):
        """
        Args:
            inp_node_features: (B, N, 9)
            inp_adj_features: (B, 4, N, N)

        Returns:
            z: [(B, node_num*9), (B, edge_num*4)]
            logdet:  ([B], [B])
        """
        inp_node_features_cont = inp_node_features.clone()  # (B, N, 9)

        inp_adj_features_cont = inp_adj_features[:, :, self.flow_core_edge_masks].clone()  # (B, 4, edge_num)
        inp_adj_features_cont = inp_adj_features_cont.permute(0, 2, 1).contiguous()  # (B, edge_num, 4)

        z = self.flow_core(inp_node_features, inp_adj_features, inp_node_features_cont, inp_adj_features_cont)
        return z

    # def generate(self, temperature=[0.3, 0.3], min_atoms=5, max_atoms=64, disconnection_patience=10):
    #     """
    #     inverse flow to generate molecule
    #     Args:
    #         temp: temperature of normal distributions, we sample from (0, temp^2 * I)
    #     """
    #
    #     disconnection_streak = 0
    #
    #     current_device = torch.device("cuda" if self.dp and torch.cuda.is_available() else "cpu")
    #
    #     with torch.no_grad():
    #         #num2bond = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
    #         num2bond =NUM2EDGETYPE
    #         #num2atom = {i: atom_list[i] for i in range(len(atom_list))}
    #         num2atom = NUM2NODETYPE
    #
    #         cur_node_features = torch.zeros([1, max_atoms, self.node_dim], device=current_device)
    #         cur_adj_features = torch.zeros([1, self.bond_dim, max_atoms, max_atoms], device=current_device)
    #
    #         node_features_each_iter_backup = cur_node_features.clone()  # backup of features, updated when newly added node is connected to previous subgraph
    #         adj_features_each_iter_backup = cur_adj_features.clone()
    #
    #         #rw_mol = Chem.RWMol()  # editable mol
    #         aig = nx.DiGraph() # editable aig
    #
    #         #mol = None
    #         graph = None
    #
    #         is_continue = True
    #         edge_idx = 0
    #         total_resample = 0
    #         #each_node_resample = np.zeros([max_atoms])
    #
    #         for i in range(max_atoms):
    #             if not is_continue:
    #                 break
    #             if i < self.edge_unroll:
    #                 num_edges_to_try = i  # used to be called edge_total
    #                 start = 0
    #             else:
    #                 num_edges_to_try = self.edge_unroll
    #                 start = i - self.edge_unroll
    #             # first generate node
    #             ## reverse flow
    #
    #             prior_node_dist = torch.distributions.OneHotCategorical(
    #                 logits=self.node_base_log_probs[i].to(current_device) * temperature[0])
    #
    #             latent_node_type_sample = prior_node_dist.sample().view(1, -1)
    #
    #             if self.dp:
    #                 latent_node_type_sample = self.flow_core.module.reverse(cur_node_features, cur_adj_features,
    #                                                                         latent_node_type_sample,mode=0).view(-1)
    #             else:
    #                 latent_node_type_sample = self.flow_core.reverse(cur_node_features, cur_adj_features,
    #                                                                  latent_node_type_sample, mode=0).view(-1)  # (9, )
    #
    #             feature_id = torch.argmax(latent_node_type_sample).item()
    #             cur_node_features[0, i, feature_id] = 1.0
    #             cur_adj_features[0, :, i, i] = 1.0
    #
    #             # Original: #rw_mol.AddAtom(Chem.Atom(num2atom[feature_id]))
    #             # Node ID in NetworkX graph is `i`
    #             aig.add_node(i, type=num2atom[feature_id])
    #
    #             is_connect = (i == 0)
    #
    #             for j in range(num_edges_to_try):
    #                 source_prev_node_idx = j + start
    #
    #                 valid = False
    #                 resample_edge = 0
    #                 edge_dis = self.edge_base_log_probs[edge_idx].clone().to(current_device)
    #                 invalid_bond_type_set = set()
    #
    #                 while not valid:
    #                     # I believe this is 3 because that is where three virtual edge index is
    #                     #if len(invalid_bond_type_set) < 3 and resample_edge <= 50:  # haven't sampled all possible bond type or is not stuck in the loop
    #                     if len(invalid_bond_type_set) < VIRTUAL_EDGE_INDEX and resample_edge <= 50:
    #                         prior_edge_dist = torch.distributions.OneHotCategorical(logits=edge_dis / temperature[1])
    #                         latent_edge = prior_edge_dist.sample().view(1, -1)
    #                         latent_id = torch.argmax(latent_edge, dim=1)
    #
    #                         if self.dp:
    #                             latent_edge = self.flow_core.module.reverse(cur_node_features, cur_adj_features,
    #                                                                         latent_edge,
    #                                                                         mode=1, edge_index=torch.Tensor(
    #                                     [[source_prev_node_idx, i]]).long().cuda()).view(-1)  # (4, )
    #                         else:
    #                             latent_edge = self.flow_core.reverse(cur_node_features, cur_adj_features, latent_edge,
    #                                                                  mode=1, edge_index=torch.Tensor(
    #                                     [[source_prev_node_idx, i]]).long()).view(-1)  # (4, )
    #                         edge_discrete_id = torch.argmax(latent_edge).item()
    #                     else:
    #                         #assert resample_edge > 50 or len(invalid_bond_type_set) == 3
    #                         assert resample_edge > 50 or len(invalid_bond_type_set) == VIRTUAL_EDGE_INDEX #I believe that is why this is 3
    #                         #edge_discrete_id = 3  # we have no choice but to choose not to add edge between (i, j+start)
    #                         edge_discrete_id = VIRTUAL_EDGE_INDEX
    #
    #                     #cur_adj_features[0, edge_discrete_id, i, source_prev_node_idx] = 1.0
    #                     cur_adj_features[0, edge_discrete_id, source_prev_node_idx, i] = 1.0
    #                     # remove because directed
    #                     if edge_discrete_id == VIRTUAL_EDGE_INDEX:  # virtual edge
    #                         valid = True
    #                     else:  # single/double/triple bond
    #                         # Original: #rw_mol.AddBond(i, j + start, num2bond[edge_discrete_id])
    #                         aig.add_edge(source_prev_node_idx, i, type=num2bond[edge_discrete_id])
    #
    #                         # Original: #valid = check_valency(rw_mol)
    #                         valid = check_validity(aig)  # Your AIG validity check
    #
    #                         if valid:
    #                             is_connect = True
    #                             # print(num2bond_symbol[edge_discrete_id])
    #                         else:  # backtrack
    #                             edge_dis[latent_id] = float('-inf')
    #                             #rw_mol.RemoveBond(i, j + start)
    #                             aig.remove_edge(source_prev_node_idx, i,)
    #                             cur_adj_features[0, edge_discrete_id, source_prev_node_idx, i] = 0.0
    #                             #cur_adj_features[0, edge_discrete_id, i, j + start] = 0.0
    #                             total_resample += 1.0
    #                             resample_edge += 1
    #                             invalid_bond_type_set.add(edge_discrete_id)
    #
    #                 edge_idx += 1
    #
    #             if is_connect:  # new generated node has at least one bond with previous node, do not stop generation, backup mol from rw_mol to mol
    #                 is_continue = True
    #                 #mol = rw_mol.GetMol()
    #                 graph = aig.copy()
    #                 node_features_each_iter_backup = cur_node_features.clone()  # update node backup since new node is valid
    #                 adj_features_each_iter_backup = cur_adj_features.clone()
    #                 disconnection_streak = 0
    #             elif not is_connect and disconnection_streak < disconnection_patience:
    #                 is_continue = True
    #                 graph = aig
    #                 node_features_each_iter_backup = cur_node_features.clone()  # update node backup since new node is valid
    #                 adj_features_each_iter_backup = cur_adj_features.clone()
    #                 disconnection_streak += 1
    #             else:
    #                 is_continue = False
    #                 #cur_mol_size = mol.GetNumAtoms()
    #                 cur_graph_size = graph.number_of_nodes()
    #                 #if cur_mol_size >= min_atoms:
    #                 if cur_graph_size >= min_atoms:
    #                     #rw_mol = Chem.RWMol(mol)
    #                     aig = graph
    #                     cur_node_features = node_features_each_iter_backup.clone()
    #                     cur_adj_features = adj_features_each_iter_backup.clone()
    #                     cur_node_features_tmp = cur_node_features.clone()
    #                     cur_adj_features_tmp = cur_adj_features.clone()
    #                     #mol_demon_edit = Chem.RWMol(rw_mol)
    #                     # TODO what do i do here?? instead of the line above
    #                     added_num = np.random.randint(1, 5)
    #                     # why added_num 1, 5???  is
    #                     for _ in range(added_num):
    #                         last_id2 = mol_demon_edit.AddAtom(Chem.Atom(6))
    #                         # TODO figure out why this and what this is?
    #                         cur_node_features_tmp[0, last_id2, 0] = 1.0
    #                         cur_adj_features_tmp[0, :, last_id2, last_id2] = 1.0
    #
    #                         flag_success = False
    #                         count = 0
    #                         while True:
    #                             if count > 100:
    #                                 break
    #                             if last_id2 > 12:
    #                                 j = np.random.randint(1, 13)
    #                             else:
    #                                 j = np.random.randint(1, last_id2 + 1)
    #
    #                             mol_demon_edit.AddBond(int(last_id2 - j), int(last_id2), Chem.rdchem.BondType.SINGLE)
    #                             # TODO figure out why this and what this is?
    #                             cur_adj_features_tmp[0, 0, last_id2 - j, last_id2] = 1.0
    #                             cur_adj_features_tmp[0, 0, last_id2, last_id2 - j] = 1.0
    #
    #                             #valid = check_valency(mol_demon_edit)
    #                             #TODO figureout what goes here
    #                             valid = check_validity()
    #                             if valid:
    #                                 flag_success = True
    #                                 break
    #                             else:
    #                                 mol_demon_edit.RemoveBond(int(last_id2 - j), int(last_id2))
    #                                 # TODO figureout what goes here and why
    #                                 cur_adj_features_tmp[0, 0, last_id2 - j, last_id2] = 0.0
    #                                 cur_adj_features_tmp[0, 0, last_id2, last_id2 - j] = 0.0
    #                                 count += 1
    #
    #                         if flag_success:
    #                             rw_mol = Chem.RWMol(mol_demon_edit)
    #                             # TODO figureout what goes here and why?
    #                             cur_node_features = cur_node_features_tmp.clone()
    #                             cur_adj_features = cur_adj_features_tmp.clone()
    #                             #mol = rw_mol.GetMol()
    #                             graph = aig
    #                             node_features_each_iter_backup = cur_node_features.clone()
    #                             adj_features_each_iter_backup = cur_adj_features.clone()
    #                         else:
    #                             break
    #
    #
    #         #assert mol is not None, 'mol is None...'
    #         assert graph is not None
    #
    #         #final_mol = convert_radical_electrons_to_hydrogens(mol)
    #         #TODO i think above is unncesary
    #         #num_atoms = final_mol.GetNumAtoms()
    #         num_nodes = graph.number_of_nodes()
    #
    #         pure_valid = 0
    #         if total_resample == 0:
    #             pure_valid = 1.0
    #
    #         return graph, pure_valid, num_nodes

    def initialize_masks(self, max_node_unroll=38, max_edge_unroll=12):
        """
        Args:
            max node unroll: maximal number of nodes in molecules to be generated (default: 38)
            max edge unroll: maximal number of edges to predict for each generated nodes (default: 12, calculated from zink250K data)
        Returns:
            node_masks: node mask for each step
            adj_masks: adjacency mask for each step
            is_node_update_mask: 1 indicate this step is for updating node features
            flow_core_edge_mask: get the distributions we want to model in adjacency matrix
        """
        num_masks = int(
            max_node_unroll + (max_edge_unroll - 1) * max_edge_unroll / 2 + (max_node_unroll - max_edge_unroll) * (
                max_edge_unroll))
        num_mask_edge = int(num_masks - max_node_unroll)

        node_masks1 = torch.zeros([max_node_unroll, max_node_unroll]).bool()
        adj_masks1 = torch.zeros([max_node_unroll, max_node_unroll, max_node_unroll]).bool()
        node_masks2 = torch.zeros([num_mask_edge, max_node_unroll]).bool()
        adj_masks2 = torch.zeros([num_mask_edge, max_node_unroll, max_node_unroll]).bool()
        link_prediction_index = torch.zeros([num_mask_edge, 2]).long()
        flow_core_edge_masks = torch.zeros([max_node_unroll, max_node_unroll]).bool()

        cnt = 0
        cnt_node = 0
        cnt_edge = 0
        for i in range(max_node_unroll):
            node_masks1[cnt_node][:i] = 1
            adj_masks1[cnt_node][:i, :i] = 1
            cnt += 1
            cnt_node += 1

            edge_total = 0
            if i < max_edge_unroll:
                start = 0
                edge_total = i
            else:
                start = i - max_edge_unroll
                edge_total = max_edge_unroll
            for j in range(edge_total):
                if j == 0:
                    node_masks2[cnt_edge][:i + 1] = 1
                    adj_masks2[cnt_edge] = adj_masks1[cnt_node - 1].clone()
                    adj_masks2[cnt_edge][i, i] = 1
                else:
                    node_masks2[cnt_edge][:i + 1] = 1
                    adj_masks2[cnt_edge] = adj_masks2[cnt_edge - 1].clone()
                    adj_masks2[cnt_edge][i, start + j - 1] = 1
                    adj_masks2[cnt_edge][start + j - 1, i] = 1
                cnt += 1
                cnt_edge += 1
        assert cnt == num_masks, 'masks cnt wrong'
        assert cnt_node == max_node_unroll, 'node masks cnt wrong'
        assert cnt_edge == num_mask_edge, 'edge masks cnt wrong'

        cnt = 0
        for i in range(max_node_unroll):
            if i < max_edge_unroll:
                start = 0
                edge_total = i
            else:
                start = i - max_edge_unroll
                edge_total = max_edge_unroll

            for j in range(edge_total):
                link_prediction_index[cnt][0] = start + j
                link_prediction_index[cnt][1] = i
                cnt += 1
        assert cnt == num_mask_edge, 'edge mask initialize fail'

        for i in range(max_node_unroll):
            if i == 0:
                continue
            if i < max_edge_unroll:
                start = 0
                end = i
            else:
                start = i - max_edge_unroll
                end = i
            flow_core_edge_masks[i][start:end] = 1

        node_masks = torch.cat((node_masks1, node_masks2), dim=0)
        adj_masks = torch.cat((adj_masks1, adj_masks2), dim=0)

        node_masks = nn.Parameter(node_masks, requires_grad=False)
        adj_masks = nn.Parameter(adj_masks, requires_grad=False)
        link_prediction_index = nn.Parameter(link_prediction_index, requires_grad=False)
        flow_core_edge_masks = nn.Parameter(flow_core_edge_masks, requires_grad=False)

        return node_masks, adj_masks, link_prediction_index, flow_core_edge_masks

    def dis_log_prob(self, z):
        x_deq, adj_deq = z
        node_base_log_probs_sm = torch.nn.functional.log_softmax(self.node_base_log_probs, dim=-1)
        ll_node = torch.sum(x_deq * node_base_log_probs_sm, dim=(-1, -2))
        edge_base_log_probs_sm = torch.nn.functional.log_softmax(self.edge_base_log_probs, dim=-1)
        ll_edge = torch.sum(adj_deq * edge_base_log_probs_sm, dim=(-1, -2))
        return -(torch.mean(ll_node + ll_edge) / (self.latent_edge_length + self.latent_node_length))