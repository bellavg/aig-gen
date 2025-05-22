

Baselines to test

|           |             |          |             |                     |             |
|-----------|-------------|----------|-------------|---------------------|-------------|
| Graph-RNN | Not Running | NA       | no test yet |                     |             |
| DiGress   | Running...  | NA       | Untested    |                     | eval fucked |
| LayerDAG  | Running...  | untested |             |                     |             |

 AIGMetrics Results Epoch 100 (Test) ---
31
  aig_metrics/structural_validity_fraction: 0.4370
32
  aig_metrics/acyclicity_fraction: 0.8690
33
  aig_validity_errors/VALID: 0.4370
34
  aig_validity_errors/NOT_DAG: 0.1310
35
  aig_validity_errors/PI_IN_DEGREE: 0.0030
36
  aig_validity_errors/AND_IN_DEGREE: 0.2770
37
  aig_validity_errors/PO_IN_DEGREE: 0.1510
38
  aig_validity_errors/PO_OUT_DEGREE: 0.0010
39
  aig_metrics/frac_graphs_mostly_padding_edges: 0.7110
40
  aig_metrics/avg_actual_aig_edges_formed: 32.9240
41
  sampling_quality/frac_unique_aigs: 0.9970
42
  sampling_quality/frac_unique_non_iso_aigs: 0.9840
43
  sampling_quality/frac_unique_non_iso_structurally_valid_aigs: 0.4220
44
  sampling_quality/frac_non_iso_to_train_aigs: 0.9850

- Graph-RNN - NA - Untested
- DiGress - Running - Untested
- DeFoG - Trying to train - Untested
- dig
  - Graph-DF - Trying to train - fix generate - fix eval
  - Graph-AF - Ready to train? - fix generate - fix eval
  - GraphEBM - Ready to train? - fix generate - fix eval
- LayerDAG - Trying to train - Untested



  - SeaDAG and Circuit Transformer - done but not available for testing 

