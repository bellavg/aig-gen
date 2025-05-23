

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


RNN
Training graphs not provided or empty. Novelty is 100% relative to an empty set if valid graphs exist.

--- AIG V.U.N. Evaluation Summary ---
Total Graphs Generated & Evaluated: 1000
Structurally Valid AIGs (V)     : 14 (1.40%)
Unique Valid AIGs             : 14
Uniqueness (U) among valid    : 1.0000 (100.00%)
Novel Valid AIGs vs Train Set : 14
Novelty (N) among valid       : 1.0000 (100.00%)

--- Average Structural Metrics (All Processed Graphs) ---
  - Avg num_nodes                  : 35.889 (Std: 16.470)
  - Percentage is_dag                : 100.00%
  - Avg and_indegree_violations    : 1.870 (Std: 1.376)
  - Avg num_pi                     : 7.445 (Std: 1.300)
  - Avg num_po                     : 14.083 (Std: 13.813)
  - Avg num_and                    : 13.361 (Std: 6.731)
  - Avg const0_indegree_violations : 0.000 (Std: 0.000)
  - Avg pi_indegree_violations     : 0.004 (Std: 0.077)
  - Avg po_outdegree_violations    : 9.539 (Std: 11.030)
  - Avg po_indegree_violations     : 0.223 (Std: 0.933)
  - Avg num_unknown_nodes          : 0.000 (Std: 0.000)
  - Avg num_unknown_edges          : 0.000 (Std: 0.000)

--- Constraint Violation Summary (Across All Graphs Attempted) ---
  (Violations summarized across 1000 graphs attempted)
  - AND in-degree != 2                           : 853    occurrences (85.3% of total graphs had this issue at least once)
  - PO out-degree != 0                           : 596    occurrences (59.6% of total graphs had this issue at least once)
  - PO in-degree == 0                            : 83     occurrences (8.3% of total graphs had this issue at least once)
  - General Validity Rules Failed                : 43     occurrences (4.3% of total graphs had this issue at least once)
  - PI in-degree != 0                            : 3      occurrences (0.3% of total graphs had this issue at least once)


- Graph-RNN - NA - Untested
- DiGress - Running - Untested
- DeFoG - Trying to train - Untested
- dig
  - Graph-DF - Trying to train - fix generate - fix eval
  - Graph-AF - Ready to train? - fix generate - fix eval
  - GraphEBM - Ready to train? - fix generate - fix eval
- LayerDAG - Trying to train - Untested



  - SeaDAG and Circuit Transformer - done but not available for testing 

