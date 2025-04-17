

# final_data.pkl

## Overview
This dataset contains 37,000 graph representations of And-Inverter Graphs (AIGs) stored in pickle format. Each graph is a NetworkX DiGraph object representing the structure and functionality of a digital logic circuit.

## Dataset Structure

### Graph Properties
- **Format**: NetworkX DiGraph objects in a pickled list
- **Number of graphs**: 37,000
- **Node count range**: Typically between 16-20 nodes per graph
- **Edge count range**: Typically between 17-27 edges per graph
- **Input size range**: 3-5 inputs
- **Output size range**: 4-5 outputs

### Node Attributes
Each node in the graph has the following attributes:
- **type**: One-hot encoded vector representing node type:
  - `[0, 0, 0]`: Constant-0 node
  - `[1, 0, 0]`: Primary Input (PI)
  - `[0, 1, 0]`: AND gate
  - `[0, 0, 1]`: Primary Output (PO)
- **feature**: Truth table for the node, represented as a binary list or numpy array
  - Length depends on number of inputs (2^inputs)
  - Common lengths: 8 (for 3 inputs), 16 (for 4 inputs), 32 (for 5 inputs), 256 (for 8 inputs)
  - Contains 0s and 1s representing the output value for each possible input combination

### Edge Attributes
Each edge has a **type** attribute:
- Numpy array representing edge type (one-hot encoded):
  - `[1, 0]`: Inverted edge (INV)
  - `[0, 1]`: Regular edge (REG)

### Graph-Level Attributes
Each graph contains the following graph-level attributes:
- **inputs**: Number of primary inputs (integer)
- **outputs**: Number of primary outputs (integer)
- **tts**: empty array for and gates, input patterns and output patterns
  - Each element is a binary list representing the node's truth table
- **full_tts**: Complete list of truth tables for all nodes so tts + symbolic simulation 
- **output_tts**: List of truth tables only for output nodes
  - Length equals number of outputs
- **binary_tt_dict**: (In some graphs) Dictionary mapping node IDs to binary string representations of truth tables

## Truth Table Structure
- First element (index 0): Constant-0 node (all zeros)
- Next elements (indices 1 to inputs): Primary inputs
  - For 4 inputs, the truth tables represent standard binary patterns:
    - Input 1: `[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]` (half 0s, half 1s)
    - Input 2: `[0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1]` (quartets of 0s and 1s)
    - Input 3: `[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]` (pairs of 0s and 1s)
    - Input 4: `[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]` (alternating 0s and 1s)
- Remaining elements: AND gates and Primary Outputs with computed truth tables

## Node Distribution (Typical)
- Constant-0: 1 node per graph
- Primary Inputs (PI): 3-5 nodes per graph
- AND gates: 7-9 nodes per graph
- Primary Outputs (PO): 4-5 nodes per graph

## Data Format Notes
- Some graphs may use numpy arrays instead of lists for node features
- Truth table lengths vary based on the number of inputs (2^inputs)
- Edge types are consistently represented as numpy arrays
- The dataset was filtered to include only graphs with:
  - Maximum 8 inputs
  - Maximum 8 outputs
  - Maximum 120 nodes

# current_data.pkl
AIG Graph Dataset Summary with Updated Node Type Encodings
Dataset Overview

Format: Pickle file containing NetworkX DiGraph objects
Size: 37,000 graph representations of And-Inverter Graphs (AIGs)
Typical graph size: 16-20 nodes and 17-27 edges per graph
Circuit parameters: 3-5 inputs and 4-5 outputs per graph

Updated Node Type Encoding
The dataset now uses a true one-hot encoding for node types:

Constant-0 node: [1, 0, 0, 0] (previously [0, 0, 0])
Primary Input (PI): [0, 1, 0, 0] (previously [1, 0, 0])
AND gate: [0, 0, 1, 0] (previously [0, 1, 0])
Primary Output (PO): [0, 0, 0, 1] (previously [0, 0, 1])

Node Structure

Each node contains:

type: One-hot encoded vector (4-dimensional) indicating node type
feature: Truth table for the node (binary list/array representing output for all input combinations)
Feature length varies with number of inputs (2^inputs): 8 values for 3 inputs, 16 for 4 inputs, etc.



Edge Properties

Edges represent connections between nodes in the AIG
Each edge has a type attribute:

[1, 0]: Inverted edge (INV)
[0, 1]: Regular edge (REG)



Graph-Level Properties
Each graph contains:

inputs: Number of primary inputs (integer)
outputs: Number of primary outputs (integer)
tts: List of truth tables (empty arrays for AND gates)
full_tts: Complete list of truth tables for all nodes
output_tts: List of truth tables only for output nodes

Truth Table Structure

The truth tables follow standard binary patterns:

For 4 inputs, the primary input truth tables are:

Input 1: [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1] (half 0s, half 1s)
Input 2: [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1] (quartets of 0s and 1s)
Input 3: [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1] (pairs of 0s and 1s)
Input 4: [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1] (alternating 0s and 1s)



This dataset with updated node type encodings maintains all the structural and functional information of the
original AIGs while using a consistent one-hot encoding scheme. This should improve machine learning model 
performance, particularly for generation tasks, by ensuring each node type has exactly one "hot" (1) value and 
eliminating the all-zero encoding previously used for constant-0.