


# AIG grammar.

# Node feature vec is node_type (onehot), fanins, fanouts, out
# Edge feature vec is one hot encoding for inv or reg

# AIG Attributes

# Input nodes PI and constant 0 CONST_0 node remain the exact same except for outgoing connections (fanouts) which can change,
# but fanins=0 ie no incoming edges, type, id, out=0 (False) remain the same

# AND nodes can only have 2 incoming edges (fanins) but arbitrarily many outgoing, out=0

# all edges have to be either inverted or regular

# all edges go from source id to a larger id for target #TODO think on this

# all output are fanins=1 ie one edge incoming, fanout=0 no edges outgoing, and need an out feature > 0

# should have less than or equal to number of nodes

# should have less than or equal to number of edges??? #TODO think on this

# no edges PI to PI
# no edges PO to PO

