# `graph.embed` Module

This module provides functionality to embed a `DependencyDAG`, which represents dependencies between operations in a quantum circuit, into a `GraphRepr`.

## About `DependencyDAG`

A `DependencyDAG` represents a quantum circuit as a Directed Acyclic Graph (DAG), where each operation (`Operation`) is a **node** and the dependencies in execution order between operations are **edges**.  
While it describes the logical structure of the circuit, it does not contain information regarding the physical placement of operations.

Each node in a `DependencyDAG` has the following attributes.

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `op` | `Operation` | The quantum operation represented by the node. |
| `modes` | `list[int]` | A list of IDs for the `mode`s targeted by the operation (length 1 for 1-mode ops, length 2 for 2-mode ops). |
| `displacements` | `list[Displacement]` | A list of `Displacement`s applied immediately before the operation (at most one element). |
| `has_nlff` | `bool` | Whether the operation includes a non-linear `FeedForward`. |

- Nodes: Correspond to operations such as `Initialization`, `Measurement`, and `PhaseRotation`.
  - Nodes also have the attributes listed above (`op`, `modes`, `displacements`, `has_nlff`).
- Edges: Represent dependencies regarding the execution order of operations. The main dependencies are as follows:
  - Operations on the same `mode`: Operations on the same `mode` are executed in the order they appear in the circuit.
  - `FeedForward`: A dependency arises when the measurement result of a certain `mode` is used as a parameter for a subsequent operation.

`DependencyDAG` is constructed from a `GraphRepr` or a `CircuitRepr`.

## Embedding Rules into `GraphRepr`

The embedding process is a transformation that maps an abstract `DependencyDAG`, which only has the logical structure of the circuit, to a `GraphRepr`, which includes the specific placement of operations.

### Requirements

All `mode`s within the `DependencyDAG` must originate from an `Initialization` and terminate at a `Measurement` (each exactly once).  
A `DependencyDAG` where `mode`s disappear midway or appear from nowhere cannot be embedded.  
The list of `Displacement`s applied immediately before an operation has at most one element.

### Inputs

- `DependencyDAG`: The directed acyclic graph to be embedded.
- `GraphEmbedSettings`: A settings object that defines the constraints for the embedding process.
  - `n_local_macronodes`: The number of macronodes to be placed in a single step (local height).
  - `feedforward_distance`: Specifies the upper and lower bounds for the difference in **macronode index** between a `Measurement` and the operation that uses its result for a `FeedForward` (see note on indexing).
  - `max_columns` (optional): The maximum number of columns considered during embedding (finite layout budget).
  - Others: Parameters for the optimization algorithm corresponding to each `GraphEmbedder` can be passed.
    - Example: For a beam-search embedder, `BeamSearchEmbedSettings` extends `GraphEmbedSettings` with `beam_width` (default 10), the number of candidate solutions retained at each step.

### Correspondence Rules

Each element of the `DependencyDAG` is mapped to an element of the `GraphRepr` according to the following rules.

| `DependencyDAG` | `GraphRepr` | Description |
| :---: | :---: | :--- |
| Node | `macronode` | Each operation is assigned to exactly one `macronode` on the `GraphRepr`. |
| Edge | Ordering of `macronode`s | If an edge exists from node $u$ to $v$, a constraint $i_u < i_v$ is imposed on the indices $i_u, i_v$ of the corresponding `macronode`s (see note on indexing). |
| `mode` | Path on `macronode`s | A series of operations associated with each `mode` in the `DependencyDAG` forms a connected path that sequentially passes through `macronode`s on the `GraphRepr`. |
| `Displacement` | Parameters between `macronode`s | `Displacement`s, which are attached to operation nodes in the `DependencyDAG`, are converted into parameters on the **edge** immediately preceding the operation in the `GraphRepr` (e.g., `displacement_k_minus_1`, `displacement_k_minus_n`). |

**Indexing and geometry note.**  
`GraphRepr` uses a 2D grid of macronodes with coordinates $(h,w)$ where $h \in \{0,\dots,N_{local}-1\}$ and $w \in \{0,\dots,W_{\max}\}$.  
The linear index of a `macronode` is defined by $i = w \times N_{local} + h$.  
This index is used for **topological ordering** and for evaluating **feedforward distance**.  
The path-length metric described below is used **only** in the optimization objective.  
An index difference of **1** corresponds to a **vertical** edge within a column **(including the column-advance from the bottom of a column to the top of the next)**, while an index difference of **N** (= `n_local_macronodes`) corresponds to a **horizontal** edge to the next column.

**Column-advance connection (bottom-to-next-top).**  
In addition to the usual within-column and within-row connections:

- Horizontal: $(h,w-1).\text{out\_right} \rightarrow (h,w).\text{in\_left}$  
- Vertical (inside a column): $(h-1,w).\text{out\_bottom} \rightarrow (h,w).\text{in\_top}$ for $h>0$  
- **Column-advance**: $(N_{local}-1, w).\text{out\_bottom} \rightarrow (0, w+1).\text{in\_top}$  
  This edge allows a mode leaving the bottom of a column to enter the top of the next column.

### Detailed Rules for Each Element

- `mode`:
  - A `mode` is represented as a path connecting `macronode`s on the `GraphRepr` and must be contiguous.
  - Each `macronode` has four ports with unit capacity: it receives `mode`s from the top (`in_top`) and left (`in_left`), and outputs `mode`s to the bottom (`out_bottom`) and right (`out_right`).
  - For `macronode`s not at the ends of a path, if `swap=False`, the input `mode` from the top is output to the bottom, and the input `mode` from the left is output to the right. If `swap=True`, the input `mode` from the top is output to the right, and the input `mode` from the left is output to the bottom.
  - At the bottom row ($h = N_{local}-1$), `out_bottom` connects to the next column’s top input as described in the column-advance rule.
  - Inputs or outputs to which no `mode` is assigned are treated as `BLANK_MODE` for convenience; `BLANK_MODE` cannot be produced or consumed except by `Initialization`/`Measurement` as described below.
- `Operation`:
  - Each `Operation` is assigned to exactly one `macronode`. Unless otherwise specified, two different operations do not share the same macronode index (i.e., if `allow_node_sharing` is disabled, $i_u \ne i_v$).
  - `Initialization`: Creates one or two `mode`s. Both inputs must be `BLANK_MODE`.  
    - 1-mode init: exactly one of the outputs (`out_bottom` or `out_right`) becomes the initialized `mode`.  
    - 2-mode init: both outputs become distinct initialized `mode`s.  
    The output port(s) and `mode` IDs are specified by the `initialized_mode` parameter.
  - `Measurement`: Measures exactly one input `mode`. The other input must be `BLANK_MODE`. Both outputs will be `BLANK_MODE`. The consumed input side (top or left) is specified by a parameter (or by a fixed convention).
  - 1-mode operation: Receives one `mode`, processes it, and then outputs that `mode` according to the `swap` routing rule.
  - 2-mode operation: Receives two **distinct** `mode`s, processes them, and then outputs those `mode`s according to the `swap` routing rule.
- `Displacement`:
  - On the `DependencyDAG`, it is not an independent node but is modeled as a list `displacements: list[Displacement]` attached to the subsequent operation node. The list has **at most one element**.
  - When embedded into a `GraphRepr`, each listed `Displacement` is converted into parameter(s) on the **edge** immediately preceding the operation (`displacement_k_minus_1`, `displacement_k_minus_n`). The exact semantics of `k, n` follow the library’s displacement type and composition rule.
  - Concretely, `displacement_k_minus_1` is attached to **vertical** edges **(including the column-advance)** (index difference = 1), and `displacement_k_minus_n` is attached to **horizontal** edges (index difference = N = `n_local_macronodes`).
  - **Capacity and mapping constraints:** (i) **At most one displacement per edge** (edge capacity = 1). (ii) **Each displacement must map to exactly one edge** immediately preceding its operation. Thus, displacement–edge mapping is one-to-one.

- `FeedForward`:
  - Let the `macronode` performing the measurement have index $i_{meas}$ and the `macronode` using its result have index $i_{op}$.  
    A strict order constraint $i_{meas} < i_{op}$ is imposed.
  - Additionally, the **feedforward distance is evaluated by index difference** and must satisfy the bounds in `GraphEmbedSettings.feedforward_distance`:
    $$
    \text{min\_dist} \le i_{op} - i_{meas} \le \text{max\_dist}.
    $$

## Embedding Optimization

Embedding is the problem of finding a placement that satisfies all the above constraints.  
However, multiple solutions may exist. To select the best one, the problem is formulated as an optimization problem to find the optimal placement according to the objective function shown below.

### Optimization Objective

Objective: Minimize, over all `mode`s $m$, the **number of macronodes traversed by each mode**, summed across modes.  
Formally, let $\operatorname{path\_len}(m)$ be the count of macronodes visited along the contiguous path of mode $m$ on the grid (including transitions via the column-advance edge from the bottom of a column to the top of the next column). Then:
$$
\min \sum_{m \in \text{modes}} \operatorname{path\_len}(m)
$$

This objective function is chosen to mitigate the propagation of noise, which is assumed to accumulate at each `macronode` a `mode` traverses.  
By seeking a solution that minimizes this value, the embedding process aims to shorten the path length of the `mode`s, thereby reducing their exposure to potential noise.
