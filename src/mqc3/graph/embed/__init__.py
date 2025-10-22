r"""Embed a |DependencyDAG| into a |GraphRepr|.

This module provides functionality to embed a |DependencyDAG|, which represents
dependencies between operations in a quantum circuit, into a |GraphRepr|.

About |DependencyDAG|
----------------------------------------------------

A |DependencyDAG| represents a quantum circuit as a Directed Acyclic Graph (DAG),
where each |Operation| is a node and the dependencies in execution order
between operations are edges. While it describes the logical structure of the circuit,
it does not contain information regarding the physical placement of operations.

Each node in a |DependencyDAG| has the following attributes.

.. list-table::
   :widths: 15 25 60
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``op``
     - |Operation|
     - The quantum operation represented by the node.
   * - ``modes``
     - ``list[int]``
     - A list of IDs for the modes targeted by the operation (length 1 for 1-mode ops, length 2 for 2-mode ops).
   * - ``displacements``
     - ``list[Displacement]``
     - A list of ``Displacement``\ s applied immediately before the operation (**at most one element**).
   * - ``has_nlff``
     - ``bool``
     - Whether the operation includes a non-linear |feedforward|.

- Nodes: Correspond to operations such as |Initialization|, |Measurement|, and
  |PhaseRotation|. They also have the attributes listed above.
- Edges: Represent dependencies regarding the execution order of operations. The main dependencies are as follows:
    - operations on the same mode: operations on the same mode are executed in the order they appear in the circuit.
    - |feedforward| : A dependency arises when the measurement result of a certain mode is used as a parameter for a
      subsequent operation.

The constructor for |DependencyDAG| accepts either a |GraphRepr| or a
|CircuitRepr| as input.

Embedding Rules into |GraphRepr|
----------------------------------------------------

The embedding process is a transformation that maps an abstract
|DependencyDAG| to a concrete |GraphRepr| with a 2D grid of macronodes.

Requirements
~~~~~~~~~~~~
All modes within the |DependencyDAG| must originate at an |Initialization| and
be terminated at a |Measurement| (each exactly once). A |DependencyDAG| where modes
disappear midway or appear from nowhere cannot be embedded.
The list of ``Displacement``\ s applied immediately before an operation has at most one element.

Inputs
~~~~~~
- |DependencyDAG|: The directed acyclic graph to be embedded.
- |GraphEmbedSettings|: A settings object that defines the constraints for the embedding process.
    - ``n_local_macronodes``: The number of macronodes to be placed in a single step (local height).
    - ``feedforward_distance``: Specifies the upper and lower bounds for the **difference in macronode indices**
      between a measurement operation and the operation that uses its result for a |feedforward|.
    - ``max_columns`` (optional): The maximum number of columns considered during embedding (finite layout budget).
    - Others: Parameters for the optimization algorithm corresponding to each |GraphEmbedder| can be passed.
        - *Example:* For a beam-search embedder, |BeamSearchEmbedSettings| extends |GraphEmbedSettings|
          with ``beam_width`` (default 10), the number of candidate solutions retained at each step.

Correspondence Rules
~~~~~~~~~~~~~~~~~~~~

Each element of the |DependencyDAG| is mapped to an element of the
|GraphRepr| according to the following rules.

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - |DependencyDAG|
     - |GraphRepr|
     - Description
   * - node
     - macronode
     - Each operation is assigned to exactly one macronode on the |GraphRepr|.
   * - edge
     - Ordering of macronodes
     - If an edge exists from node ``u`` to ``v``, a constraint :math:`i_u < i_v`
       is imposed on the indices :math:`i_u, i_v` of the corresponding
       macronodes.
   * - mode
     - Path on macronodes
     - A series of operations associated with each mode in the |DependencyDAG|
       forms a connected path that sequentially passes through macronodes on the |GraphRepr|.
   * - displacement
     - Parameters between macronodes
     - Displacements, which are aggregated in operation nodes
       in the |DependencyDAG|, are converted into parameters on the **edge** immediately
       preceding the operation (``displacement_k_minus_1``, ``displacement_k_minus_n``)
       on the |GraphRepr|.

Indexing and 2D Grid
~~~~~~~~~~~~~~~~~~~~

The |GraphRepr| is a 2D grid of macronodes with coordinates :math:`(h, w)`, where
:math:`h \in \{0,\dots,N_{local}-1\}` and :math:`w \in \{0,\dots,W_{\max}\}`.
The linear index :math:`i` of a macronode is defined by

.. math::

   i = w \times N_{local} + h

This index :math:`i` is used for **topological ordering** and for evaluating **feedforward distance**.
The path-length metric described in the optimization objective is used **only** in the objective,
not in any constraints.

**Vertical / Horizontal edges and column-advance.**
An index difference of **1** corresponds to a **vertical** edge within a column
**(including the column-advance from the bottom of a column to the top of the next)**,
while an index difference of **N** (``n_local_macronodes``) corresponds to a **horizontal**
edge to the next column.

Column-advance connection (bottom-to-next-top)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the usual within-column and within-row connections:

- Horizontal: :math:`(h,w-1).\mathrm{out\_right} \to (h,w).\mathrm{in\_left}`
- Vertical (inside a column): :math:`(h-1,w).\mathrm{out\_bottom} \to (h,w).\mathrm{in\_top}` for :math:`h>0`
- **Column-advance**: :math:`(N_{\mathrm{local}}-1, w).\mathrm{out\_bottom} \to (0, w+1).\mathrm{in\_top}`

This edge allows a mode leaving the bottom of a column to enter the top of the next column.

Detailed Rules for Each Element
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- mode:
    - A mode is represented as a path connecting macronodes on the |GraphRepr| and must be contiguous.
    - Each macronode receives modes from the top (``in_top``) and left (``in_left``),
      and outputs modes to the bottom (``out_bottom``) and right (``out_right``). Each port has unit capacity.
    - For macronodes not at the ends of a path, if ``swap=False``, the input mode
      from the top is output to the bottom, and the input mode from the left is
      output to the right. If ``swap=True``, the input mode from the top is output
      to the right, and the input mode from the left is output to the bottom.
    - Inputs or outputs to which no mode is assigned are treated as ``BLANK_MODE`` for
      convenience (``BLANK_MODE`` is only produced/consumed by |Initialization| / |Measurement|).

- |Operation|:
    - Each |Operation| is assigned to exactly one macronode.
    - |Initialization|: Creates one or two modes. Both inputs must be ``BLANK_MODE``.
      - 1-mode init: exactly one of ``out_bottom`` or ``out_right`` becomes the initialized mode.
      - 2-mode init: both outputs become distinct initialized modes.
      The output modes are specified by the ``initialized_mode`` parameter.
    - |Measurement|: Measures exactly one input mode. The other input must be ``BLANK_MODE``.
      Both outputs will be ``BLANK_MODE``. The consumed input side (top or left) is specified
      by a parameter (or by a fixed convention).
    - 1-mode operation: Receives one mode, processes it, and then outputs that mode according to the routing rule.
    - 2-mode operation: Receives two **distinct** modes, processes them, and then outputs those modes.

- displacement:
    - On the |DependencyDAG|, it is not an independent node but is modeled as a list
      ``displacements: list[Displacement]`` attached to the subsequent operation node
      (the list has **at most one element**).
    - When embedded into a |GraphRepr|, each displacement is converted into parameters on the
      **edge** immediately preceding the operation. Concretely,
      ``displacement_k_minus_1`` is attached to **vertical** edges (index difference = 1,
      including column-advance), and ``displacement_k_minus_n`` is attached to **horizontal**
      edges (index difference = :math:`N` = ``n_local_macronodes``).
    - **Mapping constraints:** (i) **At most one displacement per edge** (edge capacity = 1).
      (ii) **Each displacement must map to exactly one edge** immediately preceding its operation.
      Thus, displacement-edge mapping is one-to-one.

- |feedforward|:
    - Let the indices of the macronodes performing the measurement and using its result be
      :math:`i_{meas}` and :math:`i_{op}`, respectively. A strict order constraint
      :math:`i_{meas} < i_{op}` is imposed.
    - Furthermore, the upper and lower bounds for the **index-difference distance** are
      specified by ``feedforward_distance = (min_dist, max_dist)`` in |GraphEmbedSettings|.

    .. math::

       \mathrm{min\_dist} \le i_{op} - i_{meas} \le \mathrm{max\_dist}

Embedding Optimization
----------------------

Embedding is the problem of finding a placement that satisfies all the above
constraints. However, multiple solutions may exist. To select the best one, the problem
is formulated as an optimization problem to find the optimal placement according to the
objective function shown below.

Optimization Objective
~~~~~~~~~~~~~~~~~~~~~~

The objective is to minimize, over all modes :math:`m`, the **number of macronodes
traversed by each mode**, summed across modes. Let :math:`\operatorname{path\_len}(m)` be
the count of macronodes visited along the contiguous path of mode :math:`m` on the grid
(including the column-advance transition from the bottom of a column to the top of
the next column). Then:

.. math::

   \min \sum_{m \in \mathrm{modes}} \operatorname{path\_len}(m)

**Note.** This path-length metric is used **only** in the optimization objective and is **not**
used in any constraints (e.g., feedforward distance is index-based).

This objective is chosen to mitigate the propagation of noise, which is assumed
to accumulate at each macronode a mode traverses. By seeking a solution that
minimizes this value, the embedding process aims to shorten the path length of
the modes, thereby reducing their exposure to potential noise.

.. |GraphRepr| replace:: :class:`~mqc3.graph.GraphRepr`
.. |Operation| replace:: :class:`~mqc3.graph.ops.Operation`
.. |Initialization| replace:: :class:`~mqc3.graph.ops.Initialization`
.. |Measurement| replace:: :class:`~mqc3.graph.ops.Measurement`
.. |PhaseRotation| replace:: :class:`~mqc3.graph.ops.PhaseRotation`
.. |DependencyDAG| replace:: :class:`~mqc3.graph.embed.dep_dag.DependencyDAG`
.. |GraphEmbedder| replace:: :class:`~mqc3.graph.embed.embed.GraphEmbedder`
.. |GraphEmbedSettings| replace:: :class:`~mqc3.graph.embed.embed.GraphEmbedSettings`
.. |BeamSearchEmbedSettings| replace:: :class:`~mqc3.graph.embed.beamsearch.BeamSearchEmbedSettings`
.. |CircuitRepr| replace:: :class:`~mqc3.circuit.CircuitRepr`
.. |feedforward| replace:: :mod:`~mqc3.feedforward`
"""
