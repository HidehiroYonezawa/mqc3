(sec:theory)=

# Theory

## Introduction

The machinery handled by MQC3 is an optical quantum computer, which generates a continuous-variable two-dimensional cluster state and performs measurement-based quantum computing (MBQC).
This page provides a theoretical explanation of the operating principles and detailed terminology of MQC3.
We begin by describing the quantum optics fundamentals and the machinery configuration, and then move on to a comprehensive explanation of the theoretical framework, which is covered across several sections.

### Basic terms of quantum optics

The carrier of quantum information is light, and the quantum state is characterized by the observables

```{math}
\begin{align}
\hat{x}&:=\frac{1}{\sqrt{2}}\left(\hat{a}+\hat{a}^{\dagger}\right), \\
\hat{p}&:=\frac{1}{i\sqrt{2}}\left(\hat{a}-\hat{a}^{\dagger}\right),
\end{align}
```

which cannot be measured simultaneously due to the uncertainty relation.
Here, {math}`\hat{a}^{\dagger}` and {math}`\hat{a}` are the creation and annihilation operators, respectively.
These operators {math}`\hat{x}` and {math}`\hat{p}` are referred to as quadrature and correspond to the real and imaginary parts of the complex amplitude of the electric field.
The measured values of these observables are continuous, ranging from {math}`-\infty` to {math}`\infty`.
Such a quantum state is referred to as a mode or qumode, analogous to how a quantum state is referred to as a qubit in a two-level system.

```{note}
In MQC3, we use the convention {math}`\hbar=1`.
```

{numref}`fig:squeezed_states` illustrates two examples of such quantum states.
These images represent the probability distributions of the measured values {math}`x` and {math}`p`, corresponding to the observables {math}`\hat{x}` and {math}`\hat{p}`, respectively.
Since {math}`\hat{x}` and {math}`\hat{p}` cannot be measured simultaneously, rather than joint probability distribution, this distribution is interpreted as a quasi-probability distribution called Wigner function.
Only projection measurements along a single axis in this phase space can be performed, and the probability distribution is determined by integrating out the variable in the orthogonal direction.
In these examples, the measured values follow Gaussian distributions for any measurement basis.
Such quantum states are referred to as Gaussian states, and specifically, the states depicted here are squeezed states.
The left shows an {math}`x`-squeezed state, while the right shows a {math}`p`-squeezed state.
Squeezed states can exist at arbitrary angles.
In the machinery handled by MQC3, the desired operations are achieved by repeatedly converting one squeezed state into another.

```{figure} _images/squeezed_states.svg
:name: fig:squeezed_states
:alt: Squeezed states
:width: 70%

Conceptual diagrams of {math}`x`-squeezed and {math}`p`-squeezed states. The red ellipses represent the probability density functions of measured values.
```

### Operators and observables

In the Heisenberg picture, when a unitary operator {math}`\hat{U}` acts on a quantum state, the observable {math}`\hat{O}` is transformed as

```{math}
\hat{O} \rightarrow \hat{U}^{\dagger} \hat{O} \hat{U}.
```

For example, the 50:50 beam splitter operation {math}`\hat{B}_\downarrow` acting on a two-mode state consisting of modes 1 and 2 is defined as

```{math}
---
label: eq:beamsplitter
---
\hat{B}_\downarrow^{\dagger}\left(\begin{array}{l}
\hat{x}_1 \\
\hat{x}_2 \\
\hat{p}_1 \\
\hat{p}_2
\end{array}\right) \hat{B}_\downarrow=\frac{1}{\sqrt{2}}\left(\begin{array}{cccc}
1 & -1 & 0 & 0 \\
1 & 1 & 0 & 0 \\
0 & 0 & 1 & -1 \\
0 & 0 & 1 & 1
\end{array}\right)\left(\begin{array}{l}
\hat{x}_1 \\
\hat{x}_2 \\
\hat{p}_1 \\
\hat{p}_2
\end{array}\right)
=B_\downarrow\left(\begin{array}{l}
\hat{x}_1 \\
\hat{x}_2 \\
\hat{p}_1 \\
\hat{p}_2
\end{array}\right).
```

A real symplectic matrix {math}`S_{\hat{U}}`, such as {math}`S_{\hat{B}_\downarrow}=B_\downarrow`, is referred to as the Heisenberg action of {math}`\hat{U}` acting on {math}`\hat{O}`.

Another example of a two-mode operation is the inverse 50:50 beam splitter {math}`\hat{B}_\uparrow`, which realizes

```{math}
\hat{B}_\uparrow^{\dagger}\left(\begin{array}{l}
\hat{x}_1 \\
\hat{x}_2 \\
\hat{p}_1 \\
\hat{p}_2
\end{array}\right) \hat{B}_\uparrow
=B_\uparrow\left(\begin{array}{l}
\hat{x}_1 \\
\hat{x}_2 \\
\hat{p}_1 \\
\hat{p}_2
\end{array}\right)
=B_\downarrow^{\dagger}\left(\begin{array}{l}
\hat{x}_1 \\
\hat{x}_2 \\
\hat{p}_1 \\
\hat{p}_2
\end{array}\right).
```

```{seealso}
Other operations are listed in {ref}`sec:gates`.
```

(sec:machinery)=

### Machinery

The machinery handled by MQC3 is similar to the experimental setup in {cite}`doi:10.1126/science.aay2645` and generates a time-domain multiplexed two-dimensional cluster state.
{numref}`fig:machinery_schematic` shows a simplified schematic of the machinery.

```{figure} _images/machinery_schematic.svg
:name: fig:machinery_schematic
:alt: Schematic of the machinery

Schematic of the machinery.
Four modes whose spatial indices are labeled {math}`a`, {math}`b`, {math}`c`, and {math}`d`, are injected continuously, and the temporal modes are defined at time intervals of {math}`\Delta t`.
Modes {math}`a` and {math}`c` are {math}`p`-squeezed states, while modes {math}`b` and {math}`d` are {math}`x`-squeezed states.
The beam splitters, depicted as light blue rectangles on the far left, entangle modes {math}`a` and {math}`b`, as well as modes {math}`c` and {math}`d`.
These entanglements are represented by the thick blue lines connecting the modes indicated by yellow circles.
Subsequently, modes {math}`b` and {math}`d` are delayed by {math}`\Delta t` and {math}`N\Delta t`, respectively, through optical delay lines.
The delay shown here is based on the assumption that {math}`N=3`.
After that, the four modes at the same time position {math}`a_k`, {math}`b_k`, {math}`c_k`, and {math}`d_k` are converted by the foursplitter composed of four beam splitters and then measured separately using homodyne detection.
```

```{caution}
Although {numref}`fig:machinery_schematic` shows the case where {math}`N=3`, in the actual machinery, {math}`N=101`.
```

After the delay lines, four modes that are temporally aligned at the same position are assigned the same index {math}`k`, such as {math}`a_k`, {math}`b_k`, {math}`c_k`, and {math}`d_k`.
The tuple of these four modes is referred to as a macronode {math}`k`, while each individual mode is called micronode.
At this point, each macronode {math}`k` is already connected through the entanglements between its micronodes and those in {math}`k\pm 1` and {math}`k\pm N`, forming a lattice-like structure.
The modes {math}`a_k`, {math}`b_k`, {math}`c_k`, and {math}`d_k` are subsequently subjected to operations by a foursplitter {math}`\hat{A}`, which implements

```{math}
\hat{A}^{\dagger}
\left(\begin{array}{l}
\hat{x}_a \\
\hat{x}_b \\
\hat{x}_c \\
\hat{x}_d \\
\hat{p}_a \\
\hat{p}_b \\
\hat{p}_c \\
\hat{p}_d
\end{array}\right)
\hat{A}
=\frac{1}{2}
\left(\begin{array}{cccccccc}
1 & 1 & -1 & -1 &  &  &  &  \\
-1 & 1 & 1 & -1 &  &  &  &  \\
1 & 1 & 1 & 1 &  &  &  &  \\
-1 & 1 & -1 & 1 &  &  &  &  \\
 &  &  &  & 1 & 1 & -1 & -1 \\
 &  &  &  & -1 & 1 & 1 & -1 \\
 &  &  &  & 1 & 1 & 1 & 1 \\
 &  &  &  & -1 & 1 & -1 & 1
\end{array}\right)
\left(\begin{array}{l}
\hat{x}_a \\
\hat{x}_b \\
\hat{x}_c \\
\hat{x}_d \\
\hat{p}_a \\
\hat{p}_b \\
\hat{p}_c \\
\hat{p}_d
\end{array}\right)
=A
\left(\begin{array}{l}
\hat{x}_a \\
\hat{x}_b \\
\hat{x}_c \\
\hat{x}_d \\
\hat{p}_a \\
\hat{p}_b \\
\hat{p}_c \\
\hat{p}_d
\end{array}\right)
```

and form a quad-rail lattice (QRL) {cite}`PhysRevA.93.062326` for efficient quantum computation.
Finally, each micronode is measured.
The four measurement angles {math}`\boldsymbol{\theta}=\left(\theta_a, \theta_b, \theta_c, \theta_d\right)` are varied with a period {math}`\Delta t` as ..., {math}`\boldsymbol{\theta}_{k-1}`, {math}`\boldsymbol{\theta}_{k}`, {math}`\boldsymbol{\theta}_{k+1}`, ..., determining the operations applied to the modes.

### Overview of the following sections

The following sections cover the theory used in the computations by MQC3.
In {ref}`sec:teleportation-circuit`, the teleportation circuit is introduced as the fundamental unit of quantum computation, along with measurement-based operations and feedforward.
{ref}`sec:macronode-circuit` describes how macronode circuits are constructed by combining teleportation circuits.
In {ref}`sec:gate-construction`, the method for implementing the desired operations using macronode circuits is explained.
{ref}`sec:feedforward` discusses how feedforward is applied to measured values rather than quantum states.
{ref}`sec:displacement` explains how displacement operations are performed.

(sec:teleportation-circuit)=

## Teleportation circuit

The teleportation circuit shown on the left in {numref}`fig:teleportation_circuit` is considered the fundamental component of the QRL. The left side can be simplified as shown on the right.

```{figure} _images/teleportation_circuit.svg
:name: fig:teleportation_circuit
:alt: Teleportation circuit

Teleportation circuit.
Mode 1, represented by the quadratures {math}`\left(\hat{x}_1, \hat{p}_1\right)`, is the input mode, while mode 2, represented by the quadratures {math}`\left(\hat{x}_2, \hat{p}_2\right)`, is one of the modes of the two-mode entanglement used in the quantum teleportation protocol and serves as the medium for teleportation.
After mode 1 and 2 interact via a beam splitter, they are measured at homodyne angles {math}`\theta_1` and {math}`\theta_2`, respectively.
As a result, the information from mode 1 is transmitted to the bottom mode, which is entangled with mode 2 as indicated by the thick blue line.
Finally, the feedforward {math}`\boldsymbol{f}_\text{tel}` is applied to the bottom mode, represented by the quadratures {math}`\left(\hat{x}_\text{out}, \hat{p}_\text{out}\right)`.
The left diagram can be simplified as shown on the right.
The beam splitter is represented by a black line with light blue and blue circles at its endpoints.
Each homodyne measurement sections has also been simplified into a single detector.
Additionally, the black dashed lines representing the classical connections for feedforward have been omitted.
```

The homodyne measurement angle {math}`\theta` for a mode {math}`\left(\hat{x},\hat{p}\right)` is defined to realize the measurement of

```{math}
---
label: eq:measurement
---
\hat{x}\sin{\theta}+\hat{p}\cos{\theta}.
```

Due to the inherent uncertainty in quantum measurement, the target mode in teleportation is displaced in phase space and requires correction.
Correcting the bottom mode using the measurement results {math}`m_1` and {math}`m_2` from mode 1 and 2 is referred to as feedforward.
This is applied as a displacement (defined in {ref}`sec:gate-displacement`) operation.
The corrected mode is the output of the teleportation circuit.

From Eqs. {math:numref}`eq:beamsplitter` and {math:numref}`eq:measurement`, it can be shown that the measured values {math}`m_1` and {math}`m_2` are the results of measuring the right-hand side of

```{math}
---
label: eq:teleportation-measured-values
---
\binom{m_1}{m_2}
\xleftarrow{\text{measure}}
\frac{1}{\sqrt{2}}\left[
\left(\begin{array}{ll}
\sin \theta_1 & \cos \theta_1 \\
\sin \theta_2 & \cos \theta_2
\end{array}\right)\binom{\hat{x}_{\text {in }}}{\hat{p}_{\text {in }}}+\left(\begin{array}{cc}
-\sin \theta_1 & -\cos \theta_1 \\
\sin \theta_2 & \cos \theta_2
\end{array}\right)\binom{\hat{x}_2}{\hat{p}_2}
\right].
```

Additionally, from Eq. {math:numref}`eq:beamsplitter`, there is a relation between mode 2 and the output mode before correction, given by

```{math}
---
label: eq:teleportation-mode2
---
\binom{\hat{x}_{\text {out }}}{\hat{p}_{\text {out }}}=\binom{\hat{x}_2}{-\hat{p}_2}+\hat{N}.
```

{math}`\hat{N}` is an operator corresponding to the imperfection of the EPR state and is determined by the original squeezing resource.
In the remainder of this page, this operator will be ignored.
Letting the input mode be {math}`(\hat{x}_\text{in}, \hat{p}_\text{in})=(\hat{x}_1, \hat{p}_1)`, the output mode before correction can be derived as

```{math}
---
label: eq:teleportation-circuit-output
---
\binom{\hat{x}_{\text {out }}}{\hat{p}_{\text {out }}}=
V\left(\theta_1, \theta_2\right) \binom{\hat{x}_{\text {in}}}{\hat{p}_{\text {in}}}-
\boldsymbol{f}_\text{tel} \left(\theta_1, \theta_2; m_1, m_2\right)
```

using Eqs. {math:numref}`eq:teleportation-measured-values` and {math:numref}`eq:teleportation-mode2`.
The coefficient matrix {math}`V\left(\theta_1, \theta_2\right)` is calculated as

```{math}
---
label: eq:teleportation-circuit
---
\begin{align}
V\left(\theta_1, \theta_2\right)&=
-\frac{1}{\sin \left(\theta_2-\theta_1\right)}\left(\begin{array}{cc}
\cos \theta_2 & \cos \theta_1 \\
\sin \theta_2 & \sin \theta_1
\end{array}\right)\left(\begin{array}{cc}
\sin \theta_1 & \cos \theta_1 \\
\sin \theta_2 & \cos \theta_2
\end{array}\right)\\
&=
R\left[\frac{\theta_1+\theta_2}{2}-\frac{\pi}{2}\right] S_\text{V}\left[\cot \left(\frac{\theta_1-\theta_2}{2}\right)\right] R\left[\frac{\theta_1+\theta_2}{2}\right].
\end{align}
```

using rotation {math}`\hat{R}` (defined in {ref}`sec:gate-rotation`) and squeezing {math}`\hat{S}_\text{V}` (defined in {ref}`sec:gate-squeezing`).
The second term

```{math}
---
label: eq:teleportation-ff
---
-\boldsymbol{f}_\text{tel} \left(\theta_1, \theta_2; m_1, m_2\right)=
\frac{\sqrt{2}}{\sin \left(\theta_2-\theta_1\right)}\left(\begin{array}{cc}
\cos \theta_2 & \cos \theta_1 \\
\sin \theta_2 & \sin \theta_1
\end{array}\right)\binom{m_1}{m_2}
```

is cancelled out by the feedforward {math}`\boldsymbol{f}_\text{tel} \left(\theta_1, \theta_2; m_1, m_2\right)`.
When the feedforward is applied, Eq. {math:numref}`eq:teleportation-circuit-output` finally becomes

```{math}
V\left(\theta_1, \theta_2\right) \binom{\hat{x}_{\text {in}}}{\hat{p}_{\text {in}}}.
```

(sec:macronode-circuit)=

## Macronode circuit

The circuit within macronode {math}`k`, as well as those connecting to macronodes {math}`k+1` and {math}`k+N`, can be represented as shown on the left in {numref}`fig:macronode_measurement_circuit`.
This can be transformed into the form on the right.
In this page, these are referred to as macronode measurement circuits.

```{figure} _images/macronode_measurement_circuit.svg
:name: fig:macronode_measurement_circuit
:alt: Macronode measurement circuits

Macronode measurement circuits. The diagram on the right is an equivalent transformation of the diagram on the left, with each yellow region representing a teleportation circuit.
```

````{admonition} Derivation
:class: hint

It can be easily verified through calculations that the foursplitter can be transformed as shown in {numref}`fig:foursplitter_equivalence`.

```{figure} _images/foursplitter_equivalence.svg
:name: fig:foursplitter_equivalence
:alt: Equivalent transformation of foursplitter

Equivalent transformation of foursplitter.
```

Additionally, as shown on the left in {numref}`fig:bs_equivalence`, a beam splitter can be replaced by an inverse beam splitter applied to the modes entangled with both input modes.
This is because, as shown on the right, both configurations produce the same square cluster state.

```{figure} _images/bs_equivalence.svg
:name: fig:bs_equivalence
:alt: Equivalence of beamsplitter for EPR states

Equivalence of beamsplitter for EPR states.
The left side illustrates the equivalence of the beam splitter operation, while the right side conceptually demonstrates its validity.
```

The left side of {numref}`fig:macronode_measurement_circuit` directly utilizes the foursplitter in the form shown on the right in {numref}`fig:foursplitter_equivalence`.
The transformation from the left to right in {numref}`fig:macronode_measurement_circuit` employs the relation depicted in {numref}`fig:bs_equivalence`.
````

The feedforward within the two teleportation circuits is {math}`\boldsymbol{f}_\text{tel} \left(\theta^k_{b}, \theta^k_{a}; m^k_{b}, m^k_{a}\right)` and {math}`\boldsymbol{f}_\text{tel} \left(\theta^k_{d}, \theta^k_{c}; m^k_{d}, m^k_{c}\right)`, respectively.
By applying Eq. {math:numref}`eq:teleportation-circuit-output`, the right side of {numref}`fig:macronode_measurement_circuit` is simplified into the form of {numref}`fig:macronode_circuit`. In this page, we refer to this as the macronode circuit.

```{figure} _images/ff_before_bs.svg
:name: fig:macronode_circuit
:alt: Macronode circuit

Macronode circuit.
```

(sec:gate-construction)=

## Gate construction

This section explains the basics of determining the measurement angles needed to realize a desired gate.
For now, feedforward is disregarded, as it will be addressed in {ref}`sec:feedforward`.

(sec:through-and-swap)=

### Through and swap

When {math}`\theta_a^k=\theta_c^k` and {math}`\theta_b^k=\theta_d^k` hold, it can be proven that the {math}`\hat{V}` part of the macronode circuit and the beam splitter commute.
As a result, the macronode circuit can be transformed as shown on the left in {numref}`fig:through_swap`, where the two modes input into macronode {math}`k` undergo identical single-mode operations without interacting with each other.
The result of the operation on mode {math}`b_k` is output to mode {math}`b_{k+1}`, and the result of the operation on mode {math}`d_k` is output to mode {math}`d_{k+1}`.
A macronode that satisfies these conditions is said to "pass *through*" the modes.

In contrast, if {math}`\theta_a^k=\theta_d^k` and {math}`\theta_b^k=\theta_c^k` are satisfied, applying {math}`\hat{V}(\theta_1, \theta_2)=\hat{V}(\theta_2, \theta_1)\hat{R}(\pi)` and the previously mentioned commutativity allows the macronode circuit to be transformed as shown on the right in {numref}`fig:through_swap`.
The two modes undergo identical operations independently, similar to the through case.
However, unlike the through case, the result of the operation on mode {math}`b` is output to mode {math}`d_{k+N}`, and the result of the operation on mode {math}`d` is output to mode {math}`b_{k+1}`.
Additionally, the operations applied to the two modes are {math}`\hat{V}\left(\theta_a^k, \theta_b^k\right)` instead of {math}`\hat{V}\left(\theta_b^k, \theta_a^k\right)`.
This type of macronode is said to "*swap*" the modes.

```{figure} _images/through_swap.svg
:name: fig:through_swap
:alt: Through and swap derivation

Through and swap derivation. In the macronode circuit shown in {numref}`fig:macronode_circuit`, when {math}`\theta_a^k=\theta_c^k` and {math}`\theta_b^k=\theta_d^k`, the input modes are passed through, whereas when {math}`\theta_a^k=\theta_d^k` and {math}`\theta_b^k=\theta_c^k`, they are swapped. The feedforward is ignored.
```

In summary, when performing the same single-mode operation on two input modes within a single macronode {math}`k`, setting {math}`\boldsymbol{\theta}^k=\left(\theta_a^k, \theta_b^k, \theta_a^k, \theta_b^k\right)` results in the modes being passed through.
If the goal is to swap the modes applying the same operations as in the through case, setting {math}`\boldsymbol{\theta}^k=\left(\theta_b^k, \theta_a^k, \theta_a^k, \theta_b^k\right)` will achieve this.
This property can also be extended to two-mode operations.
The transformation shown in {numref}`fig:general_swap` can be easily verified.
This means that operating with {math}`\boldsymbol{\theta}^k=\left(\theta_b^k, \theta_a^k, \theta_c^k, \theta_d^k\right)` is equivalent to operating with {math}`\boldsymbol{\theta}^k=\left(\theta_a^k, \theta_b^k, \theta_c^k, \theta_d^k\right)` followed by a swap.

```{figure} _images/general_swap.svg
:name: fig:general_swap
:alt: Two-mode swap

Two-mode swap.
It is shown that operating with {math}`\boldsymbol{\theta}^k=\left(\theta_b^k, \theta_a^k, \theta_c^k, \theta_d^k\right)` is equivalent to operating with {math}`\boldsymbol{\theta}^k=\left(\theta_a^k, \theta_b^k, \theta_c^k, \theta_d^k\right)` followed by a swap.
```

(sec:angle-equivalence-2pi-pi)=

### Equivalence of measurement angles modulo {math}`2\pi`

Consider transforming the measurement angles {math}`\boldsymbol{\theta}^k=\left(\theta_a^k, \theta_b^k, \theta_c^k, \theta_d^k\right)` into

```{math}
\begin{array}{c}
\left(\theta_a^k+n_a\pi, \theta_b^k+n_b\pi, \theta_c^k+n_c\pi, \theta_d^k+n_d\pi\right). \\
\left(n_a, n_b, n_c, n_d\in\mathbb{Z}\right)
\end{array}
```

When {math}`n_a`, {math}`n_b`, {math}`n_c`, and {math}`n_d` are all even, the operations applied to the modes into macronode {math}`k` (including {ref}`sec:displacement`), feedforward, and measured values remain invariant under this transformation.
In other words, measurement angles congruent modulo {math}`2\pi` can be considered equivalent.

On the other hand, if any of {math}`n_a`, {math}`n_b`, {math}`n_c`, or {math}`n_d` is odd, as can be easily verified from Eqs. {math:numref}`eq:teleportation-circuit` and {math:numref}`eq:teleportation-ff`, the operations and feedforward remain unchanged, but the measured values are altered from {math}`\left(m_a^k, m_b^k, m_c^k, m_d^k\right)` to

```{math}
\begin{array}{c}
\left(\sigma_a m_a^k, \sigma_b m_b^k, \sigma_c m_c^k, \sigma_d m_d^k\right). \\
\left(\sigma_i =
\begin{cases}
1 & \text{if } n_i \text{ is odd}, \\
-1 & \text{if } n_i \text{ is even}.
\end{cases}\right)
\end{array}
```

Users must not equate measurement angles congruent modulo {math}`\pi` simply because the mode remains unchanged; the measured values differ.

(sec:measurement-and-initialization)=

### Measurement and initialization

When {math}`\theta_a\equiv\theta_b` or {math}`\theta_c\equiv\theta_d` (mod {math}`\pi`), Eqs. {math:numref}`eq:teleportation-circuit` and {math:numref}`eq:teleportation-ff` diverge, meaning that setting the measurement angles this way within a macronode is generally not allowed.
However, in the specific case where {math}`\theta_a\equiv\theta_b\equiv\theta_c\equiv\theta_d` (mod {math}`2\pi`), the macronode can be treated as the final measurement of the modes propagated through the quantum circuit and the generation of new modes.
Since the input modes to macronode {math}`k` are subjected to the foursplitter {math}`\hat{A}` before being measured, the measured values {math}`\boldsymbol{m}^k=\left(m_a^k, m_b^k, m_c^k, m_d^k\right)` differ from the values originally intended to be measured.
By applying the inverse foursplitter as

```{math}
---
label: eq:readout-measured-values
---
\boldsymbol{M}^k=
\left(\begin{array}{l}
M_a^k \\
M_b^k \\
M_c^k \\
M_d^k
\end{array}\right)=\frac{1}{2}\left(\begin{array}{cccc}
1 & -1 & 1 & -1 \\
1 & 1 & 1 & 1 \\
-1 & 1 & 1 & -1 \\
-1 & -1 & 1 & 1
\end{array}\right)\left(\begin{array}{l}
m_a^k \\
m_b^k \\
m_c^k \\
m_d^k
\end{array}\right),
```

the desired measured values for the modes input into macronode {math}`k` are obtained.
Here, {math}`M_b^k` and {math}`M_d^k` are the measured values of modes {math}`b_k` and {math}`d_k`, i.e., {math}`\left(x_b^k, p_b^k\right)` and {math}`\left(x_d^k, p_d^k\right)`, respectively.
In contrast, as described in {ref}`sec:teleportation-circuit` and {ref}`sec:macronode-circuit`, modes {math}`a_k` and {math}`c_k` serve merely as teleportation media, so {math}`M_a^k` and {math}`M_c^k` have no significance in quantum computation.
In MQC3, for any type of macronode {math}`k`, not limited to measurement macronode, the output result is {math}`\boldsymbol{M}^k` rather than {math}`\boldsymbol{m}^k`.

The output modes of measurement macronodes are squeezed states that do not retain the information of the input modes.
Thus, measurement macronodes can also be regarded as macronodes that reinitialize the states and are referred to as initialization macronodes.
In MQC3, all modes in quantum computation are treated as being generated from initialization macronodes.
In an initialization macronode {math}`k`, the squeezing level of the output mode cannot be set, but the squeezing angle can be configured.
Here squeezing angle {math}`\vartheta^k` refers to the squeezing in the direction of {math}`\hat{x}\sin{\vartheta^k}+\hat{p}\cos{\vartheta^k}`, which can be achieved simply by setting {math}`\theta_a^k\equiv\theta_b^k\equiv\theta_c^k\equiv\theta_d^k\equiv -\vartheta^k` (mod {math}`2\pi`).
The feedforward applied to the initialized modes is formulated in {ref}`sec:initialization`.

Due to the constraint between the measurement angles of the modes being measured and the squeezing angles of the modes being generated, it is uncommon for a single macronode to simultaneously perform both the desired measurement and initialization.
Typically, the roles of each macronode are clearly distinguished, such as measurement macronodes and initialization macronodes.

### List of operations

The main classification of gates handled in MQC3 consists of single-mode gates and two-mode gates.
Not all single-mode and two-mode gates can be implemented with a single macronode, and some may require multiple macronodes.
Any arbitrary single-mode Gaussian gate can be realized with at most two macronodes.
The conditions that a general single-mode gate using two macronodes must satisfy are described in {ref}`sec:general-single-mode-gates`.
Special types of operations are wiring, measurement, and initialization.
Wiring involve a macronode allowing the modes to traverse without altering the quantum states.
The meanings of measurement and initialization are as described in {ref}`sec:measurement-and-initialization`.

{numref}`tab:operations-angles` shows the set of operations supported in the graph representation of MQC3.
In this table, all measurement angles are set to allow the modes to pass through.
If a swap is desired, as described in {ref}`sec:through-and-swap`, simply exchange {math}`\theta_a` and {math}`\theta_b`.

```{seealso}
The meaning and definition of each operation symbol are listed in {ref}`sec:gates`.
```

```{table} Measurement angles to realize operations. In this table, all measurement angles are set to allow the modes to pass through. "#" represents the number of macronodes.
:name: tab:operations-angles

| Category | Name | Operation | # | Measurement angles {math}`\boldsymbol{\theta}` |
| :---: | :---: | :---: | :---: | :--- |
| Single-mode | Phase rotation | {math}`R(\phi)` | 1 | {math}`\left(\frac{\phi}{2}, \frac{\phi}{2}+\frac{\pi}{2}, \frac{\phi}{2}, \frac{\phi}{2}+\frac{\pi}{2}\right)` |
| Single-mode | Shear (X-invariant) | {math}`P(\kappa)` | 1 | {math}`\left(\arctan \kappa, \frac{\pi}{2}, \arctan \kappa, \frac{\pi}{2}\right)` |
| Single-mode | Shear (P-invariant) | {math}`Q(\eta)` | 1 | {math}`\left(0, \operatorname{arccot}\eta, 0, \operatorname{arccot}\eta\right)` |
| Single-mode | Squeezing (+ Fourier) | {math}`R\left(-\frac{\pi}{2}\right)S_\text{V}(\cot\theta)` | 1 | {math}`\left(-\theta, \theta,-\theta, \theta\right)` |
| Single-mode | 45{math}`^\circ` squeezing | {math}`R\left(-\frac{\pi}{4}\right)S_\text{V}(\cot\theta)R\left(\frac{\pi}{4}\right)` | 1 | {math}`\left(\frac{\pi}{4}-\theta, \frac{\pi}{4}+\theta, \frac{\pi}{4}-\theta, \frac{\pi}{4}+\theta\right)` |
| Single-mode | Arbitrary | {math}`R(\alpha) S(\lambda) R(\beta)` | 2 | {math}`\left(\beta-\arctan e^{-\lambda}, \beta+\arctan e^{-\lambda}, \beta-\arctan e^{-\lambda}, \beta+\arctan e^{-\lambda}\right)`<br>{math}`\left(\frac{\alpha-\beta}{2}+\frac{\pi}{4}, \frac{\alpha-\beta}{2}-\frac{\pi}{4}, \frac{\alpha-\beta}{2}+\frac{\pi}{4}, \frac{\alpha-\beta}{2}-\frac{\pi}{4}\right)` |
| Two-mode | Controlled-Z | {math}`C_Z(g)` | 1 | {math}`\left(-\arctan \left(\frac{g}{2}\right), \frac{\pi}{2},\arctan \left(\frac{g}{2}\right), \frac{\pi}{2}\right)` |
| Two-mode | Beamsplitter interaction | {math}`B\left(\sqrt{R}, \theta_\text{rel}\right)` | 1 | {math}`\left(\frac{\theta_{\text{rel}}}{2}+\frac{\arccos (\sqrt{R})}{2}, \frac{\theta_{\text{rel}}}{2}+\frac{\arccos (\sqrt{R})}{2}+\frac{\pi}{2}, \frac{\theta_{\text{rel}}}{2}-\frac{\arccos (\sqrt{R})}{2}, \frac{\theta_{\text{rel}}}{2}-\frac{\arccos (\sqrt{R})}{2}+\frac{\pi}{2}\right)` |
| Two-mode | Two-mode shear | {math}`P_2(a, b)` | 1 | {math}`\left(\arctan \left(a-\frac{b}{2}\right), \frac{\pi}{2}, \arctan \left(a+\frac{b}{2}\right), \frac{\pi}{2}\right)` |
| Two-mode | Manual | {numref}`fig:macronode_circuit` without feedforward | 1 or 2 | {math}`\left(\theta_a, \theta_b, \theta_c, \theta_d\right)` |
| Utility | Wiring | Identity | 1 | {math}`\left(0, \frac{\pi}{2}, 0, \frac{\pi}{2}\right)` |
| Utility | Measurement | {math}`M(\theta)` | 1 | {math}`\left(\theta, \theta, \theta, \theta\right)` |
| Utility | Initialization | {math}`I(\theta)` | 1 | {math}`\left(\theta, \theta, \theta, \theta\right)` |
```

(sec:feedforward)=

## Feedforward

This section presents the forms of the feedforward when it is propagated to later stages rather than performed within the teleportation circuits.
In the machinery handled by MQC3, feedforward is applied based on the measured values.
The form of the feedforward differs between measurement/initialization macronodes and other macronodes.
First, the feedforward for standard macronodes is explained, followed by the forms for measurement and initialization macronodes.

(sec:feedforward-macronode)=

### Feedforward within macronode circuit

The feedforward in {numref}`fig:macronode_circuit` can be delayed to be applied after the inverse beamsplitter as illustrated in {numref}`fig:ff_after_bs`.

```{figure} _images/ff_after_bs.svg
:name: fig:ff_after_bs
:alt: Feedforward after beamsplitter

Feedforward after beamsplitter in a macronode circuit. {math}`\boldsymbol{f}^{+1}_\text{mac}` represents the feedforward for mode {math}`b_{k+1}`, while {math}`\boldsymbol{f}^{+N}_\text{mac}` represents the feedforward for mode {math}`d_{k+N}`.
```

When the feedforward propagated to macronodes {math}`k+1` and {math}`k+N` is denoted as {math}`\boldsymbol{f}^{+1}_\text{mac} \left(\boldsymbol{\theta}^k, \boldsymbol{m}^k\right)` and {math}`\boldsymbol{f}^{+N}_\text{mac} \left(\boldsymbol{\theta}^k, \boldsymbol{m}^k\right)`, respectively, the following relation holds.

```{math}
---
label: eq:macronode-ff
---
\begin{align}
\left(\begin{array}{l}
\mathcal{X}_k^{+1} \\
\mathcal{X}_k^{+N} \\
\mathcal{P}_k^{+1} \\
\mathcal{P}_k^{+N}
\end{array}\right)
&:=
\left(\begin{array}{l}
\left[ \boldsymbol{f}^{+1}_\text{mac} \left(\boldsymbol{\theta}^k, \boldsymbol{m}^k\right) \right]_x \\
\left[ \boldsymbol{f}^{+N}_\text{mac} \left(\boldsymbol{\theta}^k, \boldsymbol{m}^k\right) \right]_x \\
\left[ \boldsymbol{f}^{+1}_\text{mac} \left(\boldsymbol{\theta}^k, \boldsymbol{m}^k\right) \right]_p \\
\left[ \boldsymbol{f}^{+N}_\text{mac} \left(\boldsymbol{\theta}^k, \boldsymbol{m}^k\right) \right]_p
\end{array}\right)
=
B_\uparrow \left(\begin{array}{l}
\left[ \boldsymbol{f}_\text{tel} \left(\theta^k_{b}, \theta^k_{a}; m^k_{b}, m^k_{a}\right) \right]_x \\
\left[ \boldsymbol{f}_\text{tel} \left(\theta^k_{d}, \theta^k_{c}; m^k_{d}, m^k_{c}\right) \right]_x \\
\left[ \boldsymbol{f}_\text{tel} \left(\theta^k_{b}, \theta^k_{a}; m^k_{b}, m^k_{a}\right) \right]_p \\
\left[ \boldsymbol{f}_\text{tel} \left(\theta^k_{d}, \theta^k_{c}; m^k_{d}, m^k_{c}\right) \right]_p
\end{array}\right) \\
&=
-\left(\begin{array}{c}
\frac{m_a^k \cos{\theta^k_b}+m_b^k \cos{\theta^k_a}}{\sin \left(\theta^k_a-\theta^k_b\right)}+\frac{m_c^k \cos{\theta^k_d}+m_d^k \cos{\theta^k_c}}{\sin \left(\theta^k_c-\theta^k_d\right)} \\
-\frac{m_a^k \cos{\theta^k_b}+m_b^k \cos{\theta^k_a}}{\sin \left(\theta^k_a-\theta^k_b\right)}+\frac{m_c^k \cos{\theta^k_d}+m_d^k \cos{\theta^k_c}}{\sin \left(\theta^k_c-\theta^k_d\right)} \\
\frac{m_a^k \sin{\theta^k_b}+m_b^k \sin{\theta^k_a}}{\sin \left(\theta^k_a-\theta^k_b\right)}+\frac{m_c^k \sin{\theta^k_d}+m_d^k \sin{\theta^k_c}}{\sin \left(\theta^k_c-\theta^k_d\right)} \\
-\frac{m_a^k \sin{\theta^k_b}+m_b^k \sin{\theta^k_a}}{\sin \left(\theta^k_a-\theta^k_b\right)}+\frac{m_c^k \sin{\theta^k_d}+m_d^k \sin{\theta^k_c}}{\sin \left(\theta^k_c-\theta^k_d\right)}
\end{array}\right)
\end{align}
```

```{admonition} Derivation
:class: hint

Eq. {math:numref}`eq:macronode-ff` is derived in {ref}`sec:macronode-ff-derivation`.
```

(sec:numerical-feedforward)=

### Numerical feedforward

When exclusively dealing with Gaussian states, feedforward can be implemented by performing operations on the measurement results after all macronode measurements are completed.
This process is equivalent to the feedforward applied to quantum states, which is discussed in {ref}`sec:teleportation-circuit` and {ref}`sec:feedforward-macronode`.
This is referred to as numerical feedforward.
The machinery used in MQC3 adopts this method.

{math}`\boldsymbol{f}^{+1}_\text{mac} \left(\boldsymbol{\theta}^k, \boldsymbol{m}^k\right)` propagates to mode {math}`b_{k+1}`.
The feedforward propagated to the initial state of macronode {math}`k+1`, {math}`(\hat{x}^{k+1}_a, \hat{x}^{k+1}_b, \hat{x}^{k+1}_c, \hat{x}^{k+1}_d, \hat{p}^{k+1}_a, \hat{p}^{k+1}_b, \hat{p}^{k+1}_c, \hat{p}^{k+1}_d)`, is

```{math}
\left(\begin{array}{l}
0 \\
\mathcal{X}_k^{+1} \\
0 \\
0 \\
0 \\
\mathcal{P}_k^{+1} \\
0 \\
0
\end{array}\right).
```

By applying a foursplitter {math}`\hat{A}` to the initial state of the macronode and measuring it, the measured values of the macronode can be obtained.
Similarly, by performing the same operation on the propagated feedforward, the feedforward applied to the measured values {math}`\left(m^{k+1}_a, m^{k+1}_b, m^{k+1}_c, m^{k+1}_d\right)` can be determined.

```{math}
\begin{align}
\boldsymbol{f}^{+1}_\text{num} \left(\boldsymbol{\theta}^k, \boldsymbol{m}^k, \boldsymbol{\theta}^{k+1}\right)
&=
\left(\begin{array}{cccccccc}
\sin{\theta^{k+1}_a} &  &  &  & \cos{\theta^{k+1}_a} &  &  &  \\
& \sin{\theta^{k+1}_b} &  &  &  & \cos{\theta^{k+1}_b} &  &  \\
&  & \sin{\theta^{k+1}_c} &  &  &  & \cos{\theta^{k+1}_c} &  \\
&  &  & \sin{\theta^{k+1}_d} &  &  &  & \cos{\theta^{k+1}_d}
\end{array}\right)
A
\left(\begin{array}{l}
0 \\
\mathcal{X}_k^{+1} \\
0 \\
0 \\
0 \\
\mathcal{P}_k^{+1} \\
0 \\
0
\end{array}\right)\\
&=
-\frac{1}{2}
\left(\begin{array}{llll}
\frac{\sin \left(\theta_a^{k+1}+\theta_b^k\right)}{\sin \left(\theta_a^k-\theta_b^k\right)} & \frac{\sin \left(\theta_a^k+\theta_a^{k+1}\right)}{\sin \left(\theta_a^k-\theta_b^k\right)} & \frac{\sin \left(\theta_a^{k+1}+\theta_d^k\right)}{\sin \left(\theta_c^k-\theta_d^k\right)} & \frac{\sin \left(\theta_a^{k+1}+\theta_c^k\right)}{\sin \left(\theta_c^k-\theta_d^k\right)} \\
\frac{\sin \left(\theta_b^k+\theta_b^{k+1}\right)}{\sin \left(\theta_a^k-\theta_b^k\right)} & \frac{\sin \left(\theta_a^k+\theta_b^{k+1}\right)}{\sin \left(\theta_a^k-\theta_b^k\right)} & \frac{\sin \left(\theta_b^{k+1}+\theta_d^k\right)}{\sin \left(\theta_c^k-\theta_d^k\right)} & \frac{\sin \left(\theta_b^{k+1}+\theta_c^k\right)}{\sin \left(\theta_c^k-\theta_d^k\right)} \\
\frac{\sin \left(\theta_b^k+\theta_c^{k+1}\right)}{\sin \left(\theta_a^k-\theta_b^k\right)} & \frac{\sin \left(\theta_a^k+\theta_c^{k+1}\right)}{\sin \left(\theta_a^k-\theta_b^k\right)} & \frac{\sin \left(\theta_c^{k+1}+\theta_d^k\right)}{\sin \left(\theta_c^k-\theta_d^k\right)} & \frac{\sin \left(\theta_c^k+\theta_c^{k+1}\right)}{\sin \left(\theta_c^k-\theta_d^k\right)} \\
\frac{\sin \left(\theta_b^k+\theta_d^{k+1}\right)}{\sin \left(\theta_a^k-\theta_b^k\right)} & \frac{\sin \left(\theta_a^k+\theta_d^{k+1}\right)}{\sin \left(\theta_a^k-\theta_b^k\right)} & \frac{\sin \left(\theta_d^k+\theta_d^{k+1}\right)}{\sin \left(\theta_c^k-\theta_d^k\right)} & \frac{\sin \left(\theta_c^k+\theta_d^{k+1}\right)}{\sin \left(\theta_c^k-\theta_d^k\right)}
\end{array}\right)
\left(\begin{array}{l}
m_a^k \\
m_b^k \\
m_c^k \\
m_d^k
\end{array}\right)
\end{align}
```

In a similar manner, considering that {math}`\boldsymbol{f}^{+N}_\text{mac} \left(\boldsymbol{\theta}^k, \boldsymbol{m}^k\right)` propagates to mode {math}`d_{k+N}`, the feedforward applied to the measured values {math}`\left(m^{k+N}_a, m^{k+N}_b, m^{k+N}_c, m^{k+N}_d\right)` can be determined.

```{math}
\boldsymbol{f}^{+N}_\text{num} \left(\boldsymbol{\theta}^k, \boldsymbol{m}^k, \boldsymbol{\theta}^{k+N}\right)
=
-\frac{1}{2}
\left(\begin{array}{cccc}
\frac{\sin \left(\theta_a^{k+N}+\theta_b^k\right)}{\sin \left(\theta_a^k-\theta_b^k\right)} & \frac{\sin \left(\theta_a^k+\theta_a^{k+N}\right)}{\sin \left(\theta_a^k-\theta_b^k\right)} & -\frac{\sin \left(\theta_a^{k+N}+\theta_d^k\right)}{\sin \left(\theta_c^k-\theta_d^k\right)} & -\frac{\sin \left(\theta_a^{k+N}+\theta_c^k\right)}{\sin \left(\theta_c^k-\theta_d^k\right)} \\
\frac{\sin \left(\theta_b^k+\theta_b^{k+N}\right)}{\sin \left(\theta_a^k-\theta_b^k\right)} & \frac{\sin \left(\theta_a^k+\theta_b^{k+N}\right)}{\sin \left(\theta_a^k-\theta_b^k\right)} & -\frac{\sin \left(\theta_b^{k+N}+\theta_d^k\right)}{\sin \left(\theta_c^k-\theta_d^k\right)} & -\frac{\sin \left(\theta_b^{k+N}+\theta_c^k\right)}{\sin \left(\theta_c^k-\theta_d^k\right)} \\
-\frac{\sin \left(\theta_b^k+\theta_c^{k+N}\right)}{\sin \left(\theta_a^k-\theta_b^k\right)} & -\frac{\sin \left(\theta_a^k+\theta_c^{k+N}\right)}{\sin \left(\theta_a^k-\theta_b^k\right)} & \frac{\sin \left(\theta_c^{k+N}+\theta_d^k\right)}{\sin \left(\theta_c^k-\theta_d^k\right)} & \frac{\sin \left(\theta_c^k+\theta_c^{k+N}\right)}{\sin \left(\theta_c^k-\theta_d^k\right)} \\
-\frac{\sin \left(\theta_b^k+\theta_d^{k+N}\right)}{\sin \left(\theta_a^k-\theta_b^k\right)} & -\frac{\sin \left(\theta_a^k+\theta_d^{k+N}\right)}{\sin \left(\theta_a^k-\theta_b^k\right)} & \frac{\sin \left(\theta_d^k+\theta_d^{k+N}\right)}{\sin \left(\theta_c^k-\theta_d^k\right)} & \frac{\sin \left(\theta_c^k+\theta_d^{k+N}\right)}{\sin \left(\theta_c^k-\theta_d^k\right)}
\end{array}\right)
\left(\begin{array}{l}
m_a^k \\
m_b^k \\
m_c^k \\
m_d^k
\end{array}\right)
```

```{admonition} Derivation
:class: hint

{math}`\boldsymbol{f}^{+1}_\text{num} \left(\boldsymbol{\theta}^k, \boldsymbol{m}^k, \boldsymbol{\theta}^{k+1}\right)` and {math}`\boldsymbol{f}^{+N}_\text{num} \left(\boldsymbol{\theta}^k, \boldsymbol{m}^k, \boldsymbol{\theta}^{k+N}\right)` are derived in {ref}`sec:numerical-ff-derivation`.
```

The feedforward for the measured values is achieved by performing the following additions sequentially, starting from the smallest {math}`k` and proceeding in order.

```{math}
\begin{align}
\boldsymbol{m}^{k+1} &+= \boldsymbol{f}^{+1}_\text{num} \left(\boldsymbol{\theta}^k, \boldsymbol{m}^k, \boldsymbol{\theta}^{k+1}\right) \\
\boldsymbol{m}^{k+N} &+= \boldsymbol{f}^{+N}_\text{num} \left(\boldsymbol{\theta}^k, \boldsymbol{m}^k, \boldsymbol{\theta}^{k+N}\right)
\end{align}
```

(sec:initialization)=

### Feedforward immediately after initialization

The feedforward immediately after initialization is implemented by rewriting the standard feedforward in Eq. {math:numref}`eq:macronode-ff` as

```{math}
---
label: eq:initialization-macronode-ff
---
\left(\begin{array}{l}
\mathcal{X}_k^{+1} \\
\mathcal{P}_k^{+1} \\
\mathcal{X}_k^{+N} \\
\mathcal{P}_k^{+N}
\end{array}\right)
\rightarrow
\left(\begin{array}{l}
\mathscr{X}_k^{+1} \left(\theta_a^k, \boldsymbol{m}^k \right) \\
\mathscr{P}_k^{+1} \left(\theta_a^k, \boldsymbol{m}^k \right) \\
\mathscr{X}_k^{+N} \left(\theta_c^k, \boldsymbol{m}^k \right) \\
\mathscr{P}_k^{+N} \left(\theta_c^k, \boldsymbol{m}^k \right)
\end{array}\right)
:=
\left(\begin{array}{l}
-M_a^k \sin{\theta_a^k} \\
 M_a^k \cos{\theta_a^k} \\
-M_c^k \sin{\theta_c^k} \\
 M_c^k \cos{\theta_c^k}
\end{array}\right).
```

As explained in {ref}`sec:displacement-in-initialization`, it is possible to apply a displacement simultaneously with initialization, but since the displacements are treated as gates and not included in the initialization configuration, as described in {ref}`sec:displacement`, this will not be done.
When the feedforward values in Eq. {math:numref}`eq:initialization-macronode-ff` are propagated to the measured values, they take the following form.

```{math}
---
label: eq:initialization-numerical-ff
---
\begin{align}
\boldsymbol{i}^{+1}_\text{num} \left(\theta_a^k, \boldsymbol{m}^k, \boldsymbol{\theta}^{k+1}\right)
&=
\frac{1}{4}
\left(\begin{array}{l}
\left(m_a^k-m_b^k+m_c^k-m_d^k\right) \cos \left(\theta_a^k+\theta_a^{k+1}\right) \\
\left(m_a^k-m_b^k+m_c^k-m_d^k\right) \cos \left(\theta_a^k+\theta_b^{k+1}\right) \\
\left(m_a^k-m_b^k+m_c^k-m_d^k\right) \cos \left(\theta_a^k+\theta_c^{k+1}\right) \\
\left(m_a^k-m_b^k+m_c^k-m_d^k\right) \cos \left(\theta_a^k+\theta_d^{k+1}\right)
\end{array}\right) \\
\boldsymbol{i}^{+N}_\text{num} \left(\theta_c^k, \boldsymbol{m}^k, \boldsymbol{\theta}^{k+N}\right)
&=
\frac{1}{4}
\left(\begin{array}{l}
\left( m_a^k-m_b^k-m_c^k+m_d^k\right) \cos \left(\theta_c^k+\theta_a^{k+N}\right) \\
\left( m_a^k-m_b^k-m_c^k+m_d^k\right) \cos \left(\theta_c^k+\theta_b^{k+N}\right) \\
\left(-m_a^k+m_b^k+m_c^k-m_d^k\right) \cos \left(\theta_c^k+\theta_c^{k+N}\right) \\
\left(-m_a^k+m_b^k+m_c^k-m_d^k\right) \cos \left(\theta_c^k+\theta_d^{k+N}\right)
\end{array}\right)
\end{align}
```

```{admonition} Derivation
:class: hint

Eq. {math:numref}`eq:initialization-numerical-ff` is derived in {ref}`sec:initialization-ff-derivation`.
```

The feedforward equations for the measured values are modified as follows.

```{math}
\begin{align}
\boldsymbol{m}^{k+1} &+= \boldsymbol{i}^{+1}_\text{num} \left(\theta_a^k, \boldsymbol{m}^k, \boldsymbol{\theta}^{k+1}\right) \\
\boldsymbol{m}^{k+N} &+= \boldsymbol{i}^{+N}_\text{num} \left(\theta_c^k, \boldsymbol{m}^k, \boldsymbol{\theta}^{k+N}\right)
\end{align}
```

(sec:displacement)=

## Displacement

The displacement operations are not applied to the quantum states themselves, but rather implemented by correcting the measured values, using the same method as numerical feedforward described in {ref}`sec:numerical-feedforward`.
To add displacement {math}`\left(\mathbb{X}_k^{+1}, \mathbb{P}_k^{+1}\right)` to the mode output from macronode {math}`k` and input into macronode {math}`k+1`, the feedforward should be modified as

```{math}
\left(\begin{array}{l}
\mathcal{X}_k^{+1} \\
\mathcal{P}_k^{+1}
\end{array}\right)
\rightarrow
\left(\begin{array}{l}
\mathcal{X}_k^{+1} + \mathbb{X}_k^{+1} \\
\mathcal{P}_k^{+1} + \mathbb{P}_k^{+1}
\end{array}\right).
```

Similarly, the displacement {math}`\left(\mathbb{X}_k^{+N}, \mathbb{P}_k^{+N}\right)` for the mode propagating from macronode {math}`k` to {math}`k+N` revises the feedforward as

```{math}
\left(\begin{array}{l}
\mathcal{X}_k^{+N} \\
\mathcal{P}_k^{+N}
\end{array}\right)
\rightarrow
\left(\begin{array}{l}
\mathcal{X}_k^{+N} + \mathbb{X}_k^{+N} \\
\mathcal{P}_k^{+N} + \mathbb{P}_k^{+N}
\end{array}\right).
```

When the displacements {math}`\left(\mathbb{X}_k^{+1}, \mathbb{P}_k^{+1}\right)` and {math}`\left(\mathbb{X}_k^{+N}, \mathbb{P}_k^{+N}\right)` are propagated to the measured values, they are expressed as follows.

```{math}
---
label: eq:numerical-displacement
---
\begin{align}
\boldsymbol{d}^{+1}_\text{num} \left(\mathbb{X}_k^{+1}, \mathbb{P}_k^{+1}, \boldsymbol{\theta}^{k+1}\right)
&=
\frac{1}{2}
\left(\begin{array}{l}
{\mathbb{X}_k^{+1} \sin{\theta_a^{k+1}}} + {\mathbb{P}_k^{+1} \cos{\theta_a^{k+1}}} \\
{\mathbb{X}_k^{+1} \sin{\theta_b^{k+1}}} + {\mathbb{P}_k^{+1} \cos{\theta_b^{k+1}}} \\
{\mathbb{X}_k^{+1} \sin{\theta_c^{k+1}}} + {\mathbb{P}_k^{+1} \cos{\theta_c^{k+1}}} \\
{\mathbb{X}_k^{+1} \sin{\theta_d^{k+1}}} + {\mathbb{P}_k^{+1} \cos{\theta_d^{k+1}}}
\end{array}\right) \\
\boldsymbol{d}^{+N}_\text{num} \left(\mathbb{X}_k^{+N}, \mathbb{P}_k^{+N}, \boldsymbol{\theta}^{k+N}\right)
&=
\frac{1}{2}
\left(\begin{array}{r}
-{\mathbb{X}_k^{+N} \sin{\theta_a^{k+N}}} -{\mathbb{P}_k^{+N} \cos{\theta_a^{k+N}}} \\
-{\mathbb{X}_k^{+N} \sin{\theta_b^{k+N}}} -{\mathbb{P}_k^{+N} \cos{\theta_b^{k+N}}} \\
{\mathbb{X}_k^{+N} \sin{\theta_c^{k+N}}} + {\mathbb{P}_k^{+N} \cos{\theta_c^{k+N}}} \\
{\mathbb{X}_k^{+N} \sin{\theta_d^{k+N}}} + {\mathbb{P}_k^{+N} \cos{\theta_d^{k+N}}}
\end{array}\right)
\end{align}.
```

```{admonition} Derivation
:class: hint

Eq. {math:numref}`eq:numerical-displacement` is derived in {ref}`sec:displacement-derivation`.
```

Using these, the numerical feedforward equations are modified to

```{math}
\begin{align}
\boldsymbol{m}^{k+1} &+= \boldsymbol{f}^{+1}_\text{num} \left(\boldsymbol{\theta}^k, \boldsymbol{m}^k, \boldsymbol{\theta}^{k+1}\right) + \boldsymbol{d}^{+1}_\text{num} \left(\mathbb{X}_k^{+1}, \mathbb{P}_k^{+1}, \boldsymbol{\theta}^{k+1}\right) \\
\boldsymbol{m}^{k+N} &+= \boldsymbol{f}^{+N}_\text{num} \left(\boldsymbol{\theta}^k, \boldsymbol{m}^k, \boldsymbol{\theta}^{k+N}\right) + \boldsymbol{d}^{+N}_\text{num} \left(\mathbb{X}_k^{+N}, \mathbb{P}_k^{+N}, \boldsymbol{\theta}^{k+N}\right)
\end{align}
```

for cases other than immediately after initialization, and to

```{math}
\begin{align}
\boldsymbol{m}^{k+1} &+= \boldsymbol{i}^{+1}_\text{num} \left(\theta_a^k, \boldsymbol{m}^k, \boldsymbol{\theta}^{k+1}\right) + \boldsymbol{d}^{+1}_\text{num} \left(\mathbb{X}_k^{+1}, \mathbb{P}_k^{+1}, \boldsymbol{\theta}^{k+1}\right) \\
\boldsymbol{m}^{k+N} &+= \boldsymbol{i}^{+N}_\text{num} \left(\theta_c^k, \boldsymbol{m}^k, \boldsymbol{\theta}^{k+N}\right) + \boldsymbol{d}^{+N}_\text{num} \left(\mathbb{X}_k^{+N}, \mathbb{P}_k^{+N}, \boldsymbol{\theta}^{k+N}\right)
\end{align}
```

for the case immediately after initialization.

## Appendix

(sec:general-single-mode-gates)=

### General single mode gates

Any arbitrary single-mode Gaussian gate can be decomposed into rotations and squeezing as

```{math}
---
label: eq:general-1mode
---
R\left(\alpha\right)S\left(\lambda\right)R\left(\beta\right).
```

Following this decomposition, although there is ambiguity in the exact homodyne detectors angles for the execution on the cluster state, we only need at most two macronode to implement this.
To perform a single-mode operation, the two inputs of each macronode must not interact with each other, requiring the use of measurement angles that realize either the through or swap conditions.
The discussion here will focus on the through case, particularly on single-mode operations that generally involves two macronodes, {math}`k` and {math}`k+N`.
This does not result in a loss of generality, as the two input modes of the macronode undergo the same operation, whether in the case of passing through or swapping, as stated in {ref}`sec:through-and-swap`.
In this case, if the measurement angles satisfy the following condition, the mode {math}`d_k` will undergo the single-mode operation described by Eq. {math:numref}`eq:general-1mode`.

```{math}
---
label: eq:general-1mode-through-N
---
R\left(\alpha\right)S\left(\lambda\right)R\left(\beta\right) = V\left(\theta_b^k, \theta_a^k\right) V\left(\theta_b^{k+N}, \theta_a^{k+N}\right)
```

Here, {math}`\theta_a^k=\theta_c^k`, {math}`\theta_b^k=\theta_d^k`, {math}`\theta_a^{k+N}=\theta_c^{k+N}`, and {math}`\theta_b^{k+N}=\theta_d^{k+N}`.
For example, three parameters {math}`\alpha`, {math}`\beta`, and {math}`\lambda` that satisfy the following constraints meet Eq. {math:numref}`eq:general-1mode-through-N`.

```{math}
---
label: eq:general-1mode-condition
---
\begin{aligned}
\beta & =\frac{1}{2}\left(\theta_{b}^k+\theta_{a}^k\right) \\
e^\lambda & =\cot \left(\frac{\theta_{b}^k-\theta_{a}^k}{2}\right) \\
\alpha & =\theta_{b}^{k+N}+\theta_{a}^{k+N}+\frac{\theta_{b}^k+\theta_{a}^k}{2}-\pi \\
\frac{\pi}{2} & =\theta_{b}^{k+N}-\theta_{a}^{k+N}
\end{aligned}
```

By solving these, the measurement angles that implement the operation in Eq. {math:numref}`eq:general-1mode` can be determined.

```{seealso}
The proof can be found in {ref}`sec:graph-arbitrary-angles-validation`.
```

(sec:displacement-in-initialization)=

### Displacements in initialization

The displacements can be configured in initialization macronode.
When the displacements of the output modes from initialization macronode {math}`k` to {math}`k+1` and {math}`k+N` are denoted as {math}`\left(d_{k,x}^{+1}, d_{k,p}^{+1}\right)` and {math}`\left(d_{k,x}^{+N}, d_{k,p}^{+N}\right)`, respectively, this configuration is achieved by rewriting the feedforward as follows.

```{math}
\left(\begin{array}{l}
\mathcal{X}_k^{+1} \\
\mathcal{P}_k^{+1} \\
\mathcal{X}_k^{+N} \\
\mathcal{P}_k^{+N}
\end{array}\right)
\rightarrow
\left(\begin{array}{l}
\mathscr{X}_k^{+1} \left(d_{k,x}^{+1}, \theta_a^k, \boldsymbol{m}^k \right) \\
\mathscr{P}_k^{+1} \left(d_{k,p}^{+1}, \theta_a^k, \boldsymbol{m}^k \right) \\
\mathscr{X}_k^{+N} \left(d_{k,x}^{+N}, \theta_c^k, \boldsymbol{m}^k \right) \\
\mathscr{P}_k^{+N} \left(d_{k,p}^{+N}, \theta_c^k, \boldsymbol{m}^k \right)
\end{array}\right)
:=
\left(\begin{array}{l}
d_{k,x}^{+1} - M_a^k \sin{\theta_a^k} \\
d_{k,p}^{+1} + M_a^k \cos{\theta_a^k} \\
d_{k,x}^{+N} - M_c^k \sin{\theta_c^k} \\
d_{k,p}^{+N} + M_c^k \cos{\theta_c^k}
\end{array}\right)
```

## References

```{bibliography}
```
