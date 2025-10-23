(sec:gates)=

# Gates

In this page, we introduce the symbols for the operations used in this document.
Futhermore, we enumerate the supported gates in both circuit and graph representations using these symbols.

```{note}
In MQC3, we use the convention {math}`\hbar=1`.
```

## Definitions of symbols

### Single mode operation symbols

(sec:gate-rotation)=

#### Phase rotation

```{math}
\hat{R}^{\dagger}(\phi)\binom{\hat{x}}{\hat{p}} \hat{R}(\phi)=\left(\begin{array}{cc}
\cos \phi & -\sin \phi \\
\sin \phi & \cos \phi
\end{array}\right)\binom{\hat{x}}{\hat{p}}
=R(\phi)\binom{\hat{x}}{\hat{p}}
```

````{note}
For observables {math}`\hat{O}_1`, {math}`\hat{O}_2`, ..., the operation {math}`\hat{U}` acts as

```{math}
\hat{U}^{\dagger}\left(\begin{array}{c}
\hat{O}_1 \\
\hat{O}_2 \\
\vdots
\end{array}\right) \hat{U}
=
\left(\begin{array}{c}
\hat{U}^{\dagger}\hat{O}_1\hat{U} \\
\hat{U}^{\dagger}\hat{O}_2\hat{U} \\
\vdots
\end{array}\right).
```
````

(sec:gate-squeezing)=

#### Squeezing

```{math}
\hat{S}^{\dagger}(r)\binom{\hat{x}}{\hat{p}} \hat{S}(r)=\left(\begin{array}{cc}
e^{-r} & 0 \\
0 & e^{r}
\end{array}\right)\binom{\hat{x}}{\hat{p}}
=S(r)\binom{\hat{x}}{\hat{p}}
```

There is also a definition of squeezing using a different operator.

```{math}
\hat{S}_\text{V}^{\dagger}(c)\binom{\hat{x}}{\hat{p}} \hat{S}_\text{V}(c)=\left(\begin{array}{cc}
\frac{1}{c} & 0 \\
0 & c
\end{array}\right)\binom{\hat{x}}{\hat{p}}
=S_\text{V}(c)\binom{\hat{x}}{\hat{p}}
```

The relation between the two operators is as follows.

```{math}
S_\text{V}(c) =
\begin{cases}
    S(\ln{c}) & \text{if } c > 0, \\
    -S(\ln{(-c)}) & \text{if } c < 0.
\end{cases}
```

#### Shear (X-invariant)

```{math}
\hat{P}^{\dagger}(\kappa)\binom{\hat{x}}{\hat{p}} \hat{P}(\kappa)=\left(\begin{array}{cc}
1 & 0 \\
2 \kappa & 1
\end{array}\right)\binom{\hat{x}}{\hat{p}}
=P(\kappa)\binom{\hat{x}}{\hat{p}}
```

#### Shear (P-invariant)

```{math}
\hat{Q}^{\dagger}(\eta)\binom{\hat{x}}{\hat{p}} \hat{Q}(\eta)=\left(\begin{array}{cc}
1 & 2 \eta \\
0 & 1
\end{array}\right)\binom{\hat{x}}{\hat{p}}
=Q(\eta)\binom{\hat{x}}{\hat{p}}
```

(sec:gate-displacement)=

#### Displacement

```{math}
\binom{\hat{x}}{\hat{p}} \overset{\hat{D}\left(d_x, d_p\right)}{\longrightarrow} \binom{\hat{x}+d_x}{\hat{p}+d_p}
```

{math}`\hat{D}\left(d_x, d_p\right)` can also be written as {math}`\hat{D}(\boldsymbol{d})`, where {math}`\boldsymbol{d}=\left(d_x, d_p\right)`.

### Two mode operation symbols

#### Controlled-Z

```{math}
\hat{C}_Z^{\dagger}(g)\left(\begin{array}{c}
\hat{x}_1 \\
\hat{x}_2 \\
\hat{p}_1 \\
\hat{p}_2
\end{array}\right) \hat{C}_Z(g)=
\left(\begin{array}{cccc}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & g & 1 & 0 \\
g & 0 & 0 & 1
\end{array}\right)
\left(\begin{array}{c}
\hat{x}_1 \\
\hat{x}_2 \\
\hat{p}_1 \\
\hat{p}_2
\end{array}\right)
=C_Z(g)\left(\begin{array}{c}
\hat{x}_1 \\
\hat{x}_2 \\
\hat{p}_1 \\
\hat{p}_2
\end{array}\right)
```

(sec:gate-bs-symbol)=

#### Beamsplitter interaction

```{math}
\hat{B}^{\dagger}(\sqrt{R}, \theta_\text{rel})\left(\begin{array}{c}
\hat{x}_1 \\
\hat{x}_2 \\
\hat{p}_1 \\
\hat{p}_2
\end{array}\right) \hat{B}(\sqrt{R}, \theta_\text{rel})
=
\left(\begin{array}{rrrr}
\cos (\alpha+\beta) \cos (\alpha-\beta) & \sin (\alpha+\beta) \sin (\alpha-\beta) & -\sin (\alpha+\beta) \cos (\alpha-\beta) & \sin (\alpha-\beta) \cos (\alpha+\beta) \\
\sin (\alpha+\beta) \sin (\alpha-\beta) & \cos (\alpha+\beta) \cos (\alpha-\beta) & \sin (\alpha-\beta) \cos (\alpha+\beta) & -\sin (\alpha+\beta) \cos (\alpha-\beta) \\
\sin (\alpha+\beta) \cos (\alpha-\beta) & -\sin (\alpha-\beta) \cos (\alpha+\beta) & \cos (\alpha+\beta) \cos (\alpha-\beta) & \sin (\alpha+\beta) \sin (\alpha-\beta) \\
-\sin (\alpha-\beta) \cos (\alpha+\beta) & \sin (\alpha+\beta) \cos (\alpha-\beta) & \sin (\alpha+\beta) \sin (\alpha-\beta) & \cos (\alpha+\beta) \cos (\alpha-\beta)
\end{array}\right)
\left(\begin{array}{c}
\hat{x}_1 \\
\hat{x}_2 \\
\hat{p}_1 \\
\hat{p}_2
\end{array}\right)
=B(\sqrt{R}, \theta_\text{rel})\left(\begin{array}{c}
\hat{x}_1 \\
\hat{x}_2 \\
\hat{p}_1 \\
\hat{p}_2
\end{array}\right)
```

```{math}
---
label: eq:bs-alpha-beta
---
\begin{align}
\alpha &= \frac{\theta_\text{rel} + \operatorname{arccos} \sqrt{R}}{2} \\
\beta &= \frac{\theta_\text{rel} - \operatorname{arccos} \sqrt{R}}{2}
\end{align}
```

````{note}
The definition using {math}`\alpha` and {math}`\beta` is more general, with the relations

```{math}
---
label: eq:bs-alpha-beta-general
---
\left\{
\begin{array}{l}
\theta_\text{rel} = \alpha + \beta, \\
R=\cos^2 (\alpha-\beta).
\end{array}
\right.
```

Eqs. {math:numref}`eq:bs-alpha-beta` are not the general solution of Eqs. {math:numref}`eq:bs-alpha-beta-general`.
````

#### Two-mode shear

```{math}
\hat{P_2}^{\dagger}(a, b)\left(\begin{array}{c}
\hat{x}_1 \\
\hat{x}_2 \\
\hat{p}_1 \\
\hat{p}_2
\end{array}\right) \hat{P_2}(a, b)=
\left(\begin{array}{cccc}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
2 a & b & 1 & 0 \\
b & 2 a & 0 & 1
\end{array}\right)
\left(\begin{array}{c}
\hat{x}_1 \\
\hat{x}_2 \\
\hat{p}_1 \\
\hat{p}_2
\end{array}\right)
=P_2(a, b)\left(\begin{array}{c}
\hat{x}_1 \\
\hat{x}_2 \\
\hat{p}_1 \\
\hat{p}_2
\end{array}\right)
```

## Supported gates in circuit representation

### Single mode gates

#### {py:class}`mqc3.circuit.ops.intrinsic.Measurement`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\theta`
*   - Operation
    - {math}`M\left(\theta\right)`: Measure {math}`\hat{x} \sin \theta+\hat{p} \cos \theta`
```

```{attention}
When converted to the machinery representation, the setting is that feedforward is not performed immediately after this operation. If feedforward is required, either {py:class}`mqc3.graph.ops.Initialization` in graph representation should be used instead, or the feedforward matrices {py:attr}`mqc3.machinery.MachineryRepr.ff_coeff_matrix_k_plus_1` and {py:attr}`mqc3.machinery.MachineryRepr.ff_coeff_matrix_k_plus_n` in the machinery representation must be manually set after the conversion.
```

#### {py:class}`mqc3.circuit.ops.intrinsic.Displacement`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`x`, {math}`p`
*   - Operation
    - {math}`D\left(x, p\right)`
```

#### {py:class}`mqc3.circuit.ops.intrinsic.PhaseRotation`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\phi`
*   - Operation
    - {math}`R(\phi)`
```

#### {py:class}`mqc3.circuit.ops.intrinsic.Squeezing`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\theta`
*   - Operation
    - {math}`R\left(-\frac{\pi}{2}\right) S_{\text{V}}(\cot \theta)`
```

#### {py:class}`mqc3.circuit.ops.std.Squeezing`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`r`
*   - Operation
    - {math}`S(r)`
```

Implemented as {math}`R(0) S(r) R(0)` using {ref}`sec:gate-circuit-arbitrary`.

#### {py:class}`mqc3.circuit.ops.intrinsic.Squeezing45`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\theta`
*   - Operation
    - {math}`R\left(-\frac{\pi}{4}\right) S_{\text{V}}(\cot \theta) R\left(\frac{\pi}{4}\right)`
```

#### {py:class}`mqc3.circuit.ops.intrinsic.ShearXInvariant`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\kappa`
*   - Operation
    - {math}`P(\kappa)`
```

#### {py:class}`mqc3.circuit.ops.intrinsic.ShearPInvariant`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\eta`
*   - Operation
    - {math}`Q(\eta)`
```

(sec:gate-circuit-arbitrary)=

#### {py:class}`mqc3.circuit.ops.intrinsic.Arbitrary`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\alpha`, {math}`\beta`, {math}`\lambda`
*   - Operation
    - {math}`R(\alpha) S(\lambda) R(\beta)`
```

### Two mode gates

#### {py:class}`mqc3.circuit.ops.intrinsic.BeamSplitter`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\sqrt{R}`, {math}`\theta_{\text{rel}}`
*   - Operation
    - {math}`B\left(\sqrt{R}, \theta_{\text{rel}}\right)`
```

(sec:gate-circuit-std-bs)=

#### {py:class}`mqc3.circuit.ops.std.BeamSplitter`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\theta`, {math}`\phi`
*   - Operation
    - {math}`R_{12} \left(\pi-\theta-\phi, \frac{\pi}{2}-\theta\right) Man\left(0, \frac{\pi}{2}, \theta, \theta+\frac{\pi}{2}\right) R_{12} \left(\phi-\pi, -\frac{\pi}{2}\right)`
```

{math}`Man\left(\theta_a, \theta_b, \theta_c, \theta_d\right)` is implemented as {ref}`sec:gate-circuit-manual`.

{math}`R_{12}(\theta_1, \theta_2)` refers to the operation

```{math}
R_{12}(\theta_1, \theta_2)
=
\left(\begin{array}{cccc}
\cos\theta_1 &  & -\sin\theta_1 &  \\
 & \cos\theta_2 &  & -\sin\theta_2 \\
\sin\theta_1 &  & \cos\theta_1 &  \\
 & \sin\theta_2 &  & \cos\theta_2
\end{array}\right),
```

which applies {math}`R(\theta_1)` to mode 1 and {math}`R(\theta_2)` to mode 2.
The operation, when expressed directly as a matrix, becomes

```{math}
R_{12} \left(\pi-\theta-\phi, \frac{\pi}{2}-\theta\right) Man\left(0, \frac{\pi}{2}, \theta, \theta+\frac{\pi}{2}\right) R_{12} \left(\phi-\pi, -\frac{\pi}{2}\right)
=
\left(\begin{array}{cccc}
\cos \theta & -\sin \theta \cos \phi & 0 & -\sin \theta \sin \phi \\
\sin \theta \cos \phi & \cos \theta & -\sin \theta \sin \phi & 0 \\
0 & \sin \theta \sin \phi & \cos \theta & -\sin \theta \cos \phi \\
\sin \theta \sin \phi & 0 & \sin \theta \cos \phi & \cos \theta
\end{array}\right).
```

```{admonition} Derivation
:class: hint

The equation manipulation is shown in {ref}`sec:std-bs-derivation`.
```

#### {py:class}`mqc3.circuit.ops.intrinsic.ControlledZ`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`g`
*   - Operation
    - {math}`C_Z(g)`
```

#### {py:class}`mqc3.circuit.ops.intrinsic.TwoModeShear`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`a`, {math}`b`
*   - Operation
    - {math}`P_2(a, b)`
```

(sec:gate-circuit-manual)=

#### {py:class}`mqc3.circuit.ops.intrinsic.Manual`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\theta_a`, {math}`\theta_b`, {math}`\theta_c`, {math}`\theta_d`
*   - Operation
    - Operation at measurement angles {math}`\left(\theta_a, \theta_b, \theta_c, \theta_d\right)` within a single macronode
```

(sec:gates-in-graph-repr)=

## Supported gates in graph representation

The measurement angles are shown only for the through case.
They are omitted for the swap case, as swapping the input modes within each macronode is achieved by exchanging the first two measurement angles, as described in {ref}`sec:through-and-swap`.

### Single mode gates

#### {py:class}`~mqc3.graph.ops.Measurement`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\theta`
*   - Operation
    - {math}`M\left(\theta\right)`: Measure {math}`\hat{x} \sin \theta+\hat{p} \cos \theta`
*   - Measurement angles (through)
    - {math}`(\theta, \theta, \theta, \theta)`
```

```{attention}
When converted to the machinery representation, the setting is that feedforward is not performed immediately after this operation. If feedforward is required, either {py:class}`mqc3.graph.ops.Initialization` should be used instead, or the feedforward matrices {py:attr}`mqc3.machinery.MachineryRepr.ff_coeff_matrix_k_plus_1` and {py:attr}`mqc3.machinery.MachineryRepr.ff_coeff_matrix_k_plus_n` in the machinery representation must be manually set after the conversion.
```

#### {py:class}`~mqc3.graph.ops.Initialization`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\theta`
*   - Operation
    - {math}`M\left(\theta\right)`: Measure {math}`\hat{x} \sin \theta+\hat{p} \cos \theta` and initialize a mode with a squeezing angle {math}`\theta + \frac{\pi}{2}`
*   - Measurement angles (through)
    - {math}`(\theta, \theta, \theta, \theta)`
```

(sec:gate-graph-repr-wiring)=

#### {py:class}`~mqc3.graph.Wiring`

```{list-table}
:stub-columns: 1

*   - Parameters
    -
*   - Operation
    - Identity operation
*   - Measurement angles (through)
    - {math}`\left(0, \frac{\pi}{2}, 0, \frac{\pi}{2}\right)`
```

(sec:gate-graph-repr-rotation)=

#### {py:class}`~mqc3.graph.ops.PhaseRotation`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\phi`
*   - Operation
    - {math}`R(\phi)`
*   - Measurement angles (through)
    - {math}`\left(\frac{\phi}{2}, \frac{\phi}{2}+\frac{\pi}{2}, \frac{\phi}{2}, \frac{\phi}{2}+\frac{\pi}{2}\right)`
```

(sec:gate-graph-repr-squeezing)=

#### {py:class}`~mqc3.graph.ops.Squeezing`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\theta`
*   - Operation
    - {math}`R\left(-\frac{\pi}{2}\right) S_{\text{V}}(\cot \theta)`
*   - Measurement angles (through)
    - {math}`(-\theta, \theta,-\theta, \theta)`
```

(sec:gate-graph-repr-squeezing45)=

#### {py:class}`~mqc3.graph.ops.Squeezing45`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\theta`
*   - Operation
    - {math}`R\left(-\frac{\pi}{4}\right) S_{\text{V}}(\cot \theta) R\left(\frac{\pi}{4}\right)`
*   - Measurement angles (through)
    - {math}`\left(\frac{\pi}{4}-\theta, \frac{\pi}{4}+\theta, \frac{\pi}{4}-\theta, \frac{\pi}{4}+\theta\right)`
```

(sec:gate-graph-repr-shear-x-inv)=

#### {py:class}`~mqc3.graph.ops.ShearXInvariant`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\kappa`
*   - Operation
    - {math}`P(\kappa)`
*   - Measurement angles (through)
    - {math}`\left(\arctan \kappa, \frac{\pi}{2}, \arctan \kappa, \frac{\pi}{2}\right)`
```

(sec:gate-graph-repr-shear-p-inv)=

#### {py:class}`~mqc3.graph.ops.ShearPInvariant`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\eta`
*   - Operation
    - {math}`Q(\eta)`
*   - Measurement angles (through)
    - {math}`(0, \operatorname{arccot} \eta, 0, \operatorname{arccot} \eta)`
```

````{note}
The definition of {math}`\operatorname{arccot}` exists in two forms, as shown in {numref}`fig:arccot_graphs`.
Here, the left definition is used, but as stated in {ref}`sec:angle-equivalence-2pi-pi`, using the definition on the right instead does not affect the operation, feedforward, or displacement.

```{figure} _images/arccot_graphs.svg
:name: fig:arccot_graphs
:alt: Graphs of the arccot function

Graphs of the {math}`\operatorname{arccot}` function. The left side has a range on {math}`(0, \pi)`, while the right side has a range of {math}`\left[-\frac{\pi}{2}, \frac{\pi}{2}\right)`
```
````

(sec:gate-graph-repr-arbitrary)=

#### {py:class}`~mqc3.graph.ops.Arbitrary`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\alpha`, {math}`\beta`, {math}`\lambda`
*   - Operation
    - {math}`R(\alpha) S(\lambda) R(\beta)`
*   - Measurement angles in the first macronode (through)
    - {math}`\left(\beta-\arctan e^{-\lambda}, \beta+\arctan e^{-\lambda}, \beta-\arctan e^{-\lambda}, \beta+\arctan e^{-\lambda}\right)`
*   - Measurement angles in the second macronode (through)
    - {math}`\left(\frac{\alpha-\beta}{2}+\frac{\pi}{4}, \frac{\alpha-\beta}{2}-\frac{\pi}{4}, \frac{\alpha-\beta}{2}+\frac{\pi}{4}, \frac{\alpha-\beta}{2}-\frac{\pi}{4}\right)`
```

### Two mode gates

(sec:gate-graph-repr-bs)=

#### {py:class}`~mqc3.graph.ops.BeamSplitter`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\sqrt{R}`, {math}`\theta_{\text{rel}}`
*   - Operation
    - {math}`B\left(\sqrt{R}, \theta_{\text{rel}}\right)`
*   - Measurement angles (through)
    - {math}`\left(\frac{\theta_{\text{rel}}}{2}+\frac{\arccos (\sqrt{R})}{2}, \frac{\theta_{\text{rel}}}{2}+\frac{\arccos (\sqrt{R})}{2}+\frac{\pi}{2}, \frac{\theta_{\text{rel}}}{2}-\frac{\arccos (\sqrt{R})}{2}, \frac{\theta_{\text{rel}}}{2}-\frac{\arccos (\sqrt{R})}{2}+\frac{\pi}{2}\right)`
```

(sec:gate-graph-repr-cz)=

#### {py:class}`~mqc3.graph.ops.ControlledZ`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`g`
*   - Operation
    - {math}`C_Z(g)`
*   - Measurement angles (through)
    - {math}`\left(-\arctan \left(\frac{g}{2}\right), \frac{\pi}{2}, \arctan \left(\frac{g}{2}\right), \frac{\pi}{2}\right)`
```

(sec:gate-graph-repr-2mode-shear)=

#### {py:class}`~mqc3.graph.ops.TwoModeShear`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`a`, {math}`b`
*   - Operation
    - {math}`P_2(a, b)`
*   - Measurement angles (through)
    - {math}`\left(\arctan \left(a-\frac{b}{2}\right), \frac{\pi}{2}, \arctan \left(a+\frac{b}{2}\right), \frac{\pi}{2}\right)`
```

(sec:gate-graph-repr-manual)=

#### {py:class}`~mqc3.graph.ops.Manual`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\theta_a`, {math}`\theta_b`, {math}`\theta_c`, {math}`\theta_d`
*   - Operation
    - Operation at measurement angles {math}`\left(\theta_a, \theta_b, \theta_c, \theta_d\right)` within a single macronode
*   - Measurement angles (through)
    - {math}`\left(\theta_a, \theta_b, \theta_c, \theta_d\right)`
```
