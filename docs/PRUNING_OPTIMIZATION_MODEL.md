# ANN Pruning Optimization Model

This note documents the optimization model used by the experimental measurement-only pruning workflow in `src/measurement_control/torch_rul_pso_milp_pruning.py`.

## Purpose

The pruning stage is not trained directly on the final CMAPSS validation or test objective. Instead, it solves a mixed-integer linear optimization model that tries to reproduce the output of a previously trained dense reference ANN on a small calibration subset.

Within the updated repository scope, this pruning model is part of the measurement pipeline only. It is not coupled to any downstream control or controller-tuning stage.

The workflow is:

1. Stage 1 PSO screens ANN architectures cheaply.
2. A dense reference ANN is trained for each selected top-k candidate.
3. A pruning MILP chooses which arcs to keep.
4. The masked pruned ANN is fine-tuned with gradient descent.
5. Final model selection still happens on the held-out validation split, and the official NASA test split is touched only once at the end.

This separation matters scientifically: the MILP is a structure-reduction step, not the final predictive training objective.

## Why Teacher Matching Instead of Direct RUL Loss?

The local solver stack uses `scipy.optimize.milp`, which supports MILP but not a native mixed-integer quadratic objective. Because of that, the pruning model minimizes the absolute deviation between the pruned network output and the dense reference network output on a calibration subset.

So the pruning MILP answers this question:

> Which arcs can be removed while keeping the pruned ANN as close as possible to the dense reference ANN on representative samples?

That is why the model uses a linearized `L1` teacher-matching objective rather than direct end-to-end ANN retraining.

## Calibration Subset

The calibration subset is created by `sample_calibration_subset(...)`. It is a deterministic, target-stratified subset of the already preprocessed training windows. This keeps the optimization problem smaller while still covering different RUL levels.

Important points:

- The calibration subset is not the official test set.
- The calibration subset is not the whole training set.
- The final predictive quality still depends on the later fine-tuning stage.

## Common Notation

Let:

- `N` be the number of calibration samples
- `d` be the flattened ANN input dimension
- `H1` be the number of neurons in hidden layer 1
- `H2` be the number of neurons in hidden layer 2
- `x^n in R^d` be calibration sample `n`
- `y_hat^n` be the scalar output of the dense reference ANN for sample `n`

The dense reference network weights and biases are fixed constants inside the MILP:

- `W^(1), b^(1)` for input to hidden layer 1
- `W^(2), b^(2)` for hidden layer 1 to hidden layer 2
- `W^(3), b^(3)` for hidden layer 2 to output

The pruning decision is encoded with binary arc-retention variables:

- `z^(1)` for input-to-hidden-1 arcs
- `z^(2)` for hidden-1-to-hidden-2 arcs
- `z^(3)` for hidden-2-to-output arcs

If a binary variable is `1`, that arc is kept. If it is `0`, that arc is pruned.

## Objective

The pruning MILP minimizes the total absolute teacher-matching error:

```math
\min \sum_{n=1}^{N} e^n
```

with auxiliary absolute-value constraints:

```math
e^n \ge y^n - \hat{y}^n, \qquad
e^n \ge -(y^n - \hat{y}^n), \qquad
e^n \ge 0
```

where `y^n` is the output of the pruned ANN for calibration sample `n`.

## One-Hidden-Layer Exact MILP

For a one-hidden-layer ReLU ANN:

```math
x \rightarrow h^{(1)} \rightarrow y
```

the model uses:

- binary arc variables `z^(1)_{j,i}` for input-to-hidden arcs
- binary arc variables `z^(2)_j` for hidden-to-output arcs
- pre-activation variables `a^{n}_j`
- ReLU activation variables `h^{n}_j`
- ReLU state binaries `delta^{n}_j`
- auxiliary variables `q^{n}_j = h^{n}_j z^{(2)}_j`

### Hidden layer equation

```math
a^n_j = \sum_{i=1}^{d} W^{(1)}_{j,i} x^n_i z^{(1)}_{j,i} + b^{(1)}_j
```

### ReLU linearization

The ReLU relation `h^n_j = max(0, a^n_j)` is enforced with big-M bounds:

```math
h^n_j \ge a^n_j
```

```math
h^n_j \ge 0
```

```math
h^n_j \le a^n_j - L^n_j (1 - \delta^n_j)
```

```math
h^n_j \le U^n_j \delta^n_j
```

where `L^n_j` and `U^n_j` are valid lower and upper bounds on `a^n_j`.

### Hidden-to-output product linearization

Because the output uses both the continuous activation `h^n_j` and the binary keep variable `z^(2)_j`, the product is linearized through:

```math
q^n_j = h^n_j z^{(2)}_j
```

with:

```math
q^n_j \le h^n_j
```

```math
q^n_j \le U^{h,n}_j z^{(2)}_j
```

```math
q^n_j \ge h^n_j - U^{h,n}_j (1 - z^{(2)}_j)
```

### Output equation

```math
y^n = \sum_{j=1}^{H1} W^{(2)}_j q^n_j + b^{(2)}
```

## Two-Hidden-Layer Exact MILP

For a two-hidden-layer ReLU ANN:

```math
x \rightarrow h^{(1)} \rightarrow h^{(2)} \rightarrow y
```

the exact MILP extends the same logic to the second hidden layer.

### Additional binary arc variables

- `z^(2)_{k,j}` for hidden-1-to-hidden-2 arcs
- `z^(3)_k` for hidden-2-to-output arcs

### Additional hidden-layer variables

- `a^{(2),n}_k` second hidden-layer pre-activation
- `h^{(2),n}_k` second hidden-layer ReLU activation
- `delta^{(2),n}_k` second hidden-layer ReLU state

### Hidden-1 to hidden-2 product linearization

The second layer depends on products of the form:

```math
q^{(12),n}_{k,j} = h^{(1),n}_j z^{(2)}_{k,j}
```

with constraints:

```math
q^{(12),n}_{k,j} \le h^{(1),n}_j
```

```math
q^{(12),n}_{k,j} \le U^{h1,n}_j z^{(2)}_{k,j}
```

```math
q^{(12),n}_{k,j} \ge h^{(1),n}_j - U^{h1,n}_j (1 - z^{(2)}_{k,j})
```

Then the second hidden layer is:

```math
a^{(2),n}_k = \sum_{j=1}^{H1} W^{(2)}_{k,j} q^{(12),n}_{k,j} + b^{(2)}_k
```

and:

```math
h^{(2),n}_k = \max(0, a^{(2),n}_k)
```

again enforced with a ReLU big-M formulation.

### Hidden-2 to output product linearization

For the output layer:

```math
q^{(23),n}_k = h^{(2),n}_k z^{(3)}_k
```

with:

```math
q^{(23),n}_k \le h^{(2),n}_k
```

```math
q^{(23),n}_k \le U^{h2,n}_k z^{(3)}_k
```

```math
q^{(23),n}_k \ge h^{(2),n}_k - U^{h2,n}_k (1 - z^{(3)}_k)
```

### Output equation

```math
y^n = \sum_{k=1}^{H2} W^{(3)}_k q^{(23),n}_k + b^{(3)}
```

## Reduced-Neighborhood Exact MILP for Two Hidden Layers

In practice, the full two-hidden-layer exact MILP can become very large. The
current code therefore uses a reduced-neighborhood exact strategy by default
for two-hidden-layer candidates.

The idea is:

1. Compute activation-aware arc scores on the calibration subset.
2. Build an initial keep mask by ranking arcs with those scores.
3. Improve that mask with a bounded local search that swaps low-value kept arcs
   with high-value dropped arcs when teacher-matching error improves.
4. Fix arcs far from the refined heuristic boundary to `1` or `0`.
5. Leave only a limited band of uncertain arcs near the boundary free.
6. Solve the exact MILP only on that reduced binary neighborhood.

This is not the same as a heuristic final solution. The solve is still exact
with respect to the remaining free binary decisions, but the search space is
smaller because many decisions are fixed before optimization.

If `z` denotes the full vector of binary arc decisions, the reduced-neighborhood
approach partitions it into:

- `z_fixed_keep = 1`
- `z_fixed_drop = 0`
- `z_free in {0,1}`

The MILP then optimizes only `z_free`, while the fixed decisions are enforced as
variable bounds.

This strategy is used because the local SciPy `milp` interface does not expose a
general MIP-start argument, so the most practical way to inject a good
heuristic structure is to shrink the exact binary search space directly.

### Activation-Aware Arc Scores

For each dense layer, the current implementation scores an arc by:

```math
\text{score}_{j,i} = |W_{j,i}| \cdot \mathbb{E}[|u_i|]
```

where `u_i` is the incoming activation for that arc on the calibration subset.

This means:

- input-to-hidden scores use the observed ANN input values
- hidden-to-hidden scores use hidden-layer activations after ReLU
- hidden-to-output scores use second-hidden-layer activations after ReLU

So an arc with a large weight but almost never-used input is not treated the
same as an arc with both a large weight and a frequently active upstream signal.

### Local Search Step

After the initial activation-aware ranking, the heuristic is improved with a
bounded hill-climbing swap search:

- keep the total pruning budget fixed
- within each layer, identify weak kept arcs and strong dropped arcs
- test a limited number of keep/drop swaps
- accept the best improving swap on the calibration teacher-matching objective
- stop when no improving swap is found or the round limit is reached

The objective used in this heuristic improvement is the calibration
teacher-matching MAE:

```math
\frac{1}{N} \sum_{n=1}^{N} |y^n - \hat{y}^n|
```

This remains aligned with the exact MILP objective while staying much cheaper
than solving the full MIP directly.

## Pruning Budget

Let `rho` be the keep fraction, for example `rho = 0.5` means keep half of the arcs.

The total number of candidate arcs is:

```math
A_total =
\begin{cases}
d H1 + H1, & \text{one hidden layer} \\
d H1 + H1 H2 + H2, & \text{two hidden layers}
\end{cases}
```

The pruning budget is:

```math
\sum z \le \rho A_{total}
```

or, when the code uses an exact budget:

```math
\sum z = \rho A_{total}
```

This means the MILP chooses where to keep the allowed arcs, not how many beyond the budget.

## Bound Construction

The big-M formulation needs valid bounds on each hidden-layer pre-activation.

For layer 1, bounds are derived directly from the calibration input values and the fixed dense weights.

For layer 2, bounds are derived from the upper bounds of hidden layer 1 and the fixed dense hidden-to-hidden weights.

This is important because:

- overly loose bounds make the MILP weaker and slower
- invalid bounds make the formulation incorrect

## Why the Model Can Become Large

The exact two-hidden-layer model grows quickly because of the auxiliary variables `q^(12)`.

The dominant term is:

```math
N \times H1 \times H2
```

So if:

- `N = 16`
- `H1 = 100`
- `H2 = 100`

then the model already needs `160,000` `q^(12)` variables alone, before adding ReLU binaries, other continuous variables, and constraints.

That is why:

- `pruning_calibration_size` must stay small
- exact two-hidden-layer pruning can be much slower than one-hidden-layer pruning
- reduced-neighborhood exact MILP is often more practical than a full exact
  two-hidden-layer MILP
- the code still keeps a magnitude-pruning fallback if the solver does not return a solution

## Relation to the Implementation

The current implementation maps to the following functions:

- `sample_calibration_subset(...)`: chooses the teacher-matching subset
- `build_milp_pruning_problem_one_hidden_layer(...)`: exact MILP for one hidden layer
- `build_milp_pruning_problem_two_hidden_layers(...)`: exact MILP for two hidden layers
- `activation_aware_arc_scores(...)`: scores arcs using weight magnitude and observed activations
- `activation_aware_local_search_masks(...)`: improves the heuristic mask before the exact solve
- `build_reduced_neighborhood_fixings(...)`: fixes obvious binary arc decisions before the two-hidden-layer exact solve
- `solve_milp_pruning(...)`: solves the MILP and falls back only if needed
- `masked_train_model(...)`: fine-tunes the pruned ANN after the optimization stage

## Practical Interpretation

This pruning optimization model should be interpreted as a structured compression step:

- PSO searches for promising dense ANN topologies
- the MILP removes arcs while preserving the dense ANN behavior on representative calibration samples
- gradient-based fine-tuning then adapts the masked network to the predictive task again

So the optimization model is exact with respect to the pruning variables and the ReLU linearization, but it is still only one component of the full predictive pipeline.
