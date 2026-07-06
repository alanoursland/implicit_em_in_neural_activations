# Proof verification scripts

Runnable numerical/symbolic checks for the constants and inequalities in
[`../draft/theory.tex`](../draft/theory.tex). Each script verifies one result and
writes a markdown record of what it confirmed. All output is ASCII to avoid
encoding/mojibake issues.

| Script | Verifies | Record |
|---|---|---|
| `lemma1.py` | Lemma 1: `grad = r`, Hessian `= rr^T - diag(r)`, `\|\|C(r)\|\|_2 <= 1/2` (tight) | `lemma1.md` |
| `prop1.py` | Prop 1: gradient block `r_j*xtilde`, Kronecker Hessian, `\|\|H\|\| <= (1/2)\|\|xtilde\|\|^2` | `prop1.md` |
| `prop2.py` | Prop 2: label clamp drops, `J@1=0`, `\|\|J\|\|_2 = sqrt(K)*\|\|r\|\|_2 <= sqrt(K)` (closed form), leading-term bound `<= (K/2) sigma_max(W2)^2`, and that the NLS residual term is real | `prop2.md` |
| `prop1_smooth.py` | Prop 1': Softplus `<= 0.75\|\|xtilde\|\|^2`; ReLU a.e. + absorbing state | `prop1_smooth.md` |

## Run

```
py supervised_study/proofs/lemma1.py
py supervised_study/proofs/prop1.py
py supervised_study/proofs/prop2.py
py supervised_study/proofs/prop1_smooth.py
```

Requires `numpy` and `sympy`. Each exits non-zero if any check fails, so they can
gate CI. Re-run and re-read the `.md` output whenever a proof's stated constant
changes.

## What these are (and are not)

These confirm the **specific inequalities and constants** hold -- and are tight
where the paper claims tightness -- at sampled and symbolically witnessed points.
They exist because the Prop 2 NLS-Jacobian norm was derived incorrectly by hand
several times before being computed here: the correct closed form is
`\|\|J\|\|_2 = sqrt(K)*\|\|r\|\|_2`, bounded by `sqrt(K)` on the simplex, NOT `2`
or `1+sqrt(K)`.

They are **not** theorem provers. In particular:

- The `prop2.py` spectrum check verifies the closed-form eigenvalues of `J^T J`
  (`{K*s once, 0 once, 1 K-2 times}`, `s = \|\|r\|\|^2`) against the numeric
  spectrum over many `r` and symbolically at exact simplex points. This makes the
  global `\|\|J\|\| <= sqrt(K)` claim a computed fact, not a sampling artifact --
  but the authoritative proof is the closed-form argument in `theory.tex`.
- Proposition 2 bounds only the **Gauss-Newton (leading) term** of the composed
  Hessian. `prop2.py` also confirms that the **NLS second-order residual term is
  genuinely present and non-negligible**, so the proposition does *not* claim a
  bound on the full composed Hessian -- only that composition introduces a
  `sigma_max(W2)^2` dependence absent from the isolated site. The full composed
  curvature is characterized empirically in the paper's mechanism section.

They are a guard against arithmetic and spectral-norm errors, complementary to the
analytic proofs in `theory.tex`.
