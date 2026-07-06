"""
Verification for Proposition 1' (Smooth monotone kernels) in
supervised_study/draft/theory.tex.

Run:  py supervised_study/proofs/prop1_smooth.py
Output: supervised_study/proofs/prop1_smooth.md

CLAIM (Proposition 1', prop:smooth-kernel)
    d_j = phi(w_j^T x + b_j), private parameters, phi' >= 0, |phi'| <= B1,
    |phi''| <= B2. Chain rule:
        Hess_theta L = J^T (Hess_d L) J + sum_j (dL/dd_j) Hess_theta d_j.
    First term <= (1/2) B1^2 ||xt||^2  (Lemma 1 and ||J||^2 <= B1^2 ||xt||^2).
    Second term <= B2 ||xt||^2  (since dL/dd_j = r_j >= 0 and sum_j r_j = 1).
    Total <= (B1^2/2 + B2) ||xt||^2, independent of theta.

    For Softplus: phi' = sigmoid in (0,1) => B1 = 1; phi'' = sigmoid(1-sigmoid)
    in (0, 1/4] => B2 = 1/4; bound = 0.75 ||xt||^2.

ROLE: this proposition is reassurance (the experiments use ReLU/Softplus, not the
clean affine kernel). The QUALITATIVE claim -- parameter-independence survives a
smooth monotone kernel -- is what the paper uses; the exact constant is not spent
anywhere downstream. The bound is verified valid (and loose) below.
"""

import numpy as np

RNG = np.random.default_rng(3)
LINES = []


def log(s=""):
    print(s)
    LINES.append(s)


def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)


def check_softplus_bound():
    """||Hess_theta L||_2 <= (B1^2/2 + B2)||xt||^2 = 0.75||xt||^2 for softplus."""
    B1, B2 = 1.0, 0.25
    ok = True
    worst_ratio = 0.0
    for K, n in [(4, 5), (8, 8)]:
        x = RNG.standard_normal(n)
        xt = np.append(x, 1.0)
        theta = RNG.standard_normal(K * (n + 1))
        m = len(theta)

        def L(th):
            th = th.reshape(K, n + 1)
            z = th[:, :n] @ x + th[:, n]
            dd = softplus(z)
            return -np.log(np.exp(-dd).sum())

        eps = 1e-4

        def g(th):
            return np.array(
                [(L(th + eps * np.eye(m)[i]) - L(th - eps * np.eye(m)[i])) / (2 * eps)
                 for i in range(m)]
            )

        Hn = np.array(
            [(g(theta + eps * np.eye(m)[i]) - g(theta - eps * np.eye(m)[i])) / (2 * eps)
             for i in range(m)]
        )
        Hn = (Hn + Hn.T) / 2
        num = np.linalg.norm(Hn, 2)
        bound = (B1 ** 2 / 2 + B2) * (xt @ xt)
        worst_ratio = max(worst_ratio, num / bound)
        ok &= num <= bound + 1e-5
    return ok, worst_ratio


def check_relu_ae_bound():
    """ReLU: the curvature bound holds a.e., but a dead unit has zero gradient
    (absorbing state). We confirm both the a.e. Hessian bound and the zero-gradient
    pathology on the inactive side."""
    ok = True
    worst_ratio = 0.0
    dead_grad_norm = 0.0
    # a.e.: away from kinks, ReLU has phi'=1, phi''=0, so bound reduces to Prop 1's 0.5||xt||^2
    for K, n in [(5, 5)]:
        x = RNG.standard_normal(n)
        xt = np.append(x, 1.0)
        kink_margin = 1e-2
        for _ in range(100):
            theta = RNG.standard_normal(K * (n + 1)) * 2.0
            th = theta.reshape(K, n + 1)
            z = th[:, :n] @ x + th[:, n]
            if np.min(np.abs(z)) > kink_margin:
                break
        else:
            raise RuntimeError("Could not sample ReLU preactivations away from kinks")
        m = len(theta)

        def relu(z):
            return np.maximum(z, 0.0)

        def L(th):
            th = th.reshape(K, n + 1)
            z = th[:, :n] @ x + th[:, n]
            dd = relu(z)
            return -np.log(np.exp(-dd).sum())

        eps = 1e-4

        def g(th):
            return np.array(
                [(L(th + eps * np.eye(m)[i]) - L(th - eps * np.eye(m)[i])) / (2 * eps)
                 for i in range(m)]
            )

        Hn = np.array(
            [(g(theta + eps * np.eye(m)[i]) - g(theta - eps * np.eye(m)[i])) / (2 * eps)
             for i in range(m)]
        )
        Hn = (Hn + Hn.T) / 2
        num = np.linalg.norm(Hn, 2)
        bound = 0.5 * (xt @ xt)
        worst_ratio = max(worst_ratio, num / bound)
        ok &= num <= bound + 1e-3  # a.e. reduces to Prop 1 (B1=1,B2=0)

    # absorbing state: a component inactive for every sample has zero numerical
    # gradient for its parameter block, not just zero derivative of a standalone if.
    K, n, batch = 4, 3, 6
    X = RNG.normal(size=(batch, n))
    theta = RNG.normal(size=(K, n + 1))
    theta[0, :n] = 0.0
    theta[0, n] = -5.0

    def relu(z):
        return np.maximum(z, 0.0)

    def L_batch(th_flat):
        th = th_flat.reshape(K, n + 1)
        z = X @ th[:, :n].T + th[:, n]
        dd = relu(z)
        return np.mean([-np.log(np.exp(-row).sum()) for row in dd])

    th0 = theta.reshape(-1)
    eps = 1e-6
    dead_indices = np.arange(n + 1)
    dead_grad = np.array([
        (L_batch(th0 + eps * np.eye(K * (n + 1))[i])
         - L_batch(th0 - eps * np.eye(K * (n + 1))[i])) / (2 * eps)
        for i in dead_indices
    ])
    dead_grad_norm = np.linalg.norm(dead_grad)
    ok &= dead_grad_norm < 1e-10
    return ok, worst_ratio, dead_grad_norm


def main():
    log("# Proposition 1' -- Smooth Monotone Kernels")
    log()
    log("Verification of `theory.tex`, Proposition `prop:smooth-kernel`. "
        "Generated by `proofs/prop1_smooth.py`.")
    log()
    log("Role: reassurance that parameter-independence survives the smooth kernel "
        "the experiments use. The paper spends the *qualitative* claim, not the "
        "exact constant.")
    log()

    ok_sp, ratio = check_softplus_bound()
    log(f"- **Softplus (B1=1, B2=1/4)**: `||Hess_theta L||_2 <= 0.75||xt||^2`; "
        f"worst numeric/bound ratio `{ratio:.3f}` (bound valid, and loose). "
        f"**{'PASS' if ok_sp else 'FAIL'}**")

    ok_relu, relu_ratio, dead_grad_norm = check_relu_ae_bound()
    log(f"- **ReLU (a.e. + absorbing state)**: away from kinks the bound reduces "
        f"to Prop 1's `0.5||xt||^2` (B1=1, B2=0), worst numeric/bound ratio "
        f"`{relu_ratio:.3f}`; a component inactive on the whole batch has "
        f"finite-difference gradient-block norm `{dead_grad_norm:.2e}` "
        f"(the documented absorbing-state pathology, not a bound violation). "
        f"**{'PASS' if ok_relu else 'FAIL'}**")

    log()
    all_ok = ok_sp and ok_relu
    log(f"**Overall: {'ALL CHECKS PASS' if all_ok else 'FAILURE -- see above'}.**")
    log()
    log("_Scope: confirms the parameter-independent bound holds for the kernels "
        "used experimentally. The constant is not tight and is not used "
        "downstream; the qualitative claim is what the paper relies on._")

    with open(__file__.replace('.py', '.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(LINES) + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
