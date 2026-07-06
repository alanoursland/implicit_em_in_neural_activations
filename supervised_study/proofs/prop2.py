"""
Verification for Proposition 2 (Composition forfeits the uniform bound) in
supervised_study/draft/theory.tex.

Run:  py supervised_study/proofs/prop2.py
Output: supervised_study/proofs/prop2.md

This is the load-bearing proposition (it is the paper's title). Its NLS-Jacobian
spectral-norm constant was gotten WRONG by hand three times before being computed
here; this script exists so that error cannot recur silently.

CLAIM (Proposition 2)
    y = NLS(d),  h = W2 y,  L_CE = CE(h, c),  p = softmax(h).
    NLS Jacobian J = I - 1 r^T  (1 = ones vector, r = softmax(-d)).
    CE-logit Hessian = diag(p) - pp^T (label clamp is linear -> drops out).
    Pulled back:  J^T W2^T (diag(p) - pp^T) W2 J.
    Bounds:
        ||diag(p) - pp^T||_2 <= 1/2                            (Lemma 1)
        J @ 1 = 0  (ones vector in kernel; triangle bound 1+sqrt(K) is LOOSE)
        ||J||_2 = sqrt(K)  exactly, attained at one-hot r      (TIGHT)
        => leading factor <= (K/2) sigma_max(W2)^2 ,  parameter-dependent.
"""

import numpy as np
import sympy as sp

RNG = np.random.default_rng(2)
LINES = []


def log(s=""):
    print(s)
    LINES.append(s)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def check_J_spectrum_closed_form():
    """Closed form: eigenvalues of J^T J are exactly {K*s (once), 0 (once),
    1 (K-2 times)} where s = ||r||_2^2.  Hence ||J||_2 = sqrt(K)*||r||_2 for ALL
    simplex r (not only vertices), and <= sqrt(K) since ||r||_2 <= 1.  This is the
    closed-form fact the proof relies on; verified here against the numeric
    spectrum, not merely sampled."""
    ok = True
    worst = 0.0
    for K in (4, 6, 10, 25, 64):
        for _ in range(500):
            r = RNG.dirichlet(RNG.choice([0.05, 0.3, 1.0, 5.0]) * np.ones(K))
            s = float(r @ r)
            J = np.eye(K) - np.outer(np.ones(K), r)
            ev = np.sort(np.linalg.eigvalsh(J.T @ J))
            pred = np.sort(np.concatenate([[0.0], [K * s], np.ones(K - 2)]))
            worst = max(worst, np.abs(ev - pred).max())
            ok &= np.allclose(ev, pred, atol=1e-8)
            # closed-form norm identity
            ok &= abs(np.linalg.norm(J, 2) - np.sqrt(K * s)) < 1e-7
            ok &= np.sqrt(K * s) <= np.sqrt(K) + 1e-9   # since s <= 1
    return ok, worst


def check_J_kernel():
    """J @ 1 = 0 for any simplex r (the ones vector is in the kernel)."""
    ok = True
    for K in (3, 5, 10, 25, 64):
        r = RNG.dirichlet(np.ones(K))
        J = np.eye(K) - np.outer(np.ones(K), r)
        ok &= np.linalg.norm(J @ np.ones(K)) < 1e-10
    return ok


def check_J_spectrum_symbolic():
    """Symbolic (small K): eigenvalues of J^T J are {0, K*s, 1,...} with the
    span{1,r} block having eigenvalues {0, K*s}. Confirms the closed form, not
    just the vertex case."""
    results = []
    ok = True
    for K in (3, 4):
        r = sp.Matrix(sp.symbols(f'r0:{K}', positive=True))
        ones = sp.ones(K, 1)
        J = sp.eye(K) - ones * r.T
        JTJ = sp.expand(J.T * J)
        # substitute the simplex constraint sum r = 1 by leaving r free and checking
        # the characteristic structure via eigenvalues at a symbolic-but-normalized point
        # (full symbolic eigvals over free r are heavy; verify the two nonzero eigs
        # equal {0, K*||r||^2} at several exact rational simplex points)
        for pt in ([sp.Rational(1, K)] * K,
                   [sp.Rational(1, 2), sp.Rational(1, 2)] + [0] * (K - 2)):
            subs = {r[i]: pt[i] for i in range(K)}
            evs = (JTJ.subs(subs)).eigenvals()
            s = sum(p ** 2 for p in pt)
            top = max(evs.keys())
            ok &= sp.simplify(top - K * s) == 0
            results.append((K, [float(p) for p in pt], sp.nsimplify(top)))
    return ok, results


def check_gauss_newton_bound():
    """Gauss-Newton (leading) term ONLY:
    ||J^T W2^T (diag p - pp^T) W2 J||_2 <= (K/2) sigma_max(W2)^2.
    This is the term the proposition bounds; the NLS second-order term is NOT
    bounded here (see check_residual_is_real)."""
    ok = True
    worst_ratio = 0.0
    for K, C in [(5, 10), (25, 10), (25, 40)]:
        ones = np.ones(K)
        for _ in range(300):
            r = RNG.dirichlet(RNG.choice([0.05, 0.3, 1.0]) * np.ones(K))
            J = np.eye(K) - np.outer(ones, r)
            W2 = RNG.standard_normal((C, K))
            p = softmax(RNG.standard_normal(C))
            Hp = np.diag(p) - np.outer(p, p)
            M = J.T @ W2.T @ Hp @ W2 @ J
            lhs = np.linalg.norm(M, 2)
            bound = 0.5 * K * np.linalg.norm(W2, 2) ** 2
            worst_ratio = max(worst_ratio, lhs / bound)
            ok &= lhs <= bound + 1e-8
    return ok, worst_ratio


def check_residual_is_real():
    """The full Hessian of L_CE wrt d is NOT equal to the Gauss-Newton term: the
    NLS second-order term is present and comparable in size. This documents why
    the proposition claims only the leading term, not the full Hessian."""
    max_resid_over_full = 0.0
    for K, C in [(25, 10)]:
        for _ in range(20):
            dv = RNG.standard_normal(K)
            r = softmax(-dv)  # careful: r = softmax(-d)
            r = np.exp(-dv - (-dv).max()); r = r / r.sum()
            W2 = RNG.standard_normal((C, K)) * 0.5
            c = int(RNG.integers(C))

            def L(dd):
                y = dd + np.log(np.exp(-dd).sum())
                h = W2 @ y
                return -h[c] + np.log(np.exp(h).sum())

            eps = 1e-4
            m = K

            def g(dd):
                return np.array([(L(dd + eps * np.eye(m)[i]) - L(dd - eps * np.eye(m)[i])) / (2 * eps)
                                 for i in range(m)])

            Hn = np.array([(g(dv + eps * np.eye(m)[i]) - g(dv - eps * np.eye(m)[i])) / (2 * eps)
                           for i in range(m)])
            Hn = (Hn + Hn.T) / 2
            y = dv + np.log(np.exp(-dv).sum())
            h = W2 @ y
            p = softmax(h)
            J = np.eye(K) - np.outer(np.ones(K), r)
            GN = J.T @ W2.T @ (np.diag(p) - np.outer(p, p)) @ W2 @ J
            full = np.linalg.norm(Hn, 2)
            resid = np.linalg.norm(Hn - GN, 2)
            if full > 1e-9:
                max_resid_over_full = max(max_resid_over_full, resid / full)
    # We assert the residual is NON-negligible (i.e., GN is not the whole story).
    return max_resid_over_full > 0.1, max_resid_over_full


def check_label_clamp_drops():
    """CE Hessian wrt logits is diag(p)-pp^T regardless of the label."""
    ok = True
    for K in (3, 10):
        h = RNG.standard_normal(K)
        p = softmax(h)
        Hp = np.diag(p) - np.outer(p, p)
        for c in range(K):
            def L(hh):
                return -hh[c] + np.log(np.exp(hh).sum())
            eps = 1e-4
            m = K

            def g(hh):
                return np.array([(L(hh + eps * np.eye(m)[i]) - L(hh - eps * np.eye(m)[i])) / (2 * eps)
                                 for i in range(m)])
            Hn = np.array([(g(h + eps * np.eye(m)[i]) - g(h - eps * np.eye(m)[i])) / (2 * eps)
                           for i in range(m)])
            Hn = (Hn + Hn.T) / 2
            ok &= np.abs(Hn - Hp).max() < 1e-4
    return ok


def main():
    log("# Proposition 2 -- Composition Forfeits the Uniform Bound")
    log()
    log("Verification of `theory.tex`, Proposition `prop:composition` "
        "(the paper's title claim). Generated by `proofs/prop2.py`.")
    log()
    log("> The NLS-Jacobian norm was derived incorrectly by hand several times "
        "(claimed `<= 2`, then `1+sqrt(K)` with a wrong tightness/range claim) "
        "before being computed. The correct closed form is "
        "`||J||_2 = sqrt(K)*||r||_2 <= sqrt(K)`.")
    log()

    ok_clamp = check_label_clamp_drops()
    log(f"- **Label clamp drops from the Hessian**: CE-logit Hessian equals "
        f"`diag(p) - pp^T` for every label c. **{'PASS' if ok_clamp else 'FAIL'}**")

    ok_kern = check_J_kernel()
    log(f"- **Kernel**: `J @ 1 = 0` for every simplex r (ones vector in kernel; "
        f"this is why the triangle bound `1+sqrt(K)` is loose). "
        f"**{'PASS' if ok_kern else 'FAIL'}**")

    ok_spec, worst = check_J_spectrum_closed_form()
    log(f"- **Closed-form spectrum**: eigenvalues of `J^T J` are exactly "
        f"`{{K*s (once), 0 (once), 1 (K-2 times)}}` with `s = ||r||_2^2`, so "
        f"`||J||_2 = sqrt(K)*||r||_2 <= sqrt(K)` for ALL simplex r (not only "
        f"vertices). Verified vs numeric spectrum over K in {{4,6,10,25,64}}; "
        f"worst eigenvalue mismatch `{worst:.2e}`. **{'PASS' if ok_spec else 'FAIL'}**")

    ok_sym, sym = check_J_spectrum_symbolic()
    log(f"- **Spectrum (symbolic spot-check)**: at exact simplex points "
        f"(uniform and two-point), the top eigenvalue of `J^T J` equals `K*s`. "
        f"**{'PASS' if ok_sym else 'FAIL'}**")

    ok_b, ratio = check_gauss_newton_bound()
    log(f"- **Gauss-Newton (leading) term** "
        f"`||J^T W2^T (diag p - pp^T) W2 J||_2 <= (K/2) sigma_max(W2)^2`: holds "
        f"over random W2, p, r; worst LHS/bound ratio `{ratio:.3f}` (<= 1). "
        f"**{'PASS' if ok_b else 'FAIL'}**")

    ok_res, resid_frac = check_residual_is_real()
    log(f"- **Residual is real (scope check)**: the NLS second-order term is "
        f"present and NON-negligible -- the full Hessian differs from the "
        f"Gauss-Newton term (worst residual/full norm ratio `{resid_frac:.2f}`). "
        f"The proposition therefore bounds only the leading term, not the full "
        f"Hessian. **{'PASS (residual confirmed present)' if ok_res else 'FAIL'}**")

    log()
    log("The leading factor `(K/2) sigma_max(W2)^2` depends on the **learned** "
        "spectrum of the corridor W2 (and on K), unlike Lemma 1's fixed data "
        "constant. That dependence -- introduced by composition and absent from "
        "the isolated site -- is the forfeited guarantee. The full composed "
        "curvature (including the residual) is measured empirically in the paper.")
    log()
    all_ok = ok_clamp and ok_kern and ok_spec and ok_sym and ok_b and ok_res
    log(f"**Overall: {'ALL CHECKS PASS' if all_ok else 'FAILURE -- see above'}.**")
    log()
    log("_Scope: confirms the Hessian decomposition, the closed-form Jacobian "
        "norm (numerically over many r and symbolically at exact points), the "
        "leading-term bound, and that the residual term is genuinely present. It "
        "does not bound the full composed Hessian analytically; the paper claims "
        "only the leading-term dependence._")

    with open(__file__.replace('.py', '.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(LINES) + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
