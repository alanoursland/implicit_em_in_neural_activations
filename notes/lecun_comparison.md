# Comparison: VICReg vs This Work's InfoMax

## VICReg (Bardes, Ponce, LeCun 2021)

**Reference:** Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. *ICLR 2022*. arXiv:2105.04906

**Loss function:**

$$L = \lambda \cdot s(Z,Z') + \mu \cdot [v(Z) + v(Z')] + \nu \cdot [c(Z) + c(Z')]$$

Where:
- **Invariance:** $s(Z,Z') = \|Z - Z'\|^2$ (MSE between views)
- **Variance:** $v(Z) = \frac{1}{d} \sum_j \max(0, \gamma - \sqrt{\text{Var}(z_j) + \epsilon})$
- **Covariance:** $c(Z) = \frac{1}{d} \sum_{i \neq j} \text{Cov}(z_i, z_j)^2$

---

## This Work's InfoMax (Oursland 2025)

**Loss function (Section 3):**

$$L = L_{\text{LSE}} + \lambda_{\text{var}} \cdot L_{\text{var}} + \lambda_{\text{tc}} \cdot L_{\text{tc}}$$

Where:
- **LSE:** $-\log \sum_j \exp(-E_j)$ (implicit EM term)
- **Variance:** $L_{\text{var}} = -\sum_j \log \text{Var}(A_j)$
- **Decorrelation:** $L_{\text{tc}} = \|\text{Corr}(A) - I\|_F^2$

---

## Key Differences

| Aspect | VICReg | This Work |
|--------|--------|-----------|
| **Variance term** | Hinge loss on std dev: $\max(0, \gamma - \sigma)$ | Log barrier: $-\log(\text{Var})$ |
| **Decorrelation** | Covariance matrix off-diagonals | Correlation matrix off-diagonals |
| **Primary objective** | Invariance between views | LSE marginal (implicit EM) |
| **Architecture** | Siamese (two views) | Single encoder |
| **Theoretical framing** | Empirical / "it works" | GMM log-determinant analogy |

---

## Are They Equivalent?

**No, but they're closely related.**

### 1. Variance Term

VICReg uses a hinge loss (soft threshold). This work uses a log barrier (hard constraint). 

The log barrier is more directly analogous to the GMM log-determinant since:

$$\sum_j \log \text{Var}(A_j) = \log \det(\text{diag}(\Sigma))$$

VICReg's hinge is a practical approximation that allows variance to exceed the threshold without penalty.

### 2. Decorrelation Term

VICReg penalizes covariance. This work penalizes correlation.

$$\text{Correlation} = \frac{\text{Covariance}}{\sigma_i \cdot \sigma_j}$$

Correlation is covariance normalized by standard deviations. When combined with variance control, these have similar effects but aren't mathematically identical.

### 3. The LSE Term

**This is what VICReg doesn't have.**

VICReg uses invariance loss (match two augmented views of the same input). This work uses LSE (soft competition among components via implicit EM).

This is the core difference—combining implicit EM structure with InfoMax-style regularization.

---

## Summary

This work's InfoMax terms are **variations on VICReg's variance/covariance terms**, not identical copies. The log-barrier vs hinge choice is theoretically motivated (GMM connection).

The *real* difference is the LSE term replacing invariance:
- VICReg: "These two views should match"
- This work: "Components should compete softly for data"

**Similar ingredients. Different recipe. Different theoretical story.**

| | VICReg | This Work |
|-|--------|-----------|
| What prevents collapse | Variance hinge | Variance log-barrier |
| What prevents redundancy | Covariance penalty | Correlation penalty |
| What drives learning | View invariance | Implicit EM (LSE) |
| Theoretical basis | Empirical | GMM log-determinant |