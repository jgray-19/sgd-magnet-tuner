# lattice_sympy_4d_2bends_small_dk_h_eq_k.py
import logging

import sympy as sp

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Global symbols
# -------------------------------------------------
logger.info("Defining global symbols and parameters")

# Single base strength / curvature k, and small deviations dk_i
k = sp.symbols("k", real=True)                  # base k, and h ≡ k
dk1, dk2, dk3 = sp.symbols("dk1 dk2 dk3", real=True)  # small deviations

# Symbolic lengths
Lb = sp.symbols("Lb", positive=True, real=True)   # physical length of each bend
Ld = sp.symbols("Ld", positive=True, real=True)   # drift length between bends

# β0 = 1 for LHC (ultra-relativistic)
beta0_val = 1

# -------------------------------------------------
# 4D primitives: (x, px, δ, 1)
# -------------------------------------------------
logger.info("Defining 4D primitive functions")


def drift_4d(L):
    """
    4x4 drift in (x, px, δ, 1).
    x -> x + L px, px -> px, δ -> δ, 1 -> 1.
    """
    L = sp.sympify(L)
    M = sp.eye(4)
    M[0, 1] = L
    return M


def bend_4d(L, k_base, dk):
    """
    4x4 sector bend in (x, px, δ, 1) with h = k_base and k0 = k_base + dk.

    From the notes (with β0 = 1):
        ω² = h k0 = k_base * k0
        x(s)  = x0 C + px0 S/ω
                + (δ0 h + h - k0) (1-C)/ω²
        px(s) = -x0 ω S + px0 C
                + (δ0 h + h - k0) S/ω

    Here h = k_base, k0 = k_base + dk.
    """
    L = sp.sympify(L)
    k_base = sp.sympify(k_base)
    dk = sp.sympify(dk)

    k0_val = k_base + dk          # local k0
    h_val = k_base                # h = k

    w2 = h_val * k0_val           # ω² = h k0 = k_base * (k_base + dk)
    w = sp.sqrt(w2)

    C = sp.cos(w * L)
    S = sp.sin(w * L)

    M = sp.eye(4)

    # (x, px) block
    M[0, 0] = C
    M[0, 1] = S / w
    M[1, 0] = -w * S
    M[1, 1] = C

    # column from δ (β0 = 1, h = k_base)
    M[0, 2] = h_val * (1 - C) / w2
    M[1, 2] = h_val * S / w

    # column from the constant term (h - k0) = (k_base - (k_base + dk)) = -dk
    M[0, 3] = (h_val - k0_val) * (1 - C) / w2
    M[1, 3] = (h_val - k0_val) * S / w

    # δ and 1 unchanged
    M[2, 2] = 1
    M[3, 3] = 1

    return sp.simplify(M)


def chain(*elements):
    """Multiply in physical left→right order: total = E_n · ... · E_2 · E_1."""
    M = sp.eye(4)
    for i, E in enumerate(elements):
        logger.debug(f"Multiplying element {i + 1}/{len(elements)}")
        M = E * M
        M = sp.simplify(M)
    return M


# -------------------------------------------------
# Helpers for small-δk expansion
# -------------------------------------------------
dk_syms = (dk1, dk2, dk3)


def series_in(expr, var):
    """Expand expr in `var` about 0 up to O(var^2) and drop O(var^2)."""
    expr = sp.sympify(expr)
    return sp.series(expr, var, 0, 2).removeO()


def linearise_bend_in_dk(M, dk):
    """Apply first-order series in dk to every matrix element."""
    return M.applyfunc(lambda e: series_in(e, dk))


def drop_quadratic(M):
    """
    After we have linearised individual bends, the full map is polynomial
    in dk1, dk2, dk3. Drop all terms with total degree > 1.
    """
    def trunc(expr):
        expr = sp.expand(expr)
        if not any(expr.has(dk) for dk in dk_syms):
            return expr
        poly = sp.Poly(expr, *dk_syms)
        res = 0
        for monom, coeff in poly.terms():
            if sum(monom) <= 1:
                term = coeff
                for s, p in zip(dk_syms, monom):
                    if p:
                        term *= s**p
                res += term
        return sp.simplify(res)

    return M.applyfunc(trunc)


# -------------------------------------------------
# Elements: two bends with equal Lb and equal base k (h = k),
# small deviations dk1, dk2, separated by a drift Ld.
# -------------------------------------------------
logger.info("Building element matrices")

# Exact bends with k0 = k + dki and h = k
MB1_exact = bend_4d(Lb, k, dk1)
MB2_exact = bend_4d(Lb, k, dk2)

# Linearise each bend in its own dk (expands sin/cos in dk)
MB1_lin = linearise_bend_in_dk(MB1_exact, dk1)
MB2_lin = linearise_bend_in_dk(MB2_exact, dk2)

D = drift_4d(Ld)

# Build chain, then drop all O(dk^2) including dk1*dk2
R_2bends_exact = chain(
    MB1_lin,
    D,
    MB2_lin,
)

R_2bends = drop_quadratic(R_2bends_exact)

# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting main execution")

    print("=== Total 4x4 R (entrance MB1 → exit MB2), linear in dk, with h = k ===")
    print(sp.pretty(R_2bends, use_unicode=True, wrap_line=False))

    print("\n=== Selected elements (x, px, δ, 1) ===")
    print("R11 (x→x):   ", sp.simplify(R_2bends[0, 0]))
    print("R12 (px→x):  ", sp.simplify(R_2bends[0, 1]))
    print("R13 (δ→x):   ", sp.simplify(R_2bends[0, 2]))
    print("R14 (1→x):   ", sp.simplify(R_2bends[0, 3]))
    print("R21 (x→px):  ", sp.simplify(R_2bends[1, 0]))
    print("R22 (px→px): ", sp.simplify(R_2bends[1, 1]))
    print("R23 (δ→px):  ", sp.simplify(R_2bends[1, 2]))
    print("R24 (1→px):  ", sp.simplify(R_2bends[1, 3]))
    print("R33 (δ→δ):   ", sp.simplify(R_2bends[2, 2]))
    print("R44 (1→1):   ", sp.simplify(R_2bends[3, 3]))

    x0, px0, delta0 = sp.symbols("x0 px0 delta0")
    old_vec = sp.Matrix([x0, px0, delta0, 1])
    new_vec = R_2bends * old_vec
    print("\n=== New coordinates after 2 bends (linear in dk, h = k) ===")
    print(sp.pretty(new_vec, use_unicode=True, wrap_line=False))
    print("\n=== New x and px expressions ===")
    print(sp.pretty(new_vec[0], use_unicode=True, wrap_line=False))
    print(sp.pretty(new_vec[1], use_unicode=True, wrap_line=False))
    print("\n=== Done ===")
