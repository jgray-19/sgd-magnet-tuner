# lattice_sympy_4d_3bends_symbolic_L.py
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

# Common curvature h; different k0 for each bend
h = sp.symbols("h", real=True)                 # curvature 1/ρ
k0_1, k0_2, k0_3 = sp.symbols("k0_1 k0_2 k0_3", real=True)

# Symbolic lengths
Lb = sp.symbols("Lb", positive=True, real=True)   # physical length of each bend
Ld = sp.symbols("Ld", positive=True, real=True)   # drift length between bends

# For LHC we set beta0 = 1 (ultra-relativistic), so it will not appear as a symbol
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


def bend_4d(L, h_val, k0_val):
    """
    4x4 sector bend in (x, px, δ, 1) with curvature h and k0.

    Lecture formulae:
        ω² = h k0
        x(s)  = x0 C + px0 S/ω
                + (δ0 h/β0 + h - k0) (1-C)/ω²
        px(s) = -x0 ω S + px0 C
                + (δ0 h/β0 + h - k0) S/ω

    Here β0 = 1 (LHC).
    """
    L = sp.sympify(L)
    h_val = sp.sympify(h_val)
    k0_val = sp.sympify(k0_val)

    w2 = h_val * k0_val          # ω² = h k0
    w = sp.sqrt(w2)

    C = sp.cos(w * L)
    S = sp.sin(w * L)

    M = sp.eye(4)

    # (x, px) block
    M[0, 0] = C
    M[0, 1] = S / w
    M[1, 0] = -w * S
    M[1, 1] = C

    # column from δ (β0 = 1)
    M[0, 2] = h_val * (1 - C) / w2
    M[1, 2] = h_val * S / w

    # column from the constant term (h - k0)
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
# Elements: three bends with equal Lb and equal h, different k0_i,
# separated by equal drifts Ld.
# -------------------------------------------------
logger.info("Building element matrices")

MB1 = bend_4d(Lb, h, k0_1)
MB2 = bend_4d(Lb, h, k0_2)
MB3 = bend_4d(Lb, h, k0_3)

D = drift_4d(Ld)

# -------------------------------------------------
# Total 4x4 map from entrance of first bend → exit of third bend
# -------------------------------------------------
logger.info("Computing total 4x4 transfer matrix for three bends")

# R_3bends = chain(
#     MB1,
#     D,
#     MB2,
#     D,
#     MB3,
# )
R_2bends = chain(
    MB1,
    D,
    MB2,
)


if __name__ == "__main__":
    logger.info("Starting main execution")

    print("=== Total 4x4 R (entrance MB1 → exit MB3) ===")
    # Print without line-wrapping (unlimited line length)
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

    old_vec = sp.Matrix([sp.symbols("x0"), sp.symbols("px0"), sp.symbols("delta0"), 1])
    new_vec = R_2bends * old_vec
    print("\n=== New coordinates after 2 bends ===")
    print(sp.pretty(new_vec, use_unicode=True, wrap_line=False))
    print("\n=== New x and px expressions ===")
    print(sp.pretty(new_vec[0], use_unicode=True, wrap_line=False))
    print(sp.pretty(new_vec[1], use_unicode=True, wrap_line=False))
    print("\n=== Done ===")