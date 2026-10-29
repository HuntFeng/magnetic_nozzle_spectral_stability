"""
How good is the thesis Frobenius + shooting treatment of the accelerating transonic flow?
 (a) exact local Taylor coefficients of P v'' + Q v' + R v = 0 at the sonic throat vs the thesis
     recurrence (which linearizes v0 and drops the O(z) parts of Q and R);
 (b) a careful shooting solver (series start at z = -delta, adaptive RK45, Newton in omega) started
     from the thesis eigenvalues, to see how far the thesis values move.
"""
import numpy as np
from scipy.integrate import solve_ivp
from common import *

s = sonic_slope()
# Taylor coefficients of v0 at the throat from the polynomial fit used in transonic_profile
zz = np.linspace(-0.05, 0.05, 101)
v0, dv0, d2v0 = transonic_profile(zz, True, h=0.06)
p = np.polynomial.Polynomial.fit(zz, v0, 10).convert()
v1, v2, v3 = p.coef[1], 2 * p.coef[2], 6 * p.coef[3]
print(f"v0'(0) = {v1:.10f} (analytic {s:.10f}),  v0''(0) = {v2:.6f},  v0'''(0) = {v3:.4f}")


def local_coeffs(w):
    """P1, P2, Q0, Q1, R0, R1 of P v'' + Q v' + R v = 0 expanded at z = 0 (P0 = 0)."""
    zz = np.linspace(-0.02, 0.02, 81)
    v0, dv0, d2v0 = transonic_profile(zz, True, h=0.03)
    P = 1 - v0**2
    Q = 2j * w * v0 - (3 * v0 + 1 / v0) * dv0
    R = w**2 + 2j * w * dv0 - (1 - 1 / v0**2) * dv0**2 - (v0 + 1 / v0) * d2v0
    fit = lambda f: np.polynomial.Polynomial.fit(zz, f, 6).convert().coef
    cP, cQ, cR = fit(P), fit(Q), fit(R)
    return cP[1], cP[2], cQ[0], cQ[1], cR[0], cR[1]


def series_coeffs(w, exact=True):
    """c0 = 1, c1, c2 of the regular Frobenius solution; exact=False reproduces the thesis recurrence."""
    P1, P2, Q0, Q1, R0, R1 = local_coeffs(w)
    c0 = 1.0
    c1 = -R0 * c0 / Q0
    if exact:
        c2 = -((Q1 + R0) * c1 + R1 * c0) / (2 * (P1 + Q0))
    else:
        a, b = Q0 / P1, R0 / P1
        c2 = -b * c1 / (2 * (1 + a))
    return c0, c1, c2


def shoot(w, delta=1e-3):
    """v(-1) for the regular solution normalised by v(0) = 1, integrating from z = -delta to -1."""
    c0, c1, c2 = series_coeffs(w)
    z0 = -delta
    y0 = [c0 + c1 * z0 + c2 * z0**2, c1 + 2 * c2 * z0]

    def rhs(z, y):
        v0, dv0, d2v0 = (np.asarray(a).item() for a in transonic_profile(np.array([z]), True))
        v, u = y
        du = -(w**2 * v + 2j * w * (v0 * u + dv0 * v) - (3 * v0 + 1 / v0) * dv0 * u
               - (1 - 1 / v0**2) * dv0**2 * v - (v0 + 1 / v0) * d2v0 * v) / (1 - v0**2)
        return [u, du]

    sol = solve_ivp(rhs, (z0, -1.0), np.array(y0, dtype=complex), rtol=1e-10, atol=1e-12, method="DOP853")
    return sol.y[0, -1]


def newton(w, tol=1e-10):
    for _ in range(40):
        f = shoot(w)
        h = 1e-6 * max(1, abs(w))
        df = (shoot(w + h) - f) / h
        dw = f / df
        w -= dw
        if abs(dw) < tol:
            return w, True
    return w, False


if __name__ == "__main__":
    thesis = np.sort_complex(np.load(os.path.join(ROOT, "data", "shooting_accelerating.npz"))["omega"])
    print("\nthesis omega          -> refined shooting omega        |shift|   c2(thesis) c2(exact)")
    out = []
    for w in thesis:
        wr, ok = newton(w)
        c2t = series_coeffs(w, exact=False)[2]
        c2e = series_coeffs(w, exact=True)[2]
        print(f"{w: .6f} -> {wr: .8f} {'' if ok else '(no conv)'} {abs(wr - w):.1e}   {c2t: .3f} {c2e: .3f}")
        out.append(wr)
    # sensitivity to the start offset delta
    w = out[0]
    print("\ndelta sensitivity of v(-1) at the first refined eigenvalue:")
    for d in (1e-2, 3e-3, 1e-3, 3e-4):
        print(f"  delta={d:.0e}  |v(-1)| = {abs(shoot(w, d)):.2e}")
    np.savez(os.path.join(DATA, "frobenius_refined.npz"), thesis=thesis, refined=np.array(out))
