#
#
#

import numpy as np


def sphere(
        ifunc,
        iflag,
        d,
        a,
        b,
        rho,
        mmax,
        rtp=None,
        btp=None,
        dtp=None,
        dhalf=None
):
    """
    Transform a spherical Earth model into an equivalent flat Earth model.

    Args:
        ifunc (int): Wave type (1=Love, 2=Rayleigh).
        iflag (int): Mode of operation (0=initialize, 1=apply transformation).
        d (ndarray): Layer thicknesses (km).
        a (ndarray): P-wave velocities (km/s).
        b (ndarray): S-wave velocities (km/s).
        rho (ndarray): Densities (g/cm³).
        mmax (int): Number of layers.

    Returns:
        tuple:
            d_new (ndarray): Transformed thicknesses.
            a_new (ndarray): Transformed P-wave velocities.
            b_new (ndarray): Transformed S-wave velocities.
            rho_new (ndarray): Transformed densities.
            aux (dict): Dictionary with temporary saved variables
                        (btp, rtp, dtp, dhalf) for later reuse.
    """
    # Earth radius in km
    ar = 6370.0

    # Copy inputs
    rho_new = rho.copy()

    # Initialize
    if iflag == 0:
        # Copy inputs
        a_new = a.copy()
        b_new = b.copy()
        d_new = d.copy()

        # Initialize storage
        dtp = d.copy()
        rtp = rho.copy()
        btp = np.zeros_like(d)

        # Cumulative thickness
        dr = 0.0

        # Initial radius
        r0 = ar

        # For each layer
        for i in range(mmax):
            # Add thickness
            dr += d[i]

            # Radius
            r1 = ar - dr

            # New depth (bottom, up)
            z0 = ar * np.log(ar / r0)
            z1 = ar * np.log(ar / r1)

            # New thickness
            d_new[i] = z1 - z0

            # Midpoint scaling
            tmp = (ar + ar) / (r0 + r1)
            a_new[i] *= tmp
            b_new[i] *= tmp
            btp[i] = tmp

            # New radius
            r0 = r1
        # end for
        dhalf = d_new[mmax - 1]
    # Apply transformation
    elif iflag == 1:
        # Check if rtp and btp are passed
        if rtp is None or btp is None:
            raise ValueError("You must pass rtp and btp when iflag == 1")
        # end if

        for i in range(mmax):
            if ifunc == 1:
                rho_new[i] = rtp[i] * btp[i] ** (-5.0)
            elif ifunc == 2:
                rho_new[i] = rtp[i] * btp[i] ** (-2.275)
            else:
                raise ValueError("Invalid ifunc: must be 1 (Love) or 2 (Rayleigh)")
            # end if ifunc
        # end for range(mmax)
    else:
        raise ValueError("Invalid iflag: must be 0 (init) or 1 (transform)")
    # end if iflag==0

    # Final adjustment
    d_new[mmax - 1] = 0.0

    return d_new, a_new, b_new, rho_new, {
        "btp": btp,
        "rtp": rtp,
        "dtp": dtp,
        "dhalf": dhalf
    }
# end sphere


def var(
        p,
        q,
        ra,
        rb,
        wvno,
        xka,
        xkb,
        dpth
):
    """
    Compute eigenfunction components and their combinations.

    Handles evanescent and oscillatory regimes using numerically stable formulas.

    Args:
        p (float): p = alpha * thickness
        q (float): q = beta * thickness
        ra (float): sqrt(wvno^2 - alpha^2)
        rb (float): sqrt(wvno^2 - beta^2)
        wvno (float): Wavenumber
        xka (float): omega / alpha
        xkb (float): omega / beta
        dpth (float): Layer thickness

    Returns:
        tuple: (
            w, cosp, exa, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz
        )
    """

    # Initialize values
    a0 = 1.0
    exa = 0.0
    pex = 0.0
    sex = 0.0

    # ---- P-wave eigenfunctions
    if wvno < xka:
        sinp = np.sin(p)
        w = sinp / ra
        x = -ra * sinp
        cosp = np.cos(p)
    elif wvno == xka:
        cosp = 1.0
        w = dpth
        x = 0.0
    else:
        pex = p
        fac = np.exp(-2.0 * p) if p < 16 else 0.0
        cosp = 0.5 * (1.0 + fac)
        sinp = 0.5 * (1.0 - fac)
        w = sinp / ra
        x = ra * sinp
    # end if

    # ---- S-wave eigenfunctions
    if wvno < xkb:
        sinq = np.sin(q)
        y = sinq / rb
        z = -rb * sinq
        cosq = np.cos(q)
    elif wvno == xkb:
        cosq = 1.0
        y = dpth
        z = 0.0
    else:
        sex = q
        fac = np.exp(-2.0 * q) if q < 16 else 0.0
        cosq = 0.5 * (1.0 + fac)
        sinq = 0.5 * (1.0 - fac)
        y = sinq / rb
        z = rb * sinq
    # end if

    # ---- Composite exponent
    exa = pex + sex
    a0 = np.exp(-exa) if exa < 60.0 else 0.0

    # ---- Eigenfunction products
    cpcq = cosp * cosq
    cpy = cosp * y
    cpz = cosp * z
    cqw = cosq * w
    cqx = cosq * x
    xy = x * y
    xz = x * z
    wy = w * y
    wz = w * z

    # Adjust cosq, y, z with exponent shift
    qmp = sex - pex
    fac = np.exp(qmp) if qmp > -40.0 else 0.0
    cosq *= fac
    y *= fac
    z *= fac

    return (
        w, cosp, exa,
        a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz
    )
# end var


def normc(ee):
    """
    Normalize the 5-element vector to prevent numerical under/overflow.

    Args:
        ee (np.ndarray): Vector of 5 elements (complex or real).

    Returns:
        tuple:
            ee_norm (np.ndarray): Normalized vector.
            ex (float): Logarithm of the normalization factor.
    """
    ee = np.asarray(ee, dtype=np.float64)
    # Find the maximum absolute value
    t1 = np.max(np.abs(ee))

    # Avoid division by zero
    if t1 < 1e-40:
        t1 = 1.0
    # end if

    # Normalize the vector
    ee_norm = ee / t1

    # Store log of scaling factor (exponent form)
    ex = np.log(t1)

    return ee_norm, ex
# end normc


def dnka(
        wvno2,
        gam,
        gammk,
        rho,
        a0,
        cpcq,
        cpy,
        cpz,
        cqw,
        cqx,
        xy,
        xz,
        wy,
        wz
):
    """
    Construct the 5x5 Dunkin's propagator matrix for one layer.

    Args:
        wvno2 (float): Square of the wavenumber (ω/c)^2.
        gam (float): gamma = 2 * (b/ω)^2 * wvno^2.
        gammk (float): 2 * (b/ω)^2.
        rho (float): Density of the layer.
        a0, cpcq, ..., wz (float): Precomputed terms from `var`.

    Returns:
        ca (np.ndarray): 5x5 Dunkin matrix.
    """
    one = 1.0
    two = 2.0

    ca = np.zeros((5, 5))

    gamm1 = gam - one
    twgm1 = gam + gamm1
    gmgmk = gam * gammk
    gmgm1 = gam * gamm1
    gm1sq = gamm1 * gamm1
    rho2 = rho * rho
    a0pq = a0 - cpcq

    # Row 1
    ca[0, 0] = cpcq - 2 * gmgm1 * a0pq - gmgmk * xz - wvno2 * gm1sq * wy
    ca[0, 1] = (wvno2 * cpy - cqx) / rho
    ca[0, 2] = -(twgm1 * a0pq + gammk * xz + wvno2 * gamm1 * wy) / rho
    ca[0, 3] = (cpz - wvno2 * cqw) / rho
    ca[0, 4] = -(2 * wvno2 * a0pq + xz + wvno2**2 * wy) / rho2

    # Row 2
    ca[1, 0] = (gmgmk * cpz - gm1sq * cqw) * rho
    ca[1, 1] = cpcq
    ca[1, 2] = gammk * cpz - gamm1 * cqw
    ca[1, 3] = -wz
    ca[1, 4] = ca[0, 3]

    # Row 4 (index 3)
    ca[3, 0] = (gm1sq * cpy - gmgmk * cqx) * rho
    ca[3, 1] = -xy
    ca[3, 2] = gamm1 * cpy - gammk * cqx
    ca[3, 3] = ca[1, 1]
    ca[3, 4] = ca[0, 1]

    # Row 5 (index 4)
    ca[4, 0] = -(2 * gmgmk * gm1sq * a0pq + gmgmk**2 * xz + gm1sq**2 * wy) * rho2
    ca[4, 1] = ca[3, 0]
    ca[4, 2] = -(gammk * gamm1 * twgm1 * a0pq + gam * gammk**2 * xz + gamm1 * gm1sq * wy) * rho
    ca[4, 3] = ca[1, 0]
    ca[4, 4] = ca[0, 0]

    # Row 3 (index 2): Combination using symmetry
    t = -2.0 * wvno2
    ca[2, 0] = t * ca[4, 2]
    ca[2, 1] = t * ca[3, 2]
    ca[2, 2] = a0 + 2 * (cpcq - ca[0, 0])
    ca[2, 3] = t * ca[1, 2]
    ca[2, 4] = t * ca[0, 2]

    return ca
# end dnka


def gtsolh(a, b, max_iter=5):
    """
    Estimate the initial phase velocity for a half-space using Newton-Raphson.

    Args:
        a (float): P-wave velocity (Vp).
        b (float): S-wave velocity (Vs).
        max_iter (int): Number of iterations for refinement (default: 5).

    Returns:
        float: Estimated phase velocity `c` for the fundamental mode.
    """
    # Initial guess: slightly below S-wave speed
    c = 0.95 * b

    # Iterations
    for _ in range(max_iter):
        # Rapport Vs/Vp
        gamma = b / a

        # Reduced velocity
        kappa = c / b
        k2 = kappa**2
        gk2 = (gamma * kappa)**2

        # Factors
        fac1 = np.sqrt(1.0 - gk2)
        fac2 = np.sqrt(1.0 - k2)

        # Dispersion function
        fr = (2.0 - k2)**2 - 4.0 * fac1 * fac2

        # Derivative of the dispersion function with respect to c (fr')
        frp = (
            -4.0 * (2.0 - k2) * kappa
            + 4.0 * fac2 * gamma**2 * kappa / fac1
            + 4.0 * fac1 * kappa / fac2
        ) / b

        # Newton-Raphson update
        c = c - fr / frp
    # end for max_iter

    return c
# end gtsolh


def dltar1(
        wvno,
        omega,
        d,
        a,
        b,
        rho,
        rtp,
        dtp,
        btp,
        mmax,
        llw,
        twopi
):
    """
    Computes the dispersion function value for SH (Love) waves
    using Haskell–Thomson formulation from halfspace to surface.

    Args:
        wvno (float): Wavenumber (omega / phase velocity).
        omega (float): Angular frequency.
        d, a, b, rho (array): Model parameters.
        rtp, dtp, btp (array): Spherical mapping parameters (unused here but kept for consistency).
        mmax (int): Number of layers.
        llw (int): Water layer index (1 or 2).
        twopi (float): Constant (2π).

    Returns:
        float: Value of the dispersion function Δ_SH (should be 0 at solution).
    """
    beta1 = float(b[mmax - 1])
    rho1 = float(rho[mmax - 1])
    xkb = omega / beta1
    wvnop = wvno + xkb
    wvnom = abs(wvno - xkb)
    rb = np.sqrt(wvnop * wvnom)

    e1 = rho1 * rb
    e2 = 1.0 / (beta1 ** 2)

    for m in range(mmax - 2, llw - 2, -1):  # from mmax-2 to llw-1
        beta1 = float(b[m])
        rho1 = float(rho[m])
        xmu = rho1 * beta1 ** 2
        xkb = omega / beta1
        wvnop = wvno + xkb
        wvnom = abs(wvno - xkb)
        rb = np.sqrt(wvnop * wvnom)
        q = float(d[m]) * rb

        if wvno < xkb:
            sinq = np.sin(q)
            y = sinq / rb
            z = -rb * sinq
            cosq = np.cos(q)
        elif wvno == xkb:
            cosq = 1.0
            y = float(d[m])
            z = 0.0
        else:
            fac = np.exp(-2.0 * q) if q < 16 else 0.0
            cosq = 0.5 * (1.0 + fac)
            sinq = 0.5 * (1.0 - fac)
            y = sinq / rb
            z = rb * sinq
        # end if

        e10 = e1 * cosq + e2 * xmu * z
        e20 = e1 * y / xmu + e2 * cosq

        xnor = max(abs(e10), abs(e20), 1.0e-40)
        e1 = e10 / xnor
        e2 = e20 / xnor
    # end for m

    return e1
# end dltar1


def dltar4(wvno, omega, d, a, b, rho, rtp, dtp, btp, mmax, llw, twopi):
    """
    Computes the dispersion function value for Rayleigh (P-SV) waves.

    Args:
        wvno (float): Wavenumber (omega / phase velocity).
        omega (float): Angular frequency.
        d, a, b, rho (array): Earth model parameters.
        rtp, dtp, btp (array): Spherical mapping parameters (unused here but passed for consistency).
        mmax (int): Number of layers.
        llw (int): Flag for water layer (1 = no water, 2 = water).
        twopi (float): 2π constant.

    Returns:
        float: Value of the dispersion function Δ_PS(V) to be zeroed.
    """
    if omega < 1.0e-4:
        omega = 1.0e-4
    # end if

    wvno2 = wvno ** 2

    # Bottom half-space parameters
    xka = omega / float(a[mmax - 1])
    xkb = omega / float(b[mmax - 1])
    wvnop = wvno + xka
    wvnom = abs(wvno - xka)
    ra = np.sqrt(wvnop * wvnom)
    wvnop = wvno + xkb
    wvnom = abs(wvno - xkb)
    rb = np.sqrt(wvnop * wvnom)

    beta = float(b[mmax - 1])
    t = beta / omega
    gammk = 2.0 * t * t
    gam = gammk * wvno2
    gamm1 = gam - 1.0
    rho1 = float(rho[mmax - 1])

    # Initial E-vector
    e = np.zeros(5)
    e[0] = rho1 ** 2 * (gamm1 ** 2 - gam * gammk * ra * rb)
    e[1] = -rho1 * ra
    e[2] = rho1 * (gamm1 - gammk * ra * rb)
    e[3] = rho1 * rb
    e[4] = wvno2 - ra * rb

    # Matrix stacking from bottom up
    for m in range(mmax - 2, llw - 2, -1):
        xka = omega / float(a[m])
        xkb = omega / float(b[m])
        beta = float(b[m])
        t = beta / omega
        gammk = 2.0 * t * t
        gam = gammk * wvno2

        wvnop = wvno + xka
        wvnom = abs(wvno - xka)
        ra = np.sqrt(wvnop * wvnom)

        wvnop = wvno + xkb
        wvnom = abs(wvno - xkb)
        rb = np.sqrt(wvnop * wvnom)

        dpth = float(d[m])
        rho1 = float(rho[m])

        p = ra * dpth
        q = rb * dpth

        # Compute interface variables
        (w, cosp, exa,
         a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz) = var(
            p, q, ra, rb, wvno, xka, xkb, dpth
        )

        # Compute Dunkin matrix
        ca = dnka(
            wvno2, gam, gammk, rho1,
            a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz
        )

        # Matrix-vector product
        ee = np.zeros(5)
        for i in range(5):
            ee[i] = sum(e[j] * ca[j][i] for j in range(5))

        # Normalize result
        ee = normc(ee, exa)

        # Update E vector
        e = ee
    # end for m

    # Handle water layer if present
    if llw != 1:
        xka = omega / float(a[0])
        wvnop = wvno + xka
        wvnom = abs(wvno - xka)
        ra = np.sqrt(wvnop * wvnom)
        dpth = float(d[0])
        rho1 = float(rho[0])
        p = ra * dpth
        znul = 1.0e-05

        (w, cosp, exa,
         a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz) = var(
            p, znul, ra, znul, wvno, xka, znul, dpth
        )
        w0 = -rho1 * w
        return cosp * e[0] + w0 * e[1]
    # end if

    return e[0]
# end dltar4


def dltar(wvno, omega, kk, d, a, b, rho, rtp, dtp, btp, mmax, llw, twopi):
    """
    Wrapper to compute the dispersion function value (Δ) for a given wave type.

    Args:
        wvno (float): Wavenumber (ω / c).
        omega (float): Angular frequency.
        kk (int): Wave type (1 = Love, 2 = Rayleigh).
        d, a, b, rho, rtp, dtp, btp (np.ndarray): Layer model parameters.
        mmax (int): Number of layers.
        llw (int): Water layer flag.
        twopi (float): 2π (for legacy compatibility).

    Returns:
        float: Value of the dispersion function Δ(wvno), to be zeroed.
    """
    if kk == 1:
        # Love wave dispersion function
        return dltar1(wvno, omega, d, a, b, rho, rtp, dtp, btp, mmax, llw, twopi)
    elif kk == 2:
        # Rayleigh wave dispersion function
        return dltar4(wvno, omega, d, a, b, rho, rtp, dtp, btp, mmax, llw, twopi)
    else:
        raise ValueError("Invalid wave type kk: must be 1 (Love) or 2 (Rayleigh)")
    # end if kk
# end dltar


def nevill(
        t,
        c1,
        c2,
        del1,
        del2,
        ifunc,
        d,
        a,
        b,
        rho,
        rtp,
        dtp,
        btp,
        mmax,
        llw,
        twopi,
        dltar_func
):
    """
    Refine the root using a hybrid method combining bisection and Neville's method.

    Args:
        t (float): Period.
        c1, c2 (float): Initial bracketing values for phase velocity.
        del1, del2 (float): Corresponding values of the dispersion function.
        ifunc (int): Wave type (1=Love, 2=Rayleigh).
        dltar_func (callable): Function to compute the dispersion value.
        d, a, b, rho, rtp, dtp, btp (np.ndarray): Model arrays.
        mmax, llw (int): Layer settings.
        twopi (float): 2π.

    Returns:
        cc (float): Refined phase velocity.
    """
    # Initialisation
    omega = twopi / t
    nev = 1
    nctrl = 1
    x = np.zeros(20)
    y = np.zeros(20)

    def half_step(
            c1,
            c2
    ):
        """
        Compute the midpoint value and corresponding dispersion function.

        Args:
            c1, c2 (float): Bracketing values for phase velocity.

        Returns:
            tuple: (c3, del3) for the midpoint and its dispersion value.
        """
        c3 = 0.5 * (c1 + c2)
        wvno = omega / c3
        del3 = dltar_func(wvno, omega, ifunc, d, a, b, rho, rtp, dtp, btp, mmax, llw, twopi)
        return c3, del3
    # end half_step

    # First estimation with bisection
    # c3 =  (c1 + c2) / 2
    # del3 = f(c3)
    c3, del3 = half_step(c1, c2)

    # While convergence or max iterations
    while nctrl < 100:
        # Increment iteration counter
        nctrl += 1

        # Check if c3 is inside [c1, c2]
        if c3 < min(c1, c2) or c3 > max(c1, c2):
            # nev ??
            nev = 0
            c3, del3 = half_step(c1, c2)
        # end if

        s13 = del1 - del3
        s32 = del3 - del2

        # Check if the root is between c1 and c3
        if np.sign(del3) * np.sign(del1) < 0:
            c2, del2 = c3, del3
        else:
            c1, del1 = c3, del3
        # end if

        # Convergence reached ?
        if abs(c1 - c2) <= 1e-6 * c1:
            break
        # end if

        # Monotonous function (+ root inside interval) => Neville
        # or Neville failed => bracketing (more robust)
        if np.sign(s13) != np.sign(s32):
            nev = 0
        # end if

        # Neville's method
        # del1: value of dispersion function at c1
        # del2: value of dispersion function at c2
        # s1, s2: torelance values for del1 and del2
        ss1 = abs(del1)
        s1 = 0.01 * ss1
        ss2 = abs(del2)
        s2 = 0.01 * ss2

        # If the gradient are too different (100x factor),
        # Neville's method is not used.
        # we go back to interval method and run again.
        if s1 > ss2 or s2 > ss1 or nev == 0:
            # No Nevill
            c3, del3 = half_step(c1, c2)
            nev = 1
            m = 1
        else:
            # Neville's method
            if nev == 2:
                # Add a point to previous interpolation
                x[m] = c3
                y[m] = del3
            else:
                # New Neville's interpolation
                x[0], y[0] = c1, del1
                x[1], y[1] = c2, del2

                # Max index
                m = 1
            # end if nev

            # Neville's interpolation (inverted to solve x(y=0))
            # From point 0 to point m-1
            for kk in range(m):
                # From last point to first point
                j = m - (kk+1)
                denom = y[m] - y[j]

                # We stop if the denominator is too small (instability)
                if abs(denom) < 1e-10 * abs(y[m]):
                    c3, del3 = half_step(c1, c2)
                    nev = 1
                    m = 1
                    break
                # end if

                x[j] = (-y[j] * x[j+1] + y[m] * x[j]) / denom
            else:
                # Get root from Neville's inverse interpolation
                c3 = x[0]
                wvno = omega / c3

                # Compute dispersion function at c3
                del3 = dltar_func(wvno, omega, ifunc, d, a, b, rho, rtp, dtp, btp, mmax, llw, twopi)

                # Check for convergence
                nev = 2
                m = min(m + 1, 10)
                continue  # back to loop
            # end for kk
        # end if

        # if Neville failed, do interval
        continue
    # End while

    return c3
# end nevill

