
import numpy as np
from surfdisp.utils import nevill  # <-- à adapter selon l'emplacement réel


def dummy_dltar(wvno, omega, ifunc, *args, **kwargs):
    """
    Dummy dispersion function: f(c) = c^2 - 2
    So f(wvno) = (omega / wvno)^2 - 2
    Root is at c = sqrt(2) ≈ 1.4142
    """
    c = omega / wvno
    return c**2 - 2
# end dummy_dltar


def test_nevill_sqrt2():
    t = 1.0
    c1 = 1.0
    c2 = 2.0
    omega = 2 * np.pi / t
    del1 = dummy_dltar(omega / c1, omega, 1)
    del2 = dummy_dltar(omega / c2, omega, 1)

    # Call the nevill function
    cc = nevill(
        t=t,
        c1=c1,
        c2=c2,
        del1=del1,
        del2=del2,
        ifunc=1,
        dltar_func=dummy_dltar,
        d=[],
        a=[],
        b=[],
        rho=[],
        rtp=[],
        dtp=[],
        btp=[],
        mmax=0,
        llw=1,
        twopi=2 * np.pi
    )

    # Check that the result is close to sqrt(2)
    assert np.isclose(cc, np.sqrt(2), atol=1e-6), f"Expected sqrt(2), got {cc}"
# end test_nevill_sqrt2

