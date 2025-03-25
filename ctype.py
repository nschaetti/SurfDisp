#
#
# 

# Imports
import numpy as np
import ctypes

# Chargement de la bibliothèque Fortran
lib = ctypes.CDLL('./fortran/libsurfdisp96.so')

# Définition de la signature
lib.surfdisp96_.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.POINTER(ctypes.c_double),
]
lib.surfdisp96_.restype = None

# Données d’entrée
nlayer = ctypes.c_int(4)
iflsph = ctypes.c_int(1)
iwave = ctypes.c_int(1)
mode = ctypes.c_int(0)
igr = ctypes.c_int(1)
kmax = ctypes.c_int(50)
err = ctypes.c_double(0.0)

# Profils
thkm = np.array([2., 5., 10., 20.], dtype=np.float64)
vpm = np.array([6., 6.2, 6.4, 6.6], dtype=np.float64)
vsm = np.array([3.5, 3.6, 3.7, 3.8], dtype=np.float64)
rhom = np.array([2.7, 2.8, 2.9, 3.0], dtype=np.float64)

# Résultats à remplir
t = np.zeros(kmax.value, dtype=np.float64)
cg = np.zeros(kmax.value, dtype=np.float64)

# Appel Fortran
lib.surfdisp96_(
    thkm,
    vpm,
    vsm,
    rhom,
    ctypes.byref(nlayer),
    ctypes.byref(iflsph),
    ctypes.byref(iwave),
    ctypes.byref(mode),
    ctypes.byref(igr),
    ctypes.byref(kmax),
    t,
    cg,
    ctypes.byref(err)
)

# Affichage des résultats
print("Fréquences (t):", t)
print("Vitesses de phase (cg):", cg)
print("Erreur retournée:", err.value)
