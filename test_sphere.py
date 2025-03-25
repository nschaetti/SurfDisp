

import numpy as np
from surfdisp.utils import sphere  # Remplace par le chemin réel si besoin

# -----------------------------
# Étape 1 : Génération des données
# -----------------------------
nlayers = 6

z_bounds = [(100, 3000), (3000, 6000), (6000, 9000),
            (9000, 12000), (12000, 15000), (0, 0)]
vs_bounds = [(500, 4500), (500, 4500), (1000, 5000),
             (1000, 5000), (1000, 5000), (1000, 5000)]

np.random.seed(42)

z = np.array([np.random.uniform(*b) for b in z_bounds])
vs = np.array([np.random.uniform(*b) for b in vs_bounds])

# -----------------------------
# Étape 2 : Calcul de vp et rho
# -----------------------------
vp = 1.73 * vs  # approximation avec facteur de Poisson constant

# Tous en km et km/s
z = z / 1000
vp = vp / 1000
vs = vs / 1000

# relation de Gardner (approximation)
rho = 0.31 * (vs ** 0.25)

print("=== Modèle initial ===")
for i in range(nlayers):
    print(f"Layer {i+1}: z={z[i]} km, vp={vp[i]:.4f} km/s, vs={vs[i]:.4f} km/s, rho={rho[i]:.4f} g/cm³")
# end for

# Appel à la fonction sphere
d_out, a_out, b_out, rho_out, aux = sphere(
    ifunc=2,
    iflag=0,
    d=z,
    a=vp,
    b=vs,
    rho=rho,
    mmax=nlayers
)

# Affichage
print("=== Résultat de la version Python de sphere ===")
for i in range(nlayers):
    print(f"Layer {i+1}: d={d_out[i]:.4f} km, a={a_out[i]:.4f} km/s, b={b_out[i]:.4f} km/s, "
          f"rho={rho_out[i]:.4f} g/cm³, btp={aux['btp'][i]:.4f}")
