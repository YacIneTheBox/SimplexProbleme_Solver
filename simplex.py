import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-9

class LinearProgram:
    def __init__(self, c, A, b, sense, is_max=True):
        """
        c : vecteur des coefficients de l'objectif (numpy 1D)
        A : matrice des contraintes (numpy 2D)
        b : second membre (numpy 1D)
        sense : liste de chaînes "<=", ">=", "=" pour chaque contrainte
        is_max : True pour Max, False pour Min
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.sense = sense
        self.is_max = is_max
        self.n_vars = len(c)
        self.n_constraints = len(b)

def to_standard_form(lp: LinearProgram):
    """
    Met le problème en forme standard type cours:
    Max Z, Ax = b, x >= 0, en ajoutant des variables d'écart.
    Ici on gère directement les contraintes <= en ajoutant e_i.
    On ne gère pas les variables artificielles (donc pas de vrai 2-phase complet).
    """
    A = lp.A.copy()
    b = lp.b.copy()
    c = lp.c.copy()
    # Si c'est un Min, on transforme en Max
    if not lp.is_max:
        c = -c

    m, n = A.shape
    slack_cols = []
    new_A = []
    new_c = list(c)
    # On construit un tableau avec variables originales + slack
    for i in range(m):
        row = list(A[i])
        if lp.sense[i] == "<=":
            # ajout variable d'écart ei >= 0
            slack = [0.0] * len(slack_cols) + [1.0]
            row_extended = row + slack
            slack_cols.append(n + len(slack_cols))
        elif lp.sense[i] == ">=":
            # pour rester simple: on multiplie par -1 pour obtenir <=
            row = (-A[i]).tolist()
            b[i] = -b[i]
            slack = [0.0] * len(slack_cols) + [1.0]
            row_extended = row + slack
            slack_cols.append(n + len(slack_cols))
        else:  # "="
            # on suppose qu'il y a déjà une variable de base ailleurs
            slack = [0.0] * len(slack_cols)
            row_extended = row + slack

        new_A.append(row_extended)

    new_A = np.array(new_A, dtype=float)
    new_c = np.array(list(c) + [0.0]*len(slack_cols), dtype=float)

    return new_c, new_A, b, slack_cols

def build_tableau(c, A, b):
    """
    Construit le tableau du simplexe classique:
    [A | b]
    [c | 0]  (ligne de la fonction objectif avec -c pour un Max)
    """
    m, n = A.shape
    tableau = np.zeros((m + 1, n + 1))
    tableau[:m, :n] = A
    tableau[:m, -1] = b
    tableau[-1, :n] = -c   # pour Max: -c en bas
    tableau[-1, -1] = 0.0
    return tableau

def can_improve(tableau):
    # Il y a amélioration possible si un coefficient de la dernière ligne est négatif
    last_row = tableau[-1, :-1]
    return np.any(last_row < -EPS)

def choose_entering_variable(tableau):
    last_row = tableau[-1, :-1]
    # règle de Dantzig: colonne du minimum (plus négatif)
    j = np.argmin(last_row)
    if last_row[j] >= -EPS:
        return None
    return j

def choose_leaving_variable(tableau, entering_col):
    m, n_plus1 = tableau.shape
    m -= 1  # sans la ligne objectif
    ratios = []
    for i in range(m):
        col_val = tableau[i, entering_col]
        if col_val > EPS:
            ratios.append(tableau[i, -1] / col_val)
        else:
            ratios.append(np.inf)
    row = np.argmin(ratios)
    if ratios[row] == np.inf:
        return None
    return row

def pivot(tableau, row, col):
    pivot_val = tableau[row, col]
    tableau[row, :] /= pivot_val
    m, n = tableau.shape
    for i in range(m):
        if i != row:
            factor = tableau[i, col]
            tableau[i, :] -= factor * tableau[row, :]

def simplex(lp: LinearProgram):
    c_std, A_std, b_std, slack_cols = to_standard_form(lp)
    tableau = build_tableau(c_std, A_std, b_std)
    m, n_plus1 = tableau.shape
    m_constraints = m - 1
    n_vars_total = n_plus1 - 1

    # variables de base initiales = slack (si <=)
    basis = [-1]*m_constraints
    for idx, col in enumerate(slack_cols):
        basis[idx] = col

    iterations = 0
    while can_improve(tableau):
        col = choose_entering_variable(tableau)
        if col is None:
            break
        row = choose_leaving_variable(tableau, col)
        if row is None:
            raise ValueError("Problème non borné (unbounded).")
        pivot(tableau, row, col)
        basis[row] = col
        iterations += 1
        if iterations > 100:
            raise RuntimeError("Trop d'itérations, arrêt.")

    # Extraire la solution
    x = np.zeros(n_vars_total)
    for i, col in enumerate(basis):
        if 0 <= col < n_vars_total:
            x[col] = tableau[i, -1]

    z = tableau[-1, -1]
    # Rappel: la dernière ligne contient -c·x, donc Z = -tableau[-1,-1]
    z = -z

    # On ne retourne que les variables "originales" (sans slack)
    x_original = x[:lp.n_vars]
    if not lp.is_max:
        z = -z

    return x_original, z, tableau, basis

# ---------- PARTIE GRAPHIQUE 2D ----------

def plot_feasible_region_and_solution(lp: LinearProgram, x_opt, z_opt):
    if lp.n_vars != 2:
        print("L'affichage graphique est limité aux problèmes à 2 variables.")
        return

    # On va chercher les intersections pour approximer la région admissible
    x1 = np.linspace(0, 20, 400)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Dessin des contraintes:
    for i in range(lp.n_constraints):
        a1, a2 = lp.A[i]
        bi = lp.b[i]
        if abs(a2) > EPS:
            y = (bi - a1 * x1) / a2
        else:
            # contrainte verticale: x = bi / a1
            x_vert = bi / a1
            ax.axvline(x_vert, color=f"C{i}", label=f"Contrainte {i+1}")
            continue

        if lp.sense[i] == "<=":
            ax.plot(x1, y, color=f"C{i}", label=f"C{i+1}: {a1}x1+{a2}x2<={bi}")
            ax.fill_between(x1, y, 0, where=(y >= 0), color=f"C{i}", alpha=0.05)
        elif lp.sense[i] == ">=":
            ax.plot(x1, y, color=f"C{i}", linestyle="--", label=f"C{i+1}: {a1}x1+{a2}x2>={bi}")
        else:
            ax.plot(x1, y, color=f"C{i}", linestyle=":", label=f"C{i+1}: {a1}x1+{a2}x2={bi}")

    # Axes et styles
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlim(0, max(10, x_opt[0]*1.4 if x_opt[0] > 0 else 5))
    ax.set_ylim(0, max(10, x_opt[1]*1.4 if x_opt[1] > 0 else 5))
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Région admissible et solution optimale (méthode du simplexe)")

    # Point optimal
    ax.scatter([x_opt[0]], [x_opt[1]], color="red", s=80, zorder=5, label="Solution optimale")
    ax.annotate(f"({x_opt[0]:.2f}, {x_opt[1]:.2f})\nZ*={z_opt:.2f}",
                (x_opt[0], x_opt[1]), xytext=(10, 10),
                textcoords="offset points", color="red")

    # Ligne iso-profit passant par la solution
    c1, c2 = lp.c
    if abs(c2) > EPS:
        y_iso = (z_opt - c1 * x1) / c2
        ax.plot(x1, y_iso, color="green", linestyle="--", label="Ligne iso-profit (Z*)")

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ---------- EXEMPLE TYPE COURS ----------

def exemple_cours():
    # Exemple du polyèdre des ateliers (LP-1_3)
    # Max Z = 300 x1 + 500 x2
    # x1 <= 4
    # x2 <= 12
    # 3x1 + 2x2 <= 18
    c = [300, 500]
    A = [
        [1, 0],
        [0, 1],
        [3, 2]
    ]
    b = [4, 12, 18]
    sense = ["<=", "<=", "<="]
    lp = LinearProgram(c, A, b, sense, is_max=True)
    x_opt, z_opt, tableau, basis = simplex(lp)
    print("Solution optimale (exemple cours) :")
    print("x1* =", x_opt[0], ", x2* =", x_opt[1], ", Z* =", z_opt)
    plot_feasible_region_and_solution(lp, x_opt, z_opt)

if __name__ == "__main__":
    # Tu peux commencer par l'exemple du cours
    exemple_cours()

