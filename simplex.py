import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-9

class LinearProgram:
    def __init__(self, c, A, b, sense, is_max=True):
        """
        c : vecteur des coefficients de l'objectif
        A : matrice des contraintes
        b : second membre
        sense : liste de "<=", ">=", "=" pour chaque contrainte
        is_max : True pour Max, False pour Min
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.sense = sense
        self.is_max = is_max
        self.n_vars = len(c)
        self.n_constraints = len(b)

def to_standard_form(lp):
    """Convertit en forme standard avec variables d'écart"""
    A = lp.A.copy()
    b = lp.b.copy()
    c = lp.c.copy()

    if not lp.is_max:
        c = -c

    m, n = A.shape

    # Compter d'abord le nombre de variables d'écart nécessaires
    n_slack = 0
    for i in range(m):
        if lp.sense[i] in ["<=", ">="]:
            n_slack += 1

    # Créer la nouvelle matrice avec la bonne taille
    new_A = np.zeros((m, n + n_slack))
    new_A[:, :n] = A
    new_b = b.copy()

    # Ajouter les variables d'écart
    slack_idx = 0
    slack_cols = []
    for i in range(m):
        if lp.sense[i] == "<=":
            new_A[i, n + slack_idx] = 1.0
            slack_cols.append(n + slack_idx)
            slack_idx += 1
        elif lp.sense[i] == ">=":
            # Transformer >= en <= en multipliant par -1
            new_A[i, :n] = -A[i]
            new_A[i, n + slack_idx] = 1.0
            new_b[i] = -b[i]
            slack_cols.append(n + slack_idx)
            slack_idx += 1
        # Pour "=", pas de variable d'écart

    # Étendre le vecteur c avec des 0 pour les variables d'écart
    new_c = np.zeros(n + n_slack)
    new_c[:n] = c

    return new_c, new_A, new_b, slack_cols

def build_tableau(c, A, b):
    """Construit le tableau du simplexe"""
    m, n = A.shape
    tableau = np.zeros((m + 1, n + 1))
    tableau[:m, :n] = A
    tableau[:m, -1] = b
    tableau[-1, :n] = -c  # -c pour Max
    tableau[-1, -1] = 0.0
    return tableau

def can_improve(tableau):
    """Vérifie s'il y a encore amélioration possible"""
    last_row = tableau[-1, :-1]
    return np.any(last_row < -EPS)

def choose_entering_variable(tableau):
    """Règle de Dantzig: choisir la colonne avec le plus petit coefficient"""
    last_row = tableau[-1, :-1]
    j = np.argmin(last_row)
    if last_row[j] >= -EPS:
        return None
    return j

def choose_leaving_variable(tableau, entering_col):
    """Règle du ratio minimum"""
    m = tableau.shape[0] - 1
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
    """Opération de pivot"""
    pivot_val = tableau[row, col]
    tableau[row, :] /= pivot_val

    m, n = tableau.shape
    for i in range(m):
        if i != row:
            factor = tableau[i, col]
            tableau[i, :] -= factor * tableau[row, :]

def simplex(lp):
    """Algorithme du simplexe"""
    c_std, A_std, b_std, slack_cols = to_standard_form(lp)
    tableau = build_tableau(c_std, A_std, b_std)

    m_constraints = A_std.shape[0]
    n_vars_total = A_std.shape[1]

    # Variables de base initiales = variables d'écart
    basis = slack_cols.copy()

    iterations = 0
    print("\n========== RESOLUTION PAR METHODE DU SIMPLEXE ==========")
    print(f"Iteration {iterations}:")
    print_tableau(tableau, basis, lp.n_vars)

    while can_improve(tableau):
        col = choose_entering_variable(tableau)
        if col is None:
            break

        row = choose_leaving_variable(tableau, col)
        if row is None:
            raise ValueError("Probleme non borne (unbounded)")

        print(f"\nVariable x{col+1} entre en base, variable en ligne {row+1} sort")
        pivot(tableau, row, col)
        basis[row] = col

        iterations += 1
        print(f"\nIteration {iterations}:")
        print_tableau(tableau, basis, lp.n_vars)

        if iterations > 100:
            raise RuntimeError("Trop d'iterations")

    # Extraire la solution
    x = np.zeros(n_vars_total)
    for i, col in enumerate(basis):
        if 0 <= col < n_vars_total:
            x[col] = tableau[i, -1]

    z = -tableau[-1, -1]
    x_original = x[:lp.n_vars]

    if not lp.is_max:
        z = -z

    return x_original, z, tableau, basis

def print_tableau(tableau, basis, n_orig_vars):
    """Affiche le tableau du simplexe"""
    m, n = tableau.shape
    m -= 1  # Sans la ligne objectif

    print("\nTableau:")
    print("Base\t", end="")
    for j in range(n-1):
        if j < n_orig_vars:
            print(f"x{j+1}\t", end="")
        else:
            print(f"e{j-n_orig_vars+1}\t", end="")
    print("b")
    print("-" * 60)

    for i in range(m):
        if basis[i] < n_orig_vars:
            print(f"x{basis[i]+1}\t", end="")
        else:
            print(f"e{basis[i]-n_orig_vars+1}\t", end="")
        for j in range(n):
            print(f"{tableau[i,j]:.2f}\t", end="")
        print()

    print("Delta\t", end="")
    for j in range(n):
        print(f"{tableau[-1,j]:.2f}\t", end="")
    print()

def plot_feasible_region_and_solution(lp, x_opt, z_opt):
    """Affichage graphique 2D"""
    if lp.n_vars != 2:
        print("Affichage graphique uniquement pour 2 variables.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Limites du graphique
    max_x1 = max(10, x_opt[0] * 1.5 if x_opt[0] > 0 else 10)
    max_x2 = max(10, x_opt[1] * 1.5 if x_opt[1] > 0 else 10)

    x1 = np.linspace(0, max_x1, 400)

    # Dessiner les contraintes
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i in range(lp.n_constraints):
        a1, a2 = lp.A[i]
        bi = lp.b[i]
        color = colors[i % len(colors)]

        if abs(a2) > EPS:
            x2 = (bi - a1 * x1) / a2
            label = f"C{i+1}: {a1:.0f}x1 + {a2:.0f}x2 {lp.sense[i]} {bi:.0f}"
            ax.plot(x1, x2, color=color, linewidth=2, label=label)

            if lp.sense[i] == "<=":
                ax.fill_between(x1, 0, x2, where=(x2 >= 0), alpha=0.1, color=color)
        else:
            # Contrainte verticale
            x1_vert = bi / a1 if abs(a1) > EPS else 0
            label = f"C{i+1}: {a1:.0f}x1 {lp.sense[i]} {bi:.0f}"
            ax.axvline(x1_vert, color=color, linewidth=2, label=label)

    # Point optimal
    ax.plot(x_opt[0], x_opt[1], 'ro', markersize=15, label='Solution optimale', zorder=5)
    ax.annotate(f'  ({x_opt[0]:.2f}, {x_opt[1]:.2f})\n  Z* = {z_opt:.2f}',
                xy=(x_opt[0], x_opt[1]), fontsize=12, color='red',
                xytext=(10, 10), textcoords='offset points')

    # Ligne iso-profit passant par la solution
    c1, c2 = lp.c
    if abs(c2) > EPS:
        x2_iso = (z_opt - c1 * x1) / c2
        ax.plot(x1, x2_iso, 'g--', linewidth=2, alpha=0.7, label=f'Iso-profit Z = {z_opt:.2f}')

        # Quelques autres lignes iso-profit
        for k in range(-2, 3):
            if k != 0:
                z_k = z_opt + k * z_opt / 5
                if z_k > 0:
                    x2_k = (z_k - c1 * x1) / c2
                    ax.plot(x1, x2_k, 'g:', linewidth=1, alpha=0.3)

    # Axes et grille
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlim(0, max_x1)
    ax.set_ylim(0, max_x2)
    ax.set_xlabel('x1', fontsize=14)
    ax.set_ylabel('x2', fontsize=14)
    ax.set_title('Programmation Lineaire - Methode du Simplexe\nRegion admissible et solution optimale',
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ========== EXEMPLES ==========

def exemple_cours():
    """Exemple du cours LP-1_3: Max Z = 300x1 + 500x2"""
    print("\n" + "="*60)
    print("EXEMPLE DU COURS: Max Z = 300x1 + 500x2")
    print("="*60)

    c = [300, 500]
    A = [[1, 0],   # x1 <= 4
         [0, 2],   # 2x2 <= 12
         [3, 2]]   # 3x1 + 2x2 <= 18
    b = [4, 12, 18]
    sense = ["<=", "<=", "<="]

    lp = LinearProgram(c, A, b, sense, is_max=True)
    x_opt, z_opt, tableau, basis = simplex(lp)

    print("\n" + "="*60)
    print("SOLUTION OPTIMALE:")
    print("="*60)
    print(f"x1* = {x_opt[0]:.4f}")
    print(f"x2* = {x_opt[1]:.4f}")
    print(f"Z*  = {z_opt:.4f}")
    print("="*60)

    plot_feasible_region_and_solution(lp, x_opt, z_opt)

def exemple_2():
    """Exemple 2: Max Z = 50x1 + 60x2"""
    print("\n" + "="*60)
    print("EXEMPLE 2: Max Z = 50x1 + 60x2")
    print("="*60)

    c = [50, 60]
    A = [[1, 2],   # x1 + 2x2 <= 8
         [2, 2],   # 2x1 + 2x2 <= 10
         [9, 4]]   # 9x1 + 4x2 <= 36
    b = [8, 10, 36]
    sense = ["<=", "<=", "<="]

    lp = LinearProgram(c, A, b, sense, is_max=True)
    x_opt, z_opt, tableau, basis = simplex(lp)

    print("\n" + "="*60)
    print("SOLUTION OPTIMALE:")
    print("="*60)
    print(f"x1* = {x_opt[0]:.4f}")
    print(f"x2* = {x_opt[1]:.4f}")
    print(f"Z*  = {z_opt:.4f}")
    print("="*60)

    plot_feasible_region_and_solution(lp, x_opt, z_opt)

def probleme_interactif():
    """Saisie interactive d'un problème"""
    print("\n" + "="*60)
    print("SAISIE INTERACTIVE D'UN PROBLEME")
    print("="*60)

    is_max = input("Type de probleme (Max/Min): ").strip().lower() == "max"

    c1 = float(input("Coefficient c1 de la fonction objectif: "))
    c2 = float(input("Coefficient c2 de la fonction objectif: "))
    c = [c1, c2]

    n_constraints = int(input("Nombre de contraintes: "))
    A = []
    b = []
    sense = []

    for i in range(n_constraints):
        print(f"\nContrainte {i+1}:")
        a1 = float(input("  Coefficient a1: "))
        a2 = float(input("  Coefficient a2: "))
        s = input("  Sens (<=, >=, =): ").strip()
        bi = float(input("  Second membre b: "))

        A.append([a1, a2])
        b.append(bi)
        sense.append(s)

    lp = LinearProgram(c, A, b, sense, is_max=is_max)
    x_opt, z_opt, tableau, basis = simplex(lp)

    print("\n" + "="*60)
    print("SOLUTION OPTIMALE:")
    print("="*60)
    print(f"x1* = {x_opt[0]:.4f}")
    print(f"x2* = {x_opt[1]:.4f}")
    print(f"Z*  = {z_opt:.4f}")
    print("="*60)

    plot_feasible_region_and_solution(lp, x_opt, z_opt)

if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   SOLVEUR PROGRAMMATION LINEAIRE - METHODE DU SIMPLEXE       ║")
    print("║              Avec Affichage Graphique Python                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    print("\n1. Exemple du cours (Max Z = 300x1 + 500x2)")
    print("2. Exemple 2 (Max Z = 50x1 + 60x2)")
    print("3. Saisie interactive")

    choix = input("\nChoix: ")

    if choix == "1":
        exemple_cours()
    elif choix == "2":
        exemple_2()
    elif choix == "3":
        probleme_interactif()
    else:
        print("Choix invalide, execution de l'exemple du cours...")
        exemple_cours()
