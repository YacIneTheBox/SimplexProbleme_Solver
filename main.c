#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <SDL2/SDL.h>

#define MAX_VARS 20
#define MAX_CONSTRAINTS 20
#define EPSILON 1e-9

// Structure pour le problème LP
typedef struct {
    int n_vars;           // Nombre de variables de décision
    int n_constraints;    // Nombre de contraintes
    double c[MAX_VARS];   // Coefficients fonction objectif
    double A[MAX_CONSTRAINTS][MAX_VARS]; // Matrice des contraintes
    double b[MAX_CONSTRAINTS];           // Second membre
    int constraint_type[MAX_CONSTRAINTS]; // 0: <=, 1: =, 2: >=
    int is_max;           // 1 si maximisation, 0 si minimisation
} LinearProgram;

// Structure pour le tableau du Simplexe
typedef struct {
    double tableau[MAX_CONSTRAINTS + 1][MAX_VARS + MAX_CONSTRAINTS + 1];
    int basis[MAX_CONSTRAINTS];  // Variables de base
    int n_rows;
    int n_cols;
    int n_original_vars;
    int n_slack_vars;
} SimplexTableau;

// Structure pour la solution
typedef struct {
    double x[MAX_VARS];
    double z_optimal;
    int status; // 0: optimal, 1: unbounded, 2: infeasible, 3: infinite solutions
    char message[256];
} Solution;

// Prototypes
void init_tableau(SimplexTableau *tab, LinearProgram *lp);
int simplex_iteration(SimplexTableau *tab);
Solution solve_simplex(LinearProgram *lp);
void print_tableau(SimplexTableau *tab, int iteration);
void display_graphical(LinearProgram *lp, Solution *sol);

// ===================== IMPLEMENTATION DU SIMPLEXE =====================

void init_tableau(SimplexTableau *tab, LinearProgram *lp) {
    int i, j;
    int slack_count = 0;
    
    tab->n_original_vars = lp->n_vars;
    tab->n_rows = lp->n_constraints;
    
    // Compter les variables d'écart nécessaires
    for (i = 0; i < lp->n_constraints; i++) {
        if (lp->constraint_type[i] != 1) slack_count++;
    }
    tab->n_slack_vars = slack_count;
    tab->n_cols = lp->n_vars + slack_count + 1; // +1 pour le second membre
    
    // Initialiser le tableau à zéro
    memset(tab->tableau, 0, sizeof(tab->tableau));
    
    // Remplir la matrice des contraintes
    slack_count = 0;
    for (i = 0; i < lp->n_constraints; i++) {
        // Coefficients des variables originales
        for (j = 0; j < lp->n_vars; j++) {
            tab->tableau[i][j] = lp->A[i][j];
        }
        
        // Variables d'écart selon le type de contrainte
        if (lp->constraint_type[i] == 0) { // <=
            tab->tableau[i][lp->n_vars + slack_count] = 1.0;
            tab->basis[i] = lp->n_vars + slack_count;
            slack_count++;
        } else if (lp->constraint_type[i] == 2) { // >=
            tab->tableau[i][lp->n_vars + slack_count] = -1.0;
            slack_count++;
            tab->basis[i] = -1; // Nécessite variable artificielle
        } else { // =
            tab->basis[i] = -1; // Nécessite variable artificielle
        }
        
        // Second membre
        tab->tableau[i][tab->n_cols - 1] = lp->b[i];
    }
    
    // Ligne de la fonction objectif (dernière ligne)
    // Pour Max Z = cx, on met -c dans la dernière ligne
    for (j = 0; j < lp->n_vars; j++) {
        if (lp->is_max) {
            tab->tableau[tab->n_rows][j] = -lp->c[j];
        } else {
            tab->tableau[tab->n_rows][j] = lp->c[j];
        }
    }
}

int find_pivot_column(SimplexTableau *tab) {
    int j, pivot_col = -1;
    double min_val = -EPSILON;
    
    // Trouver le plus petit coefficient négatif (règle de Dantzig)
    for (j = 0; j < tab->n_cols - 1; j++) {
        if (tab->tableau[tab->n_rows][j] < min_val) {
            min_val = tab->tableau[tab->n_rows][j];
            pivot_col = j;
        }
    }
    return pivot_col;
}

int find_pivot_row(SimplexTableau *tab, int pivot_col) {
    int i, pivot_row = -1;
    double min_ratio = 1e20;
    
    // Règle du ratio minimum: min{bi/air | air > 0}
    for (i = 0; i < tab->n_rows; i++) {
        if (tab->tableau[i][pivot_col] > EPSILON) {
            double ratio = tab->tableau[i][tab->n_cols - 1] / tab->tableau[i][pivot_col];
            if (ratio < min_ratio && ratio >= 0) {
                min_ratio = ratio;
                pivot_row = i;
            }
        }
    }
    return pivot_row;
}

void pivot_operation(SimplexTableau *tab, int pivot_row, int pivot_col) {
    int i, j;
    double pivot_val = tab->tableau[pivot_row][pivot_col];
    
    // Diviser la ligne pivot par l'élément pivot
    for (j = 0; j < tab->n_cols; j++) {
        tab->tableau[pivot_row][j] /= pivot_val;
    }
    
    // Éliminer les autres éléments de la colonne pivot
    for (i = 0; i <= tab->n_rows; i++) {
        if (i != pivot_row) {
            double factor = tab->tableau[i][pivot_col];
            for (j = 0; j < tab->n_cols; j++) {
                tab->tableau[i][j] -= factor * tab->tableau[pivot_row][j];
            }
        }
    }
    
    // Mettre à jour la base
    tab->basis[pivot_row] = pivot_col;
}

int simplex_iteration(SimplexTableau *tab) {
    int pivot_col = find_pivot_column(tab);
    
    // Si pas de coefficient négatif, solution optimale trouvée
    if (pivot_col == -1) return 0; // Optimal
    
    int pivot_row = find_pivot_row(tab, pivot_col);
    
    // Si pas de ligne pivot valide, problème non borné
    if (pivot_row == -1) return 1; // Unbounded
    
    // Effectuer l'opération de pivot
    pivot_operation(tab, pivot_row, pivot_col);
    
    return 2; // Continuer
}

Solution solve_simplex(LinearProgram *lp) {
    SimplexTableau tab;
    Solution sol;
    int iteration = 0;
    int status;
    
    memset(&sol, 0, sizeof(sol));
    init_tableau(&tab, lp);
    
    printf("\\n========== RESOLUTION PAR METHODE DU SIMPLEXE ==========\\n");
    print_tableau(&tab, iteration);
    
    // Itérations du Simplexe
    while ((status = simplex_iteration(&tab)) == 2) {
        iteration++;
        print_tableau(&tab, iteration);
        
        if (iteration > 100) {
            sol.status = 2;
            strcpy(sol.message, "Nombre maximum d'iterations atteint");
            return sol;
        }
    }
    
    if (status == 1) {
        sol.status = 1;
        strcpy(sol.message, "Probleme non borne (solution infinie)");
        return sol;
    }
    
    // Extraire la solution
    sol.status = 0;
    strcpy(sol.message, "Solution optimale trouvee");
    
    // Récupérer les valeurs des variables de base
    for (int i = 0; i < tab.n_rows; i++) {
        if (tab.basis[i] < lp->n_vars && tab.basis[i] >= 0) {
            sol.x[tab.basis[i]] = tab.tableau[i][tab.n_cols - 1];
        }
    }
    
    // Valeur optimale de Z
    sol.z_optimal = -tab.tableau[tab.n_rows][tab.n_cols - 1];
    if (!lp->is_max) sol.z_optimal = -sol.z_optimal;
    
    // Vérifier solutions infinies (Delta_j = 0 pour variable hors base)
    for (int j = 0; j < tab.n_cols - 1; j++) {
        if (fabs(tab.tableau[tab.n_rows][j]) < EPSILON) {
            int is_basic = 0;
            for (int i = 0; i < tab.n_rows; i++) {
                if (tab.basis[i] == j) { is_basic = 1; break; }
            }
            if (!is_basic && j < lp->n_vars) {
                sol.status = 3;
                strcpy(sol.message, "Solutions optimales multiples");
            }
        }
    }
    
    return sol;
}

void print_tableau(SimplexTableau *tab, int iteration) {
    printf("\\n--- Tableau Simplexe (Iteration %d) ---\\n", iteration);
    printf("Base\\t");
    for (int j = 0; j < tab->n_cols - 1; j++) {
        if (j < tab->n_original_vars)
            printf("x%d\\t", j + 1);
        else
            printf("e%d\\t", j - tab->n_original_vars + 1);
    }
    printf("b\\n");
    printf("---------------------------------------------------------------\\n");
    
    for (int i = 0; i < tab->n_rows; i++) {
        if (tab->basis[i] < tab->n_original_vars)
            printf("x%d\\t", tab->basis[i] + 1);
        else
            printf("e%d\\t", tab->basis[i] - tab->n_original_vars + 1);
        
        for (int j = 0; j < tab->n_cols; j++) {
            printf("%.2f\\t", tab->tableau[i][j]);
        }
        printf("\\n");
    }
    
    printf("Delta\\t");
    for (int j = 0; j < tab->n_cols; j++) {
        printf("%.2f\\t", tab->tableau[tab->n_rows][j]);
    }
    printf("\\n");
}

// ===================== AFFICHAGE GRAPHIQUE SDL2 =====================

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define MARGIN 60
#define SCALE 40

void draw_line(SDL_Renderer *renderer, int x1, int y1, int x2, int y2) {
    SDL_RenderDrawLine(renderer, x1, y1, x2, y2);
}

void draw_text(SDL_Renderer *renderer, const char *text, int x, int y) {
    // Note: SDL2 simple n'a pas de support texte intégré
    // Pour un vrai texte, utiliser SDL_ttf
    // Ici on dessine juste un marqueur
    SDL_Rect rect = {x - 2, y - 2, 4, 4};
    SDL_RenderFillRect(renderer, &rect);
}

void display_graphical(LinearProgram *lp, Solution *sol) {
    if (lp->n_vars != 2) {
        printf("\\nAffichage graphique disponible uniquement pour 2 variables.\\n");
        return;
    }
    
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window = SDL_CreateWindow(
        "Programmation Lineaire - Region Admissible",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN
    );
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    
    int running = 1;
    SDL_Event event;
    
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = 0;
            if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)
                running = 0;
        }
        
        // Fond blanc
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        SDL_RenderClear(renderer);
        
        int origin_x = MARGIN;
        int origin_y = WINDOW_HEIGHT - MARGIN;
        
        // Axes (noir)
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        draw_line(renderer, origin_x, origin_y, WINDOW_WIDTH - MARGIN, origin_y); // Axe X
        draw_line(renderer, origin_x, origin_y, origin_x, MARGIN);                  // Axe Y
        
        // Graduations
        for (int i = 0; i <= 15; i++) {
            int px = origin_x + i * SCALE;
            int py = origin_y - i * SCALE;
            draw_line(renderer, px, origin_y - 5, px, origin_y + 5);
            draw_line(renderer, origin_x - 5, py, origin_x + 5, py);
        }
        
        // Dessiner les contraintes (lignes)
        for (int i = 0; i < lp->n_constraints; i++) {
            double a1 = lp->A[i][0];
            double a2 = lp->A[i][1];
            double b = lp->b[i];
            
            // Couleurs différentes pour chaque contrainte
            SDL_SetRenderDrawColor(renderer, 
                100 + (i * 50) % 155, 
                50 + (i * 30) % 200, 
                150 - (i * 40) % 150, 255);
            
            // Calculer les points d'intersection avec les axes
            int x1, y1, x2, y2;
            
            if (fabs(a2) > EPSILON && fabs(a1) > EPSILON) {
                // Intersection avec x1 = 0: a2*x2 = b => x2 = b/a2
                x1 = origin_x;
                y1 = origin_y - (int)((b / a2) * SCALE);
                // Intersection avec x2 = 0: a1*x1 = b => x1 = b/a1
                x2 = origin_x + (int)((b / a1) * SCALE);
                y2 = origin_y;
            } else if (fabs(a1) < EPSILON) {
                // Ligne horizontale: x2 = b/a2
                x1 = origin_x;
                y1 = origin_y - (int)((b / a2) * SCALE);
                x2 = WINDOW_WIDTH - MARGIN;
                y2 = y1;
            } else {
                // Ligne verticale: x1 = b/a1
                x1 = origin_x + (int)((b / a1) * SCALE);
                y1 = origin_y;
                x2 = x1;
                y2 = MARGIN;
            }
            
            draw_line(renderer, x1, y1, x2, y2);
        }
        
        // Dessiner le point optimal (rouge)
        if (sol->status == 0 || sol->status == 3) {
            SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
            int opt_x = origin_x + (int)(sol->x[0] * SCALE);
            int opt_y = origin_y - (int)(sol->x[1] * SCALE);
            
            // Cercle autour du point optimal
            for (int r = 5; r <= 10; r++) {
                for (int angle = 0; angle < 360; angle += 5) {
                    int px = opt_x + (int)(r * cos(angle * M_PI / 180));
                    int py = opt_y + (int)(r * sin(angle * M_PI / 180));
                    SDL_RenderDrawPoint(renderer, px, py);
                }
            }
            
            // Remplir le point
            SDL_Rect point = {opt_x - 5, opt_y - 5, 10, 10};
            SDL_RenderFillRect(renderer, &point);
        }
        
        // Dessiner la fonction objectif (vert, ligne iso-profit passant par optimal)
        SDL_SetRenderDrawColor(renderer, 0, 200, 0, 255);
        double z_val = sol->z_optimal;
        double c1 = lp->c[0], c2 = lp->c[1];
        
        if (fabs(c2) > EPSILON) {
            int x1 = origin_x;
            int y1 = origin_y - (int)((z_val / c2) * SCALE);
            int x2 = origin_x + (int)((z_val / c1) * SCALE);
            int y2 = origin_y;
            
            // Dessiner plusieurs lignes iso-profit
            for (int k = -2; k <= 2; k++) {
                double z_k = z_val + k * 50;
                if (z_k >= 0) {
                    int y1k = origin_y - (int)((z_k / c2) * SCALE);
                    int x2k = origin_x + (int)((z_k / c1) * SCALE);
                    SDL_SetRenderDrawColor(renderer, 0, 150 + k * 20, 0, 100);
                    draw_line(renderer, origin_x, y1k, x2k, origin_y);
                }
            }
        }
        
        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }
    
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

// ===================== FONCTION PRINCIPALE =====================

int main() {
    LinearProgram lp;
    Solution sol;
    int choice;
    
    printf("╔══════════════════════════════════════════════════════════════╗\\n");
    printf("║    SOLVEUR DE PROGRAMMATION LINEAIRE - METHODE SIMPLEXE      ║\\n");
    printf("║           Avec Affichage Graphique SDL2                      ║\\n");
    printf("╚══════════════════════════════════════════════════════════════╝\\n");
    
    printf("\\n1. Exemple du cours (Max Z = 300x1 + 500x2)\\n");
    printf("2. Entrer un nouveau probleme\\n");
    printf("Choix: ");
    scanf("%d", &choice);
    
    if (choice == 1) {
        // Exemple du cours LP-1_3.pdf
        // Max Z = 300x1 + 500x2
        // x1 <= 4
        // 2x2 <= 12 (équivalent à x2 <= 6)
        // 3x1 + 2x2 <= 18
        // x1, x2 >= 0
        
        lp.n_vars = 2;
        lp.n_constraints = 3;
        lp.is_max = 1;
        
        lp.c[0] = 300; lp.c[1] = 500;
        
        lp.A[0][0] = 1;  lp.A[0][1] = 0;  lp.b[0] = 4;   lp.constraint_type[0] = 0;
        lp.A[1][0] = 0;  lp.A[1][1] = 2;  lp.b[1] = 12;  lp.constraint_type[1] = 0;
        lp.A[2][0] = 3;  lp.A[2][1] = 2;  lp.b[2] = 18;  lp.constraint_type[2] = 0;
        
    } else {
        // Saisie utilisateur
        printf("\\nType de probleme (1=Max, 0=Min): ");
        scanf("%d", &lp.is_max);
        
        printf("Nombre de variables de decision: ");
        scanf("%d", &lp.n_vars);
        
        printf("Coefficients de la fonction objectif:\\n");
        for (int j = 0; j < lp.n_vars; j++) {
            printf("  c%d = ", j + 1);
            scanf("%lf", &lp.c[j]);
        }
        
        printf("Nombre de contraintes: ");
        scanf("%d", &lp.n_constraints);
        
        for (int i = 0; i < lp.n_constraints; i++) {
            printf("\\nContrainte %d:\\n", i + 1);
            for (int j = 0; j < lp.n_vars; j++) {
                printf("  a%d%d = ", i + 1, j + 1);
                scanf("%lf", &lp.A[i][j]);
            }
            printf("  Type (0: <=, 1: =, 2: >=): ");
            scanf("%d", &lp.constraint_type[i]);
            printf("  b%d = ", i + 1);
            scanf("%lf", &lp.b[i]);
        }
    }
    
    // Afficher le problème
    printf("\\n==================== PROBLEME ====================\\n");
    printf("%s Z = ", lp.is_max ? "Max" : "Min");
    for (int j = 0; j < lp.n_vars; j++) {
        printf("%.0fx%d", lp.c[j], j + 1);
        if (j < lp.n_vars - 1) printf(" + ");
    }
    printf("\\n\\nSous les contraintes:\\n");
    for (int i = 0; i < lp.n_constraints; i++) {
        printf("  ");
        for (int j = 0; j < lp.n_vars; j++) {
            printf("%.0fx%d", lp.A[i][j], j + 1);
            if (j < lp.n_vars - 1) printf(" + ");
        }
        char *op = (lp.constraint_type[i] == 0) ? "<=" : 
                   (lp.constraint_type[i] == 1) ? "=" : ">=";
        printf(" %s %.0f\\n", op, lp.b[i]);
    }
    printf("  x1, x2 >= 0\\n");
    
    // Résoudre
    sol = solve_simplex(&lp);
    
    // Afficher la solution
    printf("\\n==================== SOLUTION ====================\\n");
    printf("Status: %s\\n", sol.message);
    
    if (sol.status == 0 || sol.status == 3) {
        printf("\\nSolution optimale:\\n");
        for (int j = 0; j < lp.n_vars; j++) {
            printf("  x%d = %.4f\\n", j + 1, sol.x[j]);
        }
        printf("\\nValeur optimale: Z = %.4f\\n", sol.z_optimal);
    }
    
    // Affichage graphique (seulement pour 2 variables)
    if (lp.n_vars == 2) {
        printf("\\nAppuyez sur Echap pour fermer la fenetre graphique...\\n");
        display_graphical(&lp, &sol);
    }
    
    return 0;
}


