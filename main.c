
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#define MAX_VARS 20
#define MAX_CONSTRAINTS 20
#define EPSILON 1e-9

#define WINDOW_WIDTH 1000
#define WINDOW_HEIGHT 700
#define GRAPH_X 50
#define GRAPH_Y 50
#define GRAPH_WIDTH 600
#define GRAPH_HEIGHT 550
#define INFO_X 680
#define INFO_Y 50

// ===================== STRUCTURES =====================

typedef struct {
    int n_vars;
    int n_constraints;
    double c[MAX_VARS];
    double A[MAX_CONSTRAINTS][MAX_VARS];
    double b[MAX_CONSTRAINTS];
    int constraint_type[MAX_CONSTRAINTS]; // 0: <=, 1: =, 2: >=
    int is_max;
} LinearProgram;

typedef struct {
    double tableau[MAX_CONSTRAINTS + 1][MAX_VARS + MAX_CONSTRAINTS + 1];
    int basis[MAX_CONSTRAINTS];
    int n_rows;
    int n_cols;
    int n_original_vars;
    int n_slack_vars;
} SimplexTableau;

typedef struct {
    double x[MAX_VARS];
    double z_optimal;
    int status;
    char message[256];
} Solution;

typedef struct {
    double x, y;
} Point;

// ===================== VARIABLES GLOBALES SDL =====================

SDL_Window *window = NULL;
SDL_Renderer *renderer = NULL;
TTF_Font *font = NULL;
TTF_Font *font_small = NULL;
TTF_Font *font_large = NULL;

double scale_x = 60.0;
double scale_y = 60.0;
double max_x = 10.0;
double max_y = 10.0;

// ===================== FONCTIONS GRAPHIQUES =====================

void init_sdl() {
    SDL_Init(SDL_INIT_VIDEO);
    TTF_Init();
    
    window = SDL_CreateWindow(
        "Solveur Programmation Lineaire - Methode du Simplexe",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN
    );
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    
    // Charger les polices (utiliser une police système)
    font = TTF_OpenFont("/usr/share/fonts/TTF/DejaVuSans.ttf", 14);
    if (!font) font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14);
    if (!font) font = TTF_OpenFont("/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 14);
    
    font_small = TTF_OpenFont("/usr/share/fonts/TTF/DejaVuSans.ttf", 11);
    if (!font_small) font_small = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11);
    if (!font_small) font_small = TTF_OpenFont("/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 11);
    
    font_large = TTF_OpenFont("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 18);
    if (!font_large) font_large = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18);
    if (!font_large) font_large = TTF_OpenFont("/usr/share/fonts/liberation/LiberationSans-Bold.ttf", 18);
}

void close_sdl() {
    if (font) TTF_CloseFont(font);
    if (font_small) TTF_CloseFont(font_small);
    if (font_large) TTF_CloseFont(font_large);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    TTF_Quit();
    SDL_Quit();
}

void draw_text(const char *text, int x, int y, SDL_Color color, TTF_Font *f) {
    if (!f) return;
    SDL_Surface *surface = TTF_RenderUTF8_Blended(f, text, color);
    if (surface) {
        SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
        SDL_Rect rect = {x, y, surface->w, surface->h};
        SDL_RenderCopy(renderer, texture, NULL, &rect);
        SDL_DestroyTexture(texture);
        SDL_FreeSurface(surface);
    }
}

int graph_x(double x) {
    return GRAPH_X + (int)(x * scale_x);
}

int graph_y(double y) {
    return GRAPH_Y + GRAPH_HEIGHT - (int)(y * scale_y);
}

void draw_thick_line(int x1, int y1, int x2, int y2, int thickness) {
    for (int i = -thickness/2; i <= thickness/2; i++) {
        for (int j = -thickness/2; j <= thickness/2; j++) {
            SDL_RenderDrawLine(renderer, x1+i, y1+j, x2+i, y2+j);
        }
    }
}

void draw_filled_circle(int cx, int cy, int radius) {
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            if (x*x + y*y <= radius*radius) {
                SDL_RenderDrawPoint(renderer, cx + x, cy + y);
            }
        }
    }
}

// ===================== CALCUL REGION ADMISSIBLE =====================

int line_intersection(double a1, double b1, double c1, double a2, double b2, double c2, double *x, double *y) {
    double det = a1 * b2 - a2 * b1;
    if (fabs(det) < EPSILON) return 0;
    *x = (c1 * b2 - c2 * b1) / det;
    *y = (a1 * c2 - a2 * c1) / det;
    return 1;
}

int is_feasible(LinearProgram *lp, double x, double y) {
    if (x < -EPSILON || y < -EPSILON) return 0;
    for (int i = 0; i < lp->n_constraints; i++) {
        double val = lp->A[i][0] * x + lp->A[i][1] * y;
        if (lp->constraint_type[i] == 0 && val > lp->b[i] + EPSILON) return 0;
        if (lp->constraint_type[i] == 2 && val < lp->b[i] - EPSILON) return 0;
        if (lp->constraint_type[i] == 1 && fabs(val - lp->b[i]) > EPSILON) return 0;
    }
    return 1;
}

int compare_angle(const void *a, const void *b, void *center) {
    Point *p1 = (Point *)a;
    Point *p2 = (Point *)b;
    Point *c = (Point *)center;
    double angle1 = atan2(p1->y - c->y, p1->x - c->x);
    double angle2 = atan2(p2->y - c->y, p2->x - c->x);
    return (angle1 > angle2) - (angle1 < angle2);
}

void find_vertices(LinearProgram *lp, Point *vertices, int *n_vertices) {
    *n_vertices = 0;
    
    // Ajouter l'origine si admissible
    if (is_feasible(lp, 0, 0)) {
        vertices[(*n_vertices)].x = 0;
        vertices[(*n_vertices)].y = 0;
        (*n_vertices)++;
    }
    
    // Intersections avec les axes
    for (int i = 0; i < lp->n_constraints; i++) {
        double a1 = lp->A[i][0], a2 = lp->A[i][1], b = lp->b[i];
        
        // Intersection avec x1 = 0
        if (fabs(a2) > EPSILON) {
            double y = b / a2;
            if (y >= 0 && is_feasible(lp, 0, y)) {
                vertices[(*n_vertices)].x = 0;
                vertices[(*n_vertices)].y = y;
                (*n_vertices)++;
            }
        }
        // Intersection avec x2 = 0
        if (fabs(a1) > EPSILON) {
            double x = b / a1;
            if (x >= 0 && is_feasible(lp, x, 0)) {
                vertices[(*n_vertices)].x = x;
                vertices[(*n_vertices)].y = 0;
                (*n_vertices)++;
            }
        }
    }
    
    // Intersections entre contraintes
    for (int i = 0; i < lp->n_constraints; i++) {
        for (int j = i + 1; j < lp->n_constraints; j++) {
            double x, y;
            if (line_intersection(lp->A[i][0], lp->A[i][1], lp->b[i],
                                  lp->A[j][0], lp->A[j][1], lp->b[j], &x, &y)) {
                if (is_feasible(lp, x, y)) {
                    vertices[(*n_vertices)].x = x;
                    vertices[(*n_vertices)].y = y;
                    (*n_vertices)++;
                }
            }
        }
    }
    
    // Trier les sommets par angle pour le polygone
    if (*n_vertices > 2) {
        Point center = {0, 0};
        for (int i = 0; i < *n_vertices; i++) {
            center.x += vertices[i].x;
            center.y += vertices[i].y;
        }
        center.x /= *n_vertices;
        center.y /= *n_vertices;
        
        // Tri bulle par angle
        for (int i = 0; i < *n_vertices - 1; i++) {
            for (int j = 0; j < *n_vertices - i - 1; j++) {
                double angle1 = atan2(vertices[j].y - center.y, vertices[j].x - center.x);
                double angle2 = atan2(vertices[j+1].y - center.y, vertices[j+1].x - center.x);
                if (angle1 > angle2) {
                    Point temp = vertices[j];
                    vertices[j] = vertices[j+1];
                    vertices[j+1] = temp;
                }
            }
        }
    }
}

void draw_filled_polygon(Point *vertices, int n_vertices, SDL_Color color) {
    if (n_vertices < 3) return;
    
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
    
    // Remplissage par scanline
    int min_y = graph_y(vertices[0].y), max_y = graph_y(vertices[0].y);
    for (int i = 1; i < n_vertices; i++) {
        int gy = graph_y(vertices[i].y);
        if (gy < min_y) min_y = gy;
        if (gy > max_y) max_y = gy;
    }
    
    for (int y = min_y; y <= max_y; y++) {
        int x_intersections[100];
        int n_intersections = 0;
        
        for (int i = 0; i < n_vertices; i++) {
            int j = (i + 1) % n_vertices;
            int y1 = graph_y(vertices[i].y);
            int y2 = graph_y(vertices[j].y);
            int x1 = graph_x(vertices[i].x);
            int x2 = graph_x(vertices[j].x);
            
            if ((y1 <= y && y2 > y) || (y2 <= y && y1 > y)) {
                int x = x1 + (y - y1) * (x2 - x1) / (y2 - y1);
                x_intersections[n_intersections++] = x;
            }
        }
        
        // Trier les intersections
        for (int i = 0; i < n_intersections - 1; i++) {
            for (int j = 0; j < n_intersections - i - 1; j++) {
                if (x_intersections[j] > x_intersections[j+1]) {
                    int temp = x_intersections[j];
                    x_intersections[j] = x_intersections[j+1];
                    x_intersections[j+1] = temp;
                }
            }
        }
        
        // Dessiner les segments horizontaux
        for (int i = 0; i < n_intersections - 1; i += 2) {
            SDL_RenderDrawLine(renderer, x_intersections[i], y, x_intersections[i+1], y);
        }
    }
}

// ===================== SIMPLEX (identique) =====================

void init_tableau(SimplexTableau *tab, LinearProgram *lp) {
    int i, j;
    int slack_count = 0;
    
    tab->n_original_vars = lp->n_vars;
    tab->n_rows = lp->n_constraints;
    
    for (i = 0; i < lp->n_constraints; i++) {
        if (lp->constraint_type[i] != 1) slack_count++;
    }
    tab->n_slack_vars = slack_count;
    tab->n_cols = lp->n_vars + slack_count + 1;
    
    memset(tab->tableau, 0, sizeof(tab->tableau));
    
    slack_count = 0;
    for (i = 0; i < lp->n_constraints; i++) {
        for (j = 0; j < lp->n_vars; j++) {
            tab->tableau[i][j] = lp->A[i][j];
        }
        
        if (lp->constraint_type[i] == 0) {
            tab->tableau[i][lp->n_vars + slack_count] = 1.0;
            tab->basis[i] = lp->n_vars + slack_count;
            slack_count++;
        } else if (lp->constraint_type[i] == 2) {
            tab->tableau[i][lp->n_vars + slack_count] = -1.0;
            slack_count++;
            tab->basis[i] = -1;
        } else {
            tab->basis[i] = -1;
        }
        
        tab->tableau[i][tab->n_cols - 1] = lp->b[i];
    }
    
    for (j = 0; j < lp->n_vars; j++) {
        tab->tableau[tab->n_rows][j] = lp->is_max ? -lp->c[j] : lp->c[j];
    }
}

int find_pivot_column(SimplexTableau *tab) {
    int pivot_col = -1;
    double min_val = -EPSILON;
    for (int j = 0; j < tab->n_cols - 1; j++) {
        if (tab->tableau[tab->n_rows][j] < min_val) {
            min_val = tab->tableau[tab->n_rows][j];
            pivot_col = j;
        }
    }
    return pivot_col;
}

int find_pivot_row(SimplexTableau *tab, int pivot_col) {
    int pivot_row = -1;
    double min_ratio = 1e20;
    for (int i = 0; i < tab->n_rows; i++) {
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
    double pivot_val = tab->tableau[pivot_row][pivot_col];
    
    for (int j = 0; j < tab->n_cols; j++) {
        tab->tableau[pivot_row][j] /= pivot_val;
    }
    
    for (int i = 0; i <= tab->n_rows; i++) {
        if (i != pivot_row) {
            double factor = tab->tableau[i][pivot_col];
            for (int j = 0; j < tab->n_cols; j++) {
                tab->tableau[i][j] -= factor * tab->tableau[pivot_row][j];
            }
        }
    }
    
    tab->basis[pivot_row] = pivot_col;
}

Solution solve_simplex(LinearProgram *lp) {
    SimplexTableau tab;
    Solution sol;
    int iteration = 0;
    
    memset(&sol, 0, sizeof(sol));
    init_tableau(&tab, lp);
    
    printf("\\n========== RESOLUTION PAR METHODE DU SIMPLEXE ==========\\n");
    
    while (1) {
        int pivot_col = find_pivot_column(&tab);
        if (pivot_col == -1) break;
        
        int pivot_row = find_pivot_row(&tab, pivot_col);
        if (pivot_row == -1) {
            sol.status = 1;
            strcpy(sol.message, "Probleme non borne");
            return sol;
        }
        
        pivot_operation(&tab, pivot_row, pivot_col);
        iteration++;
        
        if (iteration > 100) {
            sol.status = 2;
            strcpy(sol.message, "Max iterations");
            return sol;
        }
    }
    
    sol.status = 0;
    strcpy(sol.message, "Solution optimale trouvee");
    
    for (int i = 0; i < tab.n_rows; i++) {
        if (tab.basis[i] < lp->n_vars && tab.basis[i] >= 0) {
            sol.x[tab.basis[i]] = tab.tableau[i][tab.n_cols - 1];
        }
    }
    
    sol.z_optimal = -tab.tableau[tab.n_rows][tab.n_cols - 1];
    if (!lp->is_max) sol.z_optimal = -sol.z_optimal;
    
    printf("Iterations: %d\\n", iteration);
    printf("Solution: x1 = %.4f, x2 = %.4f\\n", sol.x[0], sol.x[1]);
    printf("Z optimal = %.4f\\n", sol.z_optimal);
    
    return sol;
}

// ===================== AFFICHAGE GRAPHIQUE COMPLET =====================

void render_graph(LinearProgram *lp, Solution *sol) {
    SDL_Color white = {255, 255, 255, 255};
    SDL_Color black = {0, 0, 0, 255};
    SDL_Color gray = {200, 200, 200, 255};
    SDL_Color light_blue = {173, 216, 230, 150};
    SDL_Color red = {220, 50, 50, 255};
    SDL_Color green = {50, 180, 50, 255};
    SDL_Color dark_blue = {50, 50, 150, 255};
    SDL_Color colors[] = {
        {220, 80, 80, 255},   // Rouge
        {80, 80, 220, 255},   // Bleu
        {80, 180, 80, 255},   // Vert
        {200, 150, 50, 255},  // Orange
        {150, 50, 200, 255}   // Violet
    };
    
    // Fond
    SDL_SetRenderDrawColor(renderer, 245, 245, 250, 255);
    SDL_RenderClear(renderer);
    
    // Cadre du graphique
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_Rect graph_rect = {GRAPH_X - 5, GRAPH_Y - 5, GRAPH_WIDTH + 10, GRAPH_HEIGHT + 10};
    SDL_RenderFillRect(renderer, &graph_rect);
    
    SDL_SetRenderDrawColor(renderer, 100, 100, 100, 255);
    SDL_RenderDrawRect(renderer, &graph_rect);
    
    // Grille
    SDL_SetRenderDrawColor(renderer, 230, 230, 230, 255);
    for (int i = 0; i <= (int)max_x; i++) {
        int x = graph_x(i);
        SDL_RenderDrawLine(renderer, x, GRAPH_Y, x, GRAPH_Y + GRAPH_HEIGHT);
    }
    for (int i = 0; i <= (int)max_y; i++) {
        int y = graph_y(i);
        SDL_RenderDrawLine(renderer, GRAPH_X, y, GRAPH_X + GRAPH_WIDTH, y);
    }
    
    // Région admissible (polygone rempli)
    Point vertices[50];
    int n_vertices;
    find_vertices(lp, vertices, &n_vertices);
    
    if (n_vertices >= 3) {
        draw_filled_polygon(vertices, n_vertices, light_blue);
        
        // Contour du polygone
        SDL_SetRenderDrawColor(renderer, 100, 150, 200, 255);
        for (int i = 0; i < n_vertices; i++) {
            int j = (i + 1) % n_vertices;
            draw_thick_line(graph_x(vertices[i].x), graph_y(vertices[i].y),
                           graph_x(vertices[j].x), graph_y(vertices[j].y), 2);
        }
    }
    
    // Contraintes (lignes)
    for (int i = 0; i < lp->n_constraints; i++) {
        double a1 = lp->A[i][0], a2 = lp->A[i][1], b = lp->b[i];
        SDL_Color col = colors[i % 5];
        SDL_SetRenderDrawColor(renderer, col.r, col.g, col.b, 255);
        
        double x1, y1, x2, y2;
        if (fabs(a2) > EPSILON && fabs(a1) > EPSILON) {
            x1 = 0; y1 = b / a2;
            x2 = b / a1; y2 = 0;
        } else if (fabs(a1) < EPSILON) {
            x1 = 0; y1 = b / a2;
            x2 = max_x; y2 = b / a2;
        } else {
            x1 = b / a1; y1 = 0;
            x2 = b / a1; y2 = max_y;
        }
        
        draw_thick_line(graph_x(x1), graph_y(y1), graph_x(x2), graph_y(y2), 2);
        
        // Label de la contrainte
        char label[64];
        snprintf(label, sizeof(label), "C%d", i + 1);
        int lx = (graph_x(x1) + graph_x(x2)) / 2 + 5;
        int ly = (graph_y(y1) + graph_y(y2)) / 2 - 15;
        draw_text(label, lx, ly, col, font_small);
    }
    
    // Lignes iso-profit (fonction objectif)
    SDL_SetRenderDrawColor(renderer, 50, 200, 50, 100);
    double c1 = lp->c[0], c2 = lp->c[1];
    for (double z = 0; z <= sol->z_optimal * 1.5; z += sol->z_optimal / 5) {
        if (fabs(c1) > EPSILON && fabs(c2) > EPSILON) {
            double x1 = 0, y1 = z / c2;
            double x2 = z / c1, y2 = 0;
            if (y1 <= max_y && x2 <= max_x) {
                SDL_RenderDrawLine(renderer, graph_x(x1), graph_y(y1), graph_x(x2), graph_y(y2));
            }
        }
    }
    
    // Ligne iso-profit optimale (plus épaisse, verte)
    SDL_SetRenderDrawColor(renderer, 0, 180, 0, 255);
    if (fabs(c1) > EPSILON && fabs(c2) > EPSILON) {
        double z = sol->z_optimal;
        double x1 = 0, y1 = z / c2;
        double x2 = z / c1, y2 = 0;
        draw_thick_line(graph_x(x1), graph_y(y1), graph_x(x2), graph_y(y2), 3);
    }
    
    // Axes
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    draw_thick_line(GRAPH_X, graph_y(0), GRAPH_X + GRAPH_WIDTH, graph_y(0), 2); // Axe X
    draw_thick_line(graph_x(0), GRAPH_Y, graph_x(0), GRAPH_Y + GRAPH_HEIGHT, 2); // Axe Y
    
    // Graduations et labels des axes
    for (int i = 0; i <= (int)max_x; i += 1) {
        int x = graph_x(i);
        SDL_RenderDrawLine(renderer, x, graph_y(0) - 5, x, graph_y(0) + 5);
        char label[16];
        snprintf(label, sizeof(label), "%d", i);
        draw_text(label, x - 5, graph_y(0) + 8, black, font_small);
    }
    for (int i = 0; i <= (int)max_y; i += 1) {
        int y = graph_y(i);
        SDL_RenderDrawLine(renderer, graph_x(0) - 5, y, graph_x(0) + 5, y);
        if (i > 0) {
            char label[16];
            snprintf(label, sizeof(label), "%d", i);
            draw_text(label, graph_x(0) - 25, y - 7, black, font_small);
        }
    }
    
    // Labels des axes
    draw_text("x1", GRAPH_X + GRAPH_WIDTH - 20, graph_y(0) + 20, black, font);
    draw_text("x2", graph_x(0) - 30, GRAPH_Y + 5, black, font);
    
    // Point optimal
    if (sol->status == 0) {
        SDL_SetRenderDrawColor(renderer, 220, 50, 50, 255);
        int ox = graph_x(sol->x[0]);
        int oy = graph_y(sol->x[1]);
        draw_filled_circle(ox, oy, 10);
        
        // Contour blanc
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        draw_filled_circle(ox, oy, 6);
        
        // Centre rouge
        SDL_SetRenderDrawColor(renderer, 220, 50, 50, 255);
        draw_filled_circle(ox, oy, 4);
        
        // Label du point optimal
        char opt_label[64];
        snprintf(opt_label, sizeof(opt_label), "(%.2f, %.2f)", sol->x[0], sol->x[1]);
        draw_text(opt_label, ox + 12, oy - 20, red, font);
    }
    
    // Sommets du polyèdre
    SDL_SetRenderDrawColor(renderer, 50, 50, 150, 255);
    for (int i = 0; i < n_vertices; i++) {
        if (fabs(vertices[i].x - sol->x[0]) > 0.01 || fabs(vertices[i].y - sol->x[1]) > 0.01) {
            draw_filled_circle(graph_x(vertices[i].x), graph_y(vertices[i].y), 5);
        }
    }
    
    // ==================== PANNEAU D'INFORMATION ====================
    
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_Rect info_rect = {INFO_X - 10, INFO_Y - 10, 300, 600};
    SDL_RenderFillRect(renderer, &info_rect);
    
    SDL_SetRenderDrawColor(renderer, 100, 100, 100, 255);
    SDL_RenderDrawRect(renderer, &info_rect);
    
    int y_pos = INFO_Y;
    
    // Titre
    draw_text("PROGRAMMATION LINEAIRE", INFO_X, y_pos, dark_blue, font_large);
    y_pos += 35;
    
    // Fonction objectif
    draw_text("Fonction objectif:", INFO_X, y_pos, black, font);
    y_pos += 22;
    
    char obj_str[128];
    snprintf(obj_str, sizeof(obj_str), "%s Z = %.0fx1 + %.0fx2", 
             lp->is_max ? "Max" : "Min", lp->c[0], lp->c[1]);
    draw_text(obj_str, INFO_X + 10, y_pos, dark_blue, font);
    y_pos += 35;
    
    // Contraintes
    draw_text("Contraintes:", INFO_X, y_pos, black, font);
    y_pos += 22;
    
    for (int i = 0; i < lp->n_constraints; i++) {
        char constr_str[128];
        const char *op = (lp->constraint_type[i] == 0) ? "<=" : 
                         (lp->constraint_type[i] == 1) ? "=" : ">=";
        snprintf(constr_str, sizeof(constr_str), "C%d: %.0fx1 + %.0fx2 %s %.0f",
                 i + 1, lp->A[i][0], lp->A[i][1], op, lp->b[i]);
        draw_text(constr_str, INFO_X + 10, y_pos, colors[i % 5], font_small);
        y_pos += 20;
    }
    draw_text("    x1, x2 >= 0", INFO_X + 10, y_pos, gray, font_small);
    y_pos += 40;
    
    // Solution
    draw_text("SOLUTION OPTIMALE", INFO_X, y_pos, red, font_large);
    y_pos += 30;
    
    char sol_str[64];
    snprintf(sol_str, sizeof(sol_str), "x1* = %.4f", sol->x[0]);
    draw_text(sol_str, INFO_X + 10, y_pos, black, font);
    y_pos += 22;
    
    snprintf(sol_str, sizeof(sol_str), "x2* = %.4f", sol->x[1]);
    draw_text(sol_str, INFO_X + 10, y_pos, black, font);
    y_pos += 30;
    
    snprintf(sol_str, sizeof(sol_str), "Z* = %.4f", sol->z_optimal);
    draw_text(sol_str, INFO_X + 10, y_pos, red, font_large);
    y_pos += 50;
    
    // Légende
    draw_text("LEGENDE", INFO_X, y_pos, black, font);
    y_pos += 25;
    
    SDL_SetRenderDrawColor(renderer, 173, 216, 230, 255);
    SDL_Rect legend_rect = {INFO_X + 10, y_pos, 20, 15};
    SDL_RenderFillRect(renderer, &legend_rect);
    draw_text("Region admissible", INFO_X + 40, y_pos, black, font_small);
    y_pos += 22;
    
    SDL_SetRenderDrawColor(renderer, 0, 180, 0, 255);
    SDL_RenderDrawLine(renderer, INFO_X + 10, y_pos + 7, INFO_X + 30, y_pos + 7);
    draw_text("Iso-profit optimal", INFO_X + 40, y_pos, black, font_small);
    y_pos += 22;
    
    SDL_SetRenderDrawColor(renderer, 220, 50, 50, 255);
    draw_filled_circle(INFO_X + 20, y_pos + 7, 6);
    draw_text("Point optimal", INFO_X + 40, y_pos, black, font_small);
    y_pos += 35;
    
    draw_text("Appuyez sur ESC pour quitter", INFO_X, y_pos, gray, font_small);
    
    SDL_RenderPresent(renderer);
}

void run_graphical(LinearProgram *lp, Solution *sol) {
    // Calculer l'échelle automatique
    max_x = 2;
    max_y = 2;
    for (int i = 0; i < lp->n_constraints; i++) {
        if (fabs(lp->A[i][0]) > EPSILON) {
            double x = lp->b[i] / lp->A[i][0];
            if (x > max_x) max_x = x;
        }
        if (fabs(lp->A[i][1]) > EPSILON) {
            double y = lp->b[i] / lp->A[i][1];
            if (y > max_y) max_y = y;
        }
    }
    max_x = ceil(max_x * 1.3);
    max_y = ceil(max_y * 1.3);
    
    scale_x = GRAPH_WIDTH / max_x;
    scale_y = GRAPH_HEIGHT / max_y;
    
    init_sdl();
    
    int running = 1;
    SDL_Event event;
    
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = 0;
            if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)
                running = 0;
        }
        render_graph(lp, sol);
        SDL_Delay(16);
    }
    
    close_sdl();
}

// ===================== MAIN =====================

int main(int argc, char *argv[]) {
    LinearProgram lp;
    Solution sol;
    int choice;
    
    printf("\\n╔══════════════════════════════════════════════════════════════╗\\n");
    printf("║   SOLVEUR PROGRAMMATION LINEAIRE - METHODE DU SIMPLEXE       ║\\n");
    printf("║              Avec Affichage Graphique SDL2                   ║\\n");
    printf("╚══════════════════════════════════════════════════════════════╝\\n");
    
    printf("\\n1. Exemple: Max Z = 300x1 + 500x2 (cours)\\n");
    printf("2. Exemple: Max Z = 50x1 + 60x2\\n");
    printf("3. Entrer un nouveau probleme\\n");
    printf("Choix: ");
    scanf("%d", &choice);
    
    if (choice == 1) {
        lp.n_vars = 2;
        lp.n_constraints = 3;
        lp.is_max = 1;
        lp.c[0] = 300; lp.c[1] = 500;
        lp.A[0][0] = 1; lp.A[0][1] = 0; lp.b[0] = 4;  lp.constraint_type[0] = 0;
        lp.A[1][0] = 0; lp.A[1][1] = 2; lp.b[1] = 12; lp.constraint_type[1] = 0;
        lp.A[2][0] = 3; lp.A[2][1] = 2; lp.b[2] = 18; lp.constraint_type[2] = 0;
    } else if (choice == 2) {
        lp.n_vars = 2;
        lp.n_constraints = 3;
        lp.is_max = 1;
        lp.c[0] = 50; lp.c[1] = 60;
        lp.A[0][0] = 1; lp.A[0][1] = 2; lp.b[0] = 8;  lp.constraint_type[0] = 0;
        lp.A[1][0] = 2; lp.A[1][1] = 2; lp.b[1] = 10; lp.constraint_type[1] = 0;
        lp.A[2][0] = 9; lp.A[2][1] = 4; lp.b[2] = 36; lp.constraint_type[2] = 0;
    } else {
        printf("\\nType (1=Max, 0=Min): ");
        scanf("%d", &lp.is_max);
        lp.n_vars = 2;
        
        printf("Coefficients objectif c1, c2: ");
        scanf("%lf %lf", &lp.c[0], &lp.c[1]);
        
        printf("Nombre de contraintes: ");
        scanf("%d", &lp.n_constraints);
        
        for (int i = 0; i < lp.n_constraints; i++) {
            printf("Contrainte %d (a1 a2 type[0/1/2] b): ", i + 1);
            scanf("%lf %lf %d %lf", &lp.A[i][0], &lp.A[i][1], 
                  &lp.constraint_type[i], &lp.b[i]);
        }
    }
    
    sol = solve_simplex(&lp);
    
    if (lp.n_vars == 2) {
        printf("\\nOuverture de la fenetre graphique...\\n");
        run_graphical(&lp, &sol);
    }
    
    return 0;
}


