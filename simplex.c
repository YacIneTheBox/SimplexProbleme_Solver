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

typedef struct { int n_vars; int n_constraints; double c[MAX_VARS]; double A[MAX_CONSTRAINTS][MAX_VARS]; double b[MAX_CONSTRAINTS]; int constraint_type[MAX_CONSTRAINTS]; int is_max; } LinearProgram;
typedef struct { double tableau[MAX_CONSTRAINTS + 1][MAX_VARS + MAX_CONSTRAINTS + 1]; int basis[MAX_CONSTRAINTS]; int n_rows, n_cols, n_original_vars, n_slack_vars; } SimplexTableau;
typedef struct { double x[MAX_VARS]; double z_optimal; int status; char message[256]; } Solution;
typedef struct { double x, y; } Point;

SDL_Window *window = NULL;
SDL_Renderer *renderer = NULL;
TTF_Font *font = NULL, *font_small = NULL, *font_large = NULL;
double scale_x = 60.0, scale_y = 60.0, max_x = 10.0, max_y = 10.0;

void init_sdl() {
    SDL_Init(SDL_INIT_VIDEO);
    TTF_Init();
    window = SDL_CreateWindow("Solveur PL - Methode Simplexe", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    font = TTF_OpenFont("/usr/share/fonts/TTF/DejaVuSans.ttf", 14);
    if (!font) font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14);
    if (!font) font = TTF_OpenFont("/usr/share/fonts/noto/NotoSans-Regular.ttf", 14);
    font_small = TTF_OpenFont("/usr/share/fonts/TTF/DejaVuSans.ttf", 11);
    if (!font_small) font_small = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11);
    font_large = TTF_OpenFont("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 18);
    if (!font_large) font_large = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18);
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

int graph_x(double x) { return GRAPH_X + (int)(x * scale_x); }
int graph_y(double y) { return GRAPH_Y + GRAPH_HEIGHT - (int)(y * scale_y); }

void draw_thick_line(int x1, int y1, int x2, int y2, int t) {
    for (int i = -t/2; i <= t/2; i++)
        for (int j = -t/2; j <= t/2; j++)
            SDL_RenderDrawLine(renderer, x1+i, y1+j, x2+i, y2+j);
}

void draw_filled_circle(int cx, int cy, int r) {
    for (int y = -r; y <= r; y++)
        for (int x = -r; x <= r; x++)
            if (x*x + y*y <= r*r) SDL_RenderDrawPoint(renderer, cx + x, cy + y);
}

int is_feasible(LinearProgram *lp, double x, double y) {
    if (x < -EPSILON || y < -EPSILON) return 0;
    for (int i = 0; i < lp->n_constraints; i++) {
        double val = lp->A[i][0] * x + lp->A[i][1] * y;
        if (lp->constraint_type[i] == 0 && val > lp->b[i] + EPSILON) return 0;
        if (lp->constraint_type[i] == 2 && val < lp->b[i] - EPSILON) return 0;
    }
    return 1;
}

int line_intersection(double a1, double b1, double c1, double a2, double b2, double c2, double *x, double *y) {
    double det = a1 * b2 - a2 * b1;
    if (fabs(det) < EPSILON) return 0;
    *x = (c1 * b2 - c2 * b1) / det;
    *y = (a1 * c2 - a2 * c1) / det;
    return 1;
}

void find_vertices(LinearProgram *lp, Point *vertices, int *n) {
    *n = 0;
    if (is_feasible(lp, 0, 0)) { vertices[(*n)].x = 0; vertices[(*n)++].y = 0; }
    for (int i = 0; i < lp->n_constraints; i++) {
        double a1 = lp->A[i][0], a2 = lp->A[i][1], b = lp->b[i];
        if (fabs(a2) > EPSILON) { double y = b/a2; if (y >= 0 && is_feasible(lp, 0, y)) { vertices[*n].x = 0; vertices[(*n)++].y = y; }}
        if (fabs(a1) > EPSILON) { double x = b/a1; if (x >= 0 && is_feasible(lp, x, 0)) { vertices[(*n)++].x = x; vertices[*n-1].y = 0; }}
    }
    for (int i = 0; i < lp->n_constraints; i++) {
        for (int j = i+1; j < lp->n_constraints; j++) {
            double x, y;
            if (line_intersection(lp->A[i][0], lp->A[i][1], lp->b[i], lp->A[j][0], lp->A[j][1], lp->b[j], &x, &y))
                if (is_feasible(lp, x, y)) { vertices[*n].x = x; vertices[(*n)++].y = y; }
        }
    }
    if (*n > 2) {
        Point center = {0, 0};
        for (int i = 0; i < *n; i++) { center.x += vertices[i].x; center.y += vertices[i].y; }
        center.x /= *n; center.y /= *n;
        for (int i = 0; i < *n-1; i++)
            for (int j = 0; j < *n-i-1; j++) {
                double a1 = atan2(vertices[j].y - center.y, vertices[j].x - center.x);
                double a2 = atan2(vertices[j+1].y - center.y, vertices[j+1].x - center.x);
                if (a1 > a2) { Point t = vertices[j]; vertices[j] = vertices[j+1]; vertices[j+1] = t; }
            }
    }
}

void draw_filled_polygon(Point *v, int n, SDL_Color col) {
    if (n < 3) return;
    SDL_SetRenderDrawColor(renderer, col.r, col.g, col.b, col.a);
    int miny = graph_y(v[0].y), maxy = miny;
    for (int i = 1; i < n; i++) { int gy = graph_y(v[i].y); if (gy < miny) miny = gy; if (gy > maxy) maxy = gy; }
    for (int y = miny; y <= maxy; y++) {
        int xi[100], ni = 0;
        for (int i = 0; i < n; i++) {
            int j = (i+1)%n, y1 = graph_y(v[i].y), y2 = graph_y(v[j].y), x1 = graph_x(v[i].x), x2 = graph_x(v[j].x);
            if ((y1 <= y && y2 > y) || (y2 <= y && y1 > y)) xi[ni++] = x1 + (y-y1)*(x2-x1)/(y2-y1);
        }
        for (int i = 0; i < ni-1; i++) for (int j = 0; j < ni-i-1; j++) if (xi[j] > xi[j+1]) { int t = xi[j]; xi[j] = xi[j+1]; xi[j+1] = t; }
        for (int i = 0; i < ni-1; i += 2) SDL_RenderDrawLine(renderer, xi[i], y, xi[i+1], y);
    }
}

void init_tableau(SimplexTableau *tab, LinearProgram *lp) {
    int slack = 0;
    tab->n_original_vars = lp->n_vars; tab->n_rows = lp->n_constraints;
    for (int i = 0; i < lp->n_constraints; i++) if (lp->constraint_type[i] != 1) slack++;
    tab->n_slack_vars = slack; tab->n_cols = lp->n_vars + slack + 1;
    memset(tab->tableau, 0, sizeof(tab->tableau));
    slack = 0;
    for (int i = 0; i < lp->n_constraints; i++) {
        for (int j = 0; j < lp->n_vars; j++) tab->tableau[i][j] = lp->A[i][j];
        if (lp->constraint_type[i] == 0) { tab->tableau[i][lp->n_vars + slack] = 1.0; tab->basis[i] = lp->n_vars + slack++; }
        else if (lp->constraint_type[i] == 2) { tab->tableau[i][lp->n_vars + slack] = -1.0; slack++; tab->basis[i] = -1; }
        else tab->basis[i] = -1;
        tab->tableau[i][tab->n_cols - 1] = lp->b[i];
    }
    for (int j = 0; j < lp->n_vars; j++) tab->tableau[tab->n_rows][j] = lp->is_max ? -lp->c[j] : lp->c[j];
}

int find_pivot_col(SimplexTableau *tab) {
    int col = -1; double minv = -EPSILON;
    for (int j = 0; j < tab->n_cols - 1; j++) if (tab->tableau[tab->n_rows][j] < minv) { minv = tab->tableau[tab->n_rows][j]; col = j; }
    return col;
}

int find_pivot_row(SimplexTableau *tab, int col) {
    int row = -1; double minr = 1e20;
    for (int i = 0; i < tab->n_rows; i++)
        if (tab->tableau[i][col] > EPSILON) {
            double r = tab->tableau[i][tab->n_cols-1] / tab->tableau[i][col];
            if (r < minr && r >= 0) { minr = r; row = i; }
        }
    return row;
}

void pivot_op(SimplexTableau *tab, int pr, int pc) {
    double pv = tab->tableau[pr][pc];
    for (int j = 0; j < tab->n_cols; j++) tab->tableau[pr][j] /= pv;
    for (int i = 0; i <= tab->n_rows; i++)
        if (i != pr) { double f = tab->tableau[i][pc]; for (int j = 0; j < tab->n_cols; j++) tab->tableau[i][j] -= f * tab->tableau[pr][j]; }
    tab->basis[pr] = pc;
}

Solution solve_simplex(LinearProgram *lp) {
    SimplexTableau tab; Solution sol; int iter = 0;
    memset(&sol, 0, sizeof(sol)); init_tableau(&tab, lp);
    printf("\n========== RESOLUTION SIMPLEXE ==========\n");
    while (1) {
        int pc = find_pivot_col(&tab); if (pc == -1) break;
        int pr = find_pivot_row(&tab, pc);
        if (pr == -1) { sol.status = 1; strcpy(sol.message, "Non borne"); return sol; }
        pivot_op(&tab, pr, pc); iter++;
        if (iter > 100) { sol.status = 2; strcpy(sol.message, "Max iter"); return sol; }
    }
    sol.status = 0; strcpy(sol.message, "Optimal");
    for (int i = 0; i < tab.n_rows; i++) if (tab.basis[i] < lp->n_vars && tab.basis[i] >= 0) sol.x[tab.basis[i]] = tab.tableau[i][tab.n_cols-1];
    sol.z_optimal = -tab.tableau[tab.n_rows][tab.n_cols-1];
    if (!lp->is_max) sol.z_optimal = -sol.z_optimal;
    printf("Iterations: %d\n", iter);
    printf("Solution: x1 = %.4f, x2 = %.4f\n", sol.x[0], sol.x[1]);
    printf("Z* = %.4f\n", sol.z_optimal);
    return sol;
}

void render_graph(LinearProgram *lp, Solution *sol) {
    SDL_Color black = {0,0,0,255}, gray = {180,180,180,255}, light_blue = {173,216,230,180};
    SDL_Color red = {220,50,50,255}, dark_blue = {50,50,150,255};
    SDL_Color colors[] = {{220,80,80,255},{80,80,220,255},{80,180,80,255},{200,150,50,255},{150,50,200,255}};

    SDL_SetRenderDrawColor(renderer, 245, 245, 250, 255); SDL_RenderClear(renderer);
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_Rect gr = {GRAPH_X-5, GRAPH_Y-5, GRAPH_WIDTH+10, GRAPH_HEIGHT+10}; SDL_RenderFillRect(renderer, &gr);
    SDL_SetRenderDrawColor(renderer, 100, 100, 100, 255); SDL_RenderDrawRect(renderer, &gr);

    SDL_SetRenderDrawColor(renderer, 230, 230, 230, 255);
    for (int i = 0; i <= (int)max_x; i++) SDL_RenderDrawLine(renderer, graph_x(i), GRAPH_Y, graph_x(i), GRAPH_Y+GRAPH_HEIGHT);
    for (int i = 0; i <= (int)max_y; i++) SDL_RenderDrawLine(renderer, GRAPH_X, graph_y(i), GRAPH_X+GRAPH_WIDTH, graph_y(i));

    Point verts[50]; int nv; find_vertices(lp, verts, &nv);
    if (nv >= 3) {
        draw_filled_polygon(verts, nv, light_blue);
        SDL_SetRenderDrawColor(renderer, 100, 150, 200, 255);
        for (int i = 0; i < nv; i++) { int j = (i+1)%nv; draw_thick_line(graph_x(verts[i].x), graph_y(verts[i].y), graph_x(verts[j].x), graph_y(verts[j].y), 2); }
    }

    for (int i = 0; i < lp->n_constraints; i++) {
        double a1 = lp->A[i][0], a2 = lp->A[i][1], b = lp->b[i];
        SDL_SetRenderDrawColor(renderer, colors[i%5].r, colors[i%5].g, colors[i%5].b, 255);
        double x1, y1, x2, y2;
        if (fabs(a2) > EPSILON && fabs(a1) > EPSILON) { x1 = 0; y1 = b/a2; x2 = b/a1; y2 = 0; }
        else if (fabs(a1) < EPSILON) { x1 = 0; y1 = b/a2; x2 = max_x; y2 = b/a2; }
        else { x1 = b/a1; y1 = 0; x2 = b/a1; y2 = max_y; }
        draw_thick_line(graph_x(x1), graph_y(y1), graph_x(x2), graph_y(y2), 2);
        char lbl[16]; snprintf(lbl, 16, "C%d", i+1);
        draw_text(lbl, (graph_x(x1)+graph_x(x2))/2+5, (graph_y(y1)+graph_y(y2))/2-15, colors[i%5], font_small);
    }

    SDL_SetRenderDrawColor(renderer, 50, 200, 50, 100);
    double c1 = lp->c[0], c2 = lp->c[1];
    if (fabs(c1) > EPSILON && fabs(c2) > EPSILON) {
        for (double z = 0; z <= sol->z_optimal*1.5; z += sol->z_optimal/5) {
            double lx1 = 0, ly1 = z/c2, lx2 = z/c1, ly2 = 0;
            if (ly1 <= max_y && lx2 <= max_x) SDL_RenderDrawLine(renderer, graph_x(lx1), graph_y(ly1), graph_x(lx2), graph_y(ly2));
        }
        SDL_SetRenderDrawColor(renderer, 0, 180, 0, 255);
        draw_thick_line(graph_x(0), graph_y(sol->z_optimal/c2), graph_x(sol->z_optimal/c1), graph_y(0), 3);
    }

    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    draw_thick_line(GRAPH_X, graph_y(0), GRAPH_X+GRAPH_WIDTH, graph_y(0), 2);
    draw_thick_line(graph_x(0), GRAPH_Y, graph_x(0), GRAPH_Y+GRAPH_HEIGHT, 2);
    for (int i = 0; i <= (int)max_x; i++) { char l[8]; snprintf(l, 8, "%d", i); draw_text(l, graph_x(i)-5, graph_y(0)+8, black, font_small); }
    for (int i = 1; i <= (int)max_y; i++) { char l[8]; snprintf(l, 8, "%d", i); draw_text(l, graph_x(0)-25, graph_y(i)-7, black, font_small); }
    draw_text("x1", GRAPH_X+GRAPH_WIDTH-20, graph_y(0)+20, black, font);
    draw_text("x2", graph_x(0)-30, GRAPH_Y+5, black, font);

    if (sol->status == 0) {
        int ox = graph_x(sol->x[0]), oy = graph_y(sol->x[1]);
        SDL_SetRenderDrawColor(renderer, 220, 50, 50, 255); draw_filled_circle(ox, oy, 12);
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255); draw_filled_circle(ox, oy, 8);
        SDL_SetRenderDrawColor(renderer, 220, 50, 50, 255); draw_filled_circle(ox, oy, 5);
        char ol[32]; snprintf(ol, 32, "(%.2f, %.2f)", sol->x[0], sol->x[1]);
        draw_text(ol, ox+15, oy-25, red, font);
    }

    SDL_SetRenderDrawColor(renderer, 50, 50, 150, 255);
    for (int i = 0; i < nv; i++)
        if (fabs(verts[i].x - sol->x[0]) > 0.01 || fabs(verts[i].y - sol->x[1]) > 0.01)
            draw_filled_circle(graph_x(verts[i].x), graph_y(verts[i].y), 5);

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_Rect ir = {INFO_X-10, INFO_Y-10, 310, 600}; SDL_RenderFillRect(renderer, &ir);
    SDL_SetRenderDrawColor(renderer, 100, 100, 100, 255); SDL_RenderDrawRect(renderer, &ir);

    int yp = INFO_Y;
    draw_text("PROGRAMMATION LINEAIRE", INFO_X, yp, dark_blue, font_large); yp += 35;
    draw_text("Fonction objectif:", INFO_X, yp, black, font); yp += 22;
    char os[64]; snprintf(os, 64, "%s Z = %.0fx1 + %.0fx2", lp->is_max?"Max":"Min", lp->c[0], lp->c[1]);
    draw_text(os, INFO_X+10, yp, dark_blue, font); yp += 35;
    draw_text("Contraintes:", INFO_X, yp, black, font); yp += 22;
    for (int i = 0; i < lp->n_constraints; i++) {
        char cs[64]; const char *op = lp->constraint_type[i]==0 ? "<=" : lp->constraint_type[i]==1 ? "=" : ">=";
        snprintf(cs, 64, "C%d: %.0fx1 + %.0fx2 %s %.0f", i+1, lp->A[i][0], lp->A[i][1], op, lp->b[i]);
        draw_text(cs, INFO_X+10, yp, colors[i%5], font_small); yp += 20;
    }
    draw_text("    x1, x2 >= 0", INFO_X+10, yp, gray, font_small); yp += 40;
    draw_text("SOLUTION OPTIMALE", INFO_X, yp, red, font_large); yp += 30;
    char ss[32]; snprintf(ss, 32, "x1* = %.4f", sol->x[0]); draw_text(ss, INFO_X+10, yp, black, font); yp += 22;
    snprintf(ss, 32, "x2* = %.4f", sol->x[1]); draw_text(ss, INFO_X+10, yp, black, font); yp += 30;
    snprintf(ss, 32, "Z* = %.4f", sol->z_optimal); draw_text(ss, INFO_X+10, yp, red, font_large); yp += 50;
    draw_text("LEGENDE", INFO_X, yp, black, font); yp += 25;
    SDL_SetRenderDrawColor(renderer, 173, 216, 230, 255); SDL_Rect lr = {INFO_X+10, yp, 20, 15}; SDL_RenderFillRect(renderer, &lr);
    draw_text("Region admissible", INFO_X+40, yp, black, font_small); yp += 22;
    SDL_SetRenderDrawColor(renderer, 0, 180, 0, 255); SDL_RenderDrawLine(renderer, INFO_X+10, yp+7, INFO_X+30, yp+7);
    draw_text("Iso-profit optimal", INFO_X+40, yp, black, font_small); yp += 22;
    SDL_SetRenderDrawColor(renderer, 220, 50, 50, 255); draw_filled_circle(INFO_X+20, yp+7, 6);
    draw_text("Point optimal", INFO_X+40, yp, black, font_small); yp += 35;
    draw_text("[ESC] pour quitter", INFO_X, yp, gray, font_small);
    SDL_RenderPresent(renderer);
}

void run_graphical(LinearProgram *lp, Solution *sol) {
    max_x = max_y = 2;
    for (int i = 0; i < lp->n_constraints; i++) {
        if (fabs(lp->A[i][0]) > EPSILON) { double x = lp->b[i]/lp->A[i][0]; if (x > max_x) max_x = x; }
        if (fabs(lp->A[i][1]) > EPSILON) { double y = lp->b[i]/lp->A[i][1]; if (y > max_y) max_y = y; }
    }
    max_x = ceil(max_x * 1.3); max_y = ceil(max_y * 1.3);
    scale_x = GRAPH_WIDTH / max_x; scale_y = GRAPH_HEIGHT / max_y;
    init_sdl();
    int run = 1; SDL_Event e;
    while (run) {
        while (SDL_PollEvent(&e)) { if (e.type == SDL_QUIT || (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)) run = 0; }
        render_graph(lp, sol); SDL_Delay(16);
    }
    close_sdl();
}

int main(int argc, char *argv[]) {
    LinearProgram lp; Solution sol; int choice;
    printf("\n==================================================\n");
    printf("  SOLVEUR PROGRAMMATION LINEAIRE - SIMPLEXE\n");
    printf("  Avec Affichage Graphique SDL2\n");
    printf("==================================================\n");
    printf("\n1. Exemple: Max Z = 300x1 + 500x2 (cours)\n");
    printf("2. Exemple: Max Z = 50x1 + 60x2\n");
    printf("3. Entrer un nouveau probleme\n");
    printf("Choix: ");
    scanf("%d", &choice);

    if (choice == 1) {
        lp.n_vars = 2; lp.n_constraints = 3; lp.is_max = 1;
        lp.c[0] = 300; lp.c[1] = 500;
        lp.A[0][0] = 1; lp.A[0][1] = 0; lp.b[0] = 4; lp.constraint_type[0] = 0;
        lp.A[1][0] = 0; lp.A[1][1] = 2; lp.b[1] = 12; lp.constraint_type[1] = 0;
        lp.A[2][0] = 3; lp.A[2][1] = 2; lp.b[2] = 18; lp.constraint_type[2] = 0;
    } else if (choice == 2) {
        lp.n_vars = 2; lp.n_constraints = 3; lp.is_max = 1;
        lp.c[0] = 50; lp.c[1] = 60;
        lp.A[0][0] = 1; lp.A[0][1] = 2; lp.b[0] = 8; lp.constraint_type[0] = 0;
        lp.A[1][0] = 2; lp.A[1][1] = 2; lp.b[1] = 10; lp.constraint_type[1] = 0;
        lp.A[2][0] = 9; lp.A[2][1] = 4; lp.b[2] = 36; lp.constraint_type[2] = 0;
    } else {
        printf("\nType (1=Max, 0=Min): "); scanf("%d", &lp.is_max);
        lp.n_vars = 2;
        printf("Coefficients c1 c2: "); scanf("%lf %lf", &lp.c[0], &lp.c[1]);
        printf("Nombre de contraintes: "); scanf("%d", &lp.n_constraints);
        for (int i = 0; i < lp.n_constraints; i++) {
            printf("C%d (a1 a2 type[0/1/2] b): ", i+1);
            scanf("%lf %lf %d %lf", &lp.A[i][0], &lp.A[i][1], &lp.constraint_type[i], &lp.b[i]);
        }
    }
    sol = solve_simplex(&lp);
    if (lp.n_vars == 2) { printf("\nOuverture fenetre graphique...\n"); run_graphical(&lp, &sol); }
    return 0;
}
