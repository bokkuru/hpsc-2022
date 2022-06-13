#include <cmath>
#include <cstring>

const int nx = 41;
const int ny = 41;
const int nt = 5;
const int nit = 50;
const double dt = 0.01;
const double rho = 1;
const double nu = 0.02;

int main(){
    
    double dx = 2.0/(nx-1);
    double dy = 2.0/(ny-1);
    
    
    double u[ny][nx],v[ny][nx],p[ny][nx],b[ny][nx],pn[ny][nx],un[ny][nx],vn[ny][nx];
#pragma acc parallel loop collapse(2)
    for (int i = 0; i < ny; ++i){
        for (int j = 0; j < nx; ++j){
            u[i][j] = 0;
            v[i][j] = 0;
            p[i][j] = 0;
            b[i][j] = 0;
        }
    }
    for (int n = 0; n < nt; n++){
#pragma acc parallel loop collapse(2)
        for (int j = 1; j < ny - 1; j++){
            for (int i = 1; i < nx - 1; i++){
                b[j][i] = rho*(1/dt*
                          ((u[j][i+1] - u[j][i-1])/(2*dx)+(v[j+1][i]-v[j-1][i])/(2*dy))-
                          pow((u[j][i+1]-u[j][i-1])/(2*dx),2) - 2 *((u[j+1][i]-u[j-1][i])/(2*dy)*
                          (v[j][i+1]-v[j][i-1])/(2*dx))-pow((v[j+1][i]-v[j-1][i])/(2*dy),2));
            }
        }
        for (int it = 0; it < nit; it++){
            memcpy(pn, p, sizeof(p));
#pragma acc parallel loop collapse(2)
            for (int j = 1; j < ny - 1; j++){
                for (int i = 1; i < nx - 1; i++){
                    p[j][i] = (dy * dy * (pn[j][i + 1] + pn[j][i - 1]) +
                              dx * dx * (pn[j + 1][i] + pn[j - 1][i]) -
                              b[j][i]  * dx * dx * dy * dy)
                              /(2 * (dx * dy + dy * dy));
                }
            }
#pragma acc parallel loop
            for (int i = 0; i < nx; i++){
                p[ny - 1][i] = 0;
                p[0][i] = p[1][i];
            }
#pragma acc parallel loop
            for (int i = 0; i < ny; i++){
                p[i][nx - 1] = p[i][nx - 2];
                p[i][0] = p[i][1];
            }
        }
        memcpy(un, u, sizeof(u));
        memcpy(vn, v, sizeof(v));
#pragma acc parallel loop collapse(2)
        for (int j = 1; j < ny - 1; j++){
            for (int i = 1; i < nx - 1; i++){
                u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1])
                          - un[j][i] * dt / dy * (un[j][i] - un[j - 1][i])
                          - dt / (2 * rho * dx) * (p[j][i + 1] - p[j][i - 1])
                          + nu * dt / (dx * dx) * (un[j][i + 1] - 2 * un[j][i] + un[j][i - 1])
                          + nu * dt / (dy * dy) * (un[j + 1][i] - 2 * un[j][i] + un[j - 1][i]);
                v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1])
                          - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i])
                          - dt / (2 * rho * dx) * (p[j + 1][i] - p[j - 1][i])
                          + nu * dt / (dx * dx) * (vn[j][i + 1] - 2 * vn[j][i] + vn[j][i - 1])
                          + nu * dt / (dy * dy) * (vn[j + 1][i] - 2 * vn[j][i] + vn[j - 1][i]);
                //printf("%lf,%lf",u[j][i],v[j][i]);
            }
        }
#pragma acc parallel loop
        for (int j = 0; j < ny; j++){
            u[j][0] = 0;
            u[j][nx - 1] = 0;
            v[j][0] = 0;
            v[j][nx - 1] = 0;
        }
#pragma acc parallel loop
        for (int i = 0; i < nx; i++){
            u[0][i] = 0;
            u[ny - 1][i] = 1;
            v[0][i] = 0;
            v[ny - 1][i] = 0;
        }
        printf("%d\n",n);
        printf("u\n");
        for (int j = 1; j < ny - 1; j++){
            for (int i = 1; i < nx - 1; i++){
                printf("%.2f ", u[j][i]*1000);
            }
            printf("\n");
        }
        printf("v\n");
        for (int j = 1; j < ny - 1; j++){
            for (int i = 1; i < nx - 1; i++){
                printf("%.2f ", v[j][i]*1000);
            }
            printf("\n");
        }
    }
    printf("\n");
    return 0;
}