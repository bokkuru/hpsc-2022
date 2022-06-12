#include <cstdio>
#include <cmath>
#include <cstring>

const int nx = 41;
const int ny = 41;
const int nt = 50;
const int nit = 50;
const double dt = 0.01;
const double rho = 1;
const double nu = 0.02;

__device__ double pow2(double x){
	return x*x;
}

__global__ void cavity(double *u,double *v,double *b,double *p,double *un,double *vn,double *pn,int dx,int dy){
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(j > 0){printf("%d,%d\n",i,j);}

    if((j < ny-1) && (i < nx-1)){	
            b[j+i*ny] = rho*(1/dt*
                        ((u[j+(i+1)*ny] - u[j+(i-1)*ny])/(2*dx)+(v[j+1+i*ny]-v[j-1+i*ny])/(2*dy))-
                        pow2((u[j+(i+1)*ny]-u[j+(i-1)*ny])/(2*dx)) - 2 *((u[j+1+i*ny]-u[j-1+i*ny])/(2*dy)*
                        (v[j+(i+1)*ny]-v[j+(i-1)*ny])/(2*dx))-pow2((v[j+1+i*ny]-v[j-1+i*ny])/(2*dy)));
        for (int it = 0; it < nit; it++){
            __syncthreads();
            pn[j+i*ny] = p[j+i*ny];
            __syncthreads();
            p[j+i*ny] = (dy * dy * (pn[j+(i+1)*ny] + pn[j+(i-1)*ny]) +
                        dx * dx * (pn[j+1+i*ny] + pn[j-1+i*ny]) -
                        b[j+i*ny]  * dx * dx * dy * dy)
                        /(2 * (dx * dy + dy * dy));
            __syncthreads();
            p[ny - 1+i*ny] = 0;
            p[i*ny] = p[1+i*ny];
            p[j+ny*(nx - 1)] = p[j+ny*(nx - 2)];
            p[j] = p[j+ny];
        }
        __syncthreads();
        un[j+i*ny] = u[j+i*ny];
        vn[j+i*ny] = v[j+i*ny];
        __syncthreads();
        u[j+i*ny] = un[j+i*ny] - un[j+i*ny] * dt / dx * (un[j+i*ny] - un[j+(i-1)*ny])
                    - un[j+i*ny] * dt / dy * (un[j+i*ny] - un[j-1+i*ny])
                    - dt / (2 * rho * dx) * (p[j+(i+1)*ny] - p[j+(i-1)*ny])
                    + nu * dt / (dx * dx) * (un[j+(i+1)*ny] - 2 * un[j+i*ny] + un[j+(i-1)*ny])
                    + nu * dt / (dy * dy) * (un[j+1+i*ny] - 2 * un[j+i*ny] + un[j-1+i*ny]);
        v[j+i*ny] = vn[j+i*ny] - vn[j+i*ny] * dt / dx * (vn[j+i*ny] - vn[j+(i-1)*ny])
                    - vn[j+i*ny] * dt / dy * (vn[j+i*ny] - vn[j-1+i*ny])
                    - dt / (2 * rho * dx) * (p[j+1+i*ny] - p[j-1+i*ny])
                    + nu * dt / (dx * dx) * (vn[j+(i+1)*ny] - 2 * vn[j+i*ny] + vn[j+(i-1)*ny])
                    + nu * dt / (dy * dy) * (vn[j+1+i*ny] - 2 * vn[j+i*ny] + vn[j-1+i*ny]);
        __syncthreads();
        u[j] = 0;
        u[j+ny*(nx - 1)] = 0;
        v[j] = 0;
        v[j+ny*(nx - 1)] = 0;
        u[ny*i] = 0;
        u[ny - 1+ny*i] = 1;
        v[ny*i] = 0;
        v[ny - 1+ny*i] = 0;
        //printf("%.2f,%.2f\n",u[j+i*ny]*1000,v[j+i*ny]);
    }
    return;
}

int main(){
    
    double dx = 2.0/(nx-1);
    double dy = 2.0/(ny-1);
    
    
    double *u,*v,*p,*b,*pn,*un,*vn;
    cudaMallocManaged(&u, ny * nx * sizeof(double));
    cudaMallocManaged(&v, ny * nx * sizeof(double));
    cudaMallocManaged(&p, ny * nx * sizeof(double));
    cudaMallocManaged(&b, ny * nx * sizeof(double));
    cudaMallocManaged(&un, ny * nx * sizeof(double));
    cudaMallocManaged(&vn, ny * nx * sizeof(double));
    cudaMallocManaged(&pn, ny * nx * sizeof(double));

    for (int i = 0; i < ny*nx; ++i){
            u[i]= 0;
            v[i]= 0;
            p[i]= 0;
            b[i]= 0;
    }
    for (int n = 0; n < nt; n++){
        cavity<<<(nx*ny-1)/1024+1,1024>>>(u,v,b,p,un,vn,pn,dx,dy);
        cudaDeviceSynchronize();
        printf("%d\n",n);
        printf("u\n");
        /*
        for (int j = 1; j < ny - 1; j++){
            for (int i = 1; i < nx - 1; i++){
                printf("%.2f ", u[j+i*ny]*1000);
            }
            printf("\n");
        }
        printf("v\n");
        for (int j = 1; j < ny - 1; j++){
            for (int i = 1; i < nx - 1; i++){
                printf("%.2f ", v[j+i*ny]*1000);
            }
            printf("\n");
        }
        */
    }
    cudaFree(u);
    cudaFree(v);
    cudaFree(p);
    cudaFree(b);
    cudaFree(un);
    cudaFree(vn);
    cudaFree(pn);
    return 0;
}