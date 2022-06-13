#include <cstdio>
#include <cstring>
#include <math.h>
#include <cmath>
#include <iostream>
#include <vector>
const int nx = 41;
const int ny = 41;
const int nt = 5;
const int nit = 50;
const double dt = 0.01;
const double rho = 1.0;
const double nu = 0.02;

const int N = ny * nx;
const int M = 1024;

__global__ void init(double *u, double *v, double *p, double *b) {
    int num = blockIdx.x * blockDim.x + threadIdx.x;
    if(num >= ny*nx){return;}

    u[num] = 0;
    v[num] = 0;
    p[num] = 0;
    b[num] = 0;
}

__global__ void cavity(double *u, double *v, double *p, double *b, double *un, double *vn, double *pn,int nx, int ny, int nit, double dx, double dy, double dt, double rho, double nu) {
    int  num = blockIdx.x * blockDim.x + threadIdx.x;
    if ( num >= ny * nx) return;
    int j =  num / ny;
    int i =  num % ny;

    if((0 < j) && (j < ny-1) && (0 < i) && (i < nx-1)){
        b[num] = rho*(1/dt*
                ((u[j*ny+i+1] - u[j*ny+(i-1)])/(2*dx)+(v[(j+1)*ny+i]-v[(j-1)*ny+i])/(2*dy))-
                pow((u[j*ny+(i+1)]-u[j*ny+(i-1)])/(2*dx),2) - 2 *((u[(j+1)*ny+i]-u[(j-1)*ny+i])/(2*dy)*
                (v[j*ny+(i+1)]-v[j*ny+(i-1)])/(2*dx))-pow((v[(j+1)*ny+i]-v[(j-1)*ny+i])/(2*dy),2));
    }

    for (int it = 0; it < nit; it++) {
        pn[num] = p[num];
        __syncthreads();
        if((0 < j) && (j < ny-1) && (0 < i) && (i < nx-1)){
            p[num] = (dy * dy * (pn[j*ny+(i+1)] + pn[j*ny+(i-1)]) +
                     dx * dx * (pn[(j+1)*ny+i] + pn[(j-1)*ny+i]) -
                     b[j*ny+i]  * dx * dx * dy * dy)
                     /(2 * (dx * dy + dy * dy));
        }
        __syncthreads();
        p[(ny-1)*ny+i]=0;
        p[0*ny+i]=p[1*ny+i];
        p[j*ny+(nx-1)]=p[j*ny+(nx-2)];
        p[j*ny+0]=p[j*ny+1];
        __syncthreads();
    }
    __syncthreads();
    un[num] = u[num];
    vn[num] = v[num];
    __syncthreads();
    if((0 < j) && (j < ny-1) && (0 < i) && (i < nx-1)){
        u[num] = un[j*ny+i]-un[j*ny+i]*dt/dx*(un[j*ny+i]-un[j*ny+(i-1)]) 
                 -un[j*ny+i]*dt/dy*(un[j*ny+i]-un[(j-1)*ny+i]) 
                 -dt/(2.0*rho*dx)*(p[j*ny+(i+1)]-p[j*ny+(i-1)]) 
                 +nu*dt/(dx*dx)*(un[j*ny+(i+1)]-2.0*un[j*ny+i]+un[j*ny+(i-1)]) 
                 +nu*dt/(dy*dy)*(un[(j+1)*ny+i]-2.0*un[j*ny+i]+un[(j-1)*ny+i]);
        v[num] =vn[j*ny+i]-vn[j*ny+i]*dt/dx*(vn[j*ny+i]-vn[j*ny+(i-1)]) 
                -vn[j*ny+i]*dt/dy*(vn[j*ny+i]-vn[(j-1)*ny+i])-
                -dt/(2.0*rho*dx)*(p[(j+1)*ny+i]-p[(j-1)*ny+i])
                +nu*dt/(dx*dx)*(vn[j*ny+(i+1)]-2.0*vn[j*ny+i]+vn[j*ny+(i-1)])
                +nu*dt/(dy*dy)*(vn[(j+1)*ny+i]-2.0*vn[j*ny+i]+vn[(j-1)*ny+i]);
    }
    __syncthreads();
    u[j*ny]=0;
    u[j*ny+(nx-1)]=0;
    v[j*ny]=0;
    v[j*ny+(nx-1)]=0;
    u[i]=0;
    u[(ny-1)*ny+i]=1;
    v[i]=0;
    v[(ny-1)*ny+i]=0;
    __syncthreads();
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

    init<<<(N+M-1)/M,M>>>(u,v,p,b);
    /*
    for (int i = 0; i < ny*nx; ++i){
        u[i]= 0;
        v[i]= 0;
        p[i]= 0;
        b[i]= 0;
    }
    */
    for (int n = 0; n < nt; n++){
        cavity<<<(N + M - 1) / M, M>>>(u,v, p, b, un, vn, pn, nx, ny, nit, dx, dy, dt, rho, nu);
        cudaDeviceSynchronize();
        printf("%d\n",n);
        printf("u\n");
        for (int j =0; j < ny; j++){
            for (int i = 0; i < nx; i++){
                printf("%f ", u[j+i*ny]);
            }
            printf("\n");
        }
        printf("v\n");
        for (int j = 0; j < ny; j++){
            for (int i = 0; i < nx; i++){
                printf("%f ", v[j+i*ny]);
            }
            printf("\n");
        }
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