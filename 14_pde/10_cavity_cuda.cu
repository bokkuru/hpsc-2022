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
    int i = blockDim.x*blockIdx.x+threadIdx.x+1;
	int j = blockDim.y*blockIdx.y+threadIdx.y+1;

    if((j < ny-1) && (i < nx-1)){	
            b[j+i*ny] = rho*(1/dt*
                        ((u[j+(i+1)*ny] - u[j+(i-1)*ny])/(2*dx)+(v[j+1+i*ny]-v[j-1+i*ny])/(2*dy))-
                        pow((u[j+(i+1)*ny]-u[j+(i-1)*ny])/(2*dx),2) - 2 *((u[j+1+i*ny]-u[j-1+i*ny])/(2*dy)*
                        (v[j+(i+1)*ny]-v[j+(i-1)*ny])/(2*dx))-pow((v[j+1+i*ny]-v[j-1+i*ny])/(2*dy),2));
        for (int it = 0; it < nit; it++){
            pn = p;
            p[j+i*ny] = (dy * dy * (pn[j+(i+1)*ny] + pn[j+(i-1)*ny]) +
                        dx * dx * (pn[j+1+i*ny] + pn[j-1+i*ny]) -
                        b[j+i*ny]  * dx * dx * dy * dy)
                        /(2 * (dx * dy + dy * dy));
            p[ny - 1+i*ny] = 0;
            p[i*ny] = p[1+i*ny];
            p[i+ny*(nx - 1)] = p[i+ny*(nx - 2)];
            p[i] = p[i+ny];
        }
        un = u;
        vn = v;
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
        //printf("%lf,%lf",u[j+i*ny],v[j+i*ny]);
        /*
        u[j] = 0;
        u[j+ny*(nx - 1)] = 0;
        v[j] = 0;
        v[j+ny*(nx - 1)] = 0;
        u[ny*i] = 0;
        u[ny - 1+ny*i] = 1;
        v[ny*i] = 0;
        v[ny - 1+ny*i] = 0;
        */
       u[i*ny] = 0;
		u[i*ny + nx-1] = 1;
		v[i*ny] = 0;
		v[i*ny + nx-1] = 0;
        u[j] = 0;
        	u[i*(ny-1) + j] = 0;
        	v[j] = 0;
        	v[i*(ny-1) + j] = 0;
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
        cudaDeviceSynchronize();
        cavity<<<nx*ny/1024+1,1024>>>(u,v,b,p,un,vn,pn,dx,dy);
        cudaDeviceSynchronize();
        printf("%d\n",n);
        printf("u\n");
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