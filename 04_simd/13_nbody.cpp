#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

void printVec(__m256 xvec){
  float x[8];
  _mm256_store_ps(x, xvec);
  for(int i=0;i<8;i++){printf("%g ",x[i]);};
  printf("\n");
}

int main() {
  const int N = 8;
  float x[N], y[N], m[N], n[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    n[i] = i;
    fx[i] = fy[i] = 0;
  }
  __m256 fxvec = _mm256_load_ps(fx);
  printVec(fxvec);
  __m256 fyvec = _mm256_load_ps(fy);
  printVec(fyvec);
  __m256 xvec = _mm256_load_ps(x);
  printVec(xvec);
  __m256 yvec = _mm256_load_ps(y);
  printVec(yvec);
  __m256 mvec = _mm256_load_ps(m);
  printVec(mvec);
  __m256 nvec = _mm256_load_ps(n);
  printVec(nvec);
  for(int i=0; i<N; i++) {
    __m256 ivec = _mm256_set1_ps(i);
    printVec(ivec);
    __m256 mask = _mm256_cmp_ps(nvec, ivec, _CMP_EQ_OQ);
    printVec(mask);
    __m256 ixvec = _mm256_set1_ps(x[i]);
    printVec(ixvec);
    __m256 iyvec = _mm256_set1_ps(y[i]);
    printVec(iyvec);
    __m256 rxvec = _mm256_sub_ps(ixvec, xvec);
    printVec(rxvec);
    __m256 ryvec = _mm256_sub_ps(iyvec, yvec);
    printVec(ryvec);
    __m256 rxtmpvec = _mm256_mul_ps(rxvec,rxvec);
    printVec(rxtmpvec);
    __m256 rytmpvec = _mm256_mul_ps(ryvec,ryvec);
    printVec(rytmpvec);
    __m256 rtmpvec = _mm256_add_ps(rxtmpvec, rytmpvec);
    printVec(rtmpvec);
    __m256 rvec = _mm256_rsqrt_ps(rtmpvec);
    printVec(rvec);
    __m256 tmpvec = _mm256_mul_ps(rvec, rvec);
    printVec(tmpvec);
    __m256 tmp1vec = _mm256_mul_ps(tmpvec, rvec);
    printVec(tmp1vec);
    __m256 tmp2vec = _mm256_mul_ps(mvec, tmp1vec);
    printVec(tmp2vec);
    __m256 fxtmpvec = _mm256_mul_ps(rxvec, tmp2vec);
    printVec(fxtmpvec);
    __m256 fytmpvec = _mm256_mul_ps(ryvec, tmp2vec);
    printVec(fytmpvec);
    __m256 fx1tmpvec = _mm256_mul_ps(fxtmpvec, mask);
    printVec(fx1tmpvec);
    __m256 fy1tmpvec = _mm256_mul_ps(fytmpvec, mask);
    printVec(fy1tmpvec);
    fxvec = _mm256_sub_ps(fxvec, fx1tmpvec);
    fyvec = _mm256_sub_ps(fyvec, fy1tmpvec);
    _mm256_store_ps(fx, fxvec);
    _mm256_store_ps(fy, fyvec);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
