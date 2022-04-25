#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

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
  __m256 fyvec = _mm256_load_ps(fy);
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  __m256 nvec = _mm256_load_ps(n);
  for(int i=0; i<N; i++) {
    __m256 ivec = _mm256_set1_ps(i);
    __m256 mask = _mm256_cmp_ps(nvec, ivec, _CMP_NEQ_UQ);
    __m256 ixvec = _mm256_set1_ps(x[i]);
    __m256 iyvec = _mm256_set1_ps(y[i]);
    __m256 rxvec = _mm256_sub_ps(ixvec, xvec);
    __m256 ryvec = _mm256_sub_ps(iyvec, yvec);
    __m256 rxtmpvec = _mm256_mul_ps(rxvec,rxvec);
    __m256 rytmpvec = _mm256_mul_ps(ryvec,ryvec);
    __m256 rtmpvec = _mm256_add_ps(rxtmpvec, rytmpvec);
    __m256 rvec = _mm256_rsqrt_ps(rtmpvec);
    __m256 tmpvec = _mm256_mul_ps(rvec, rvec);
    __m256 tmp1vec = _mm256_mul_ps(tmpvec, rvec);
    __m256 tmp2vec = _mm256_mul_ps(mvec, tmp1vec);
    __m256 fxtmpvec = _mm256_mul_ps(rxvec, tmp2vec);
    __m256 fytmpvec = _mm256_mul_ps(ryvec, tmp2vec);
    __m256 fx1tmpvec = _mm256_mul_ps(fxtmpvec, mask);
    __m256 fy1tmpvec = _mm256_mul_ps(fytmpvec, mask);
    fxvec = _mm256_sub_ps(fxvec, fx1tmpvec);
    fyvec = _mm256_sub_ps(fyvec, fy1tmpvec);
    _mm256_store_ps(fx, fxvec);
    _mm256_store_ps(fy, fyvec);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
