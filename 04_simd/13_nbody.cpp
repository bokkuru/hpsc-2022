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
  __m256 onevec = _mm256_set1_ps(1);
  for(int i=0; i<N; i++) {
    __m256 xvec = _mm256_load_ps(x);
    __m256 yvec = _mm256_load_ps(y);
    __m256 mvec = _mm256_load_ps(m);
    __m256 ivec = _mm256_set1_ps(i);
    __m256 nvec = _mm256_load_ps(n);
    __m256 rmask = _mm256_cmp_ps(nvec, ivec, _CMP_NEQ_UQ);
    __m256 mask = _mm256_sub_ps(onevec, rmask);
    __m256 ixvec = _mm256_set1_ps(x[i]);
    __m256 iyvec = _mm256_set1_ps(y[i]);
    __m256 rxvec = _mm256_sub_ps(ixvec, xvec);
    __m256 ryvec = _mm256_sub_ps(iyvec, yvec);
    __m256 rvec = _mm256_mul_ps(rxvec,rxvec);
    __m256 rtmpvec = _mm256_mul_ps(ryvec,ryvec);
    rvec = _mm256_add_ps(rvec, rtmpvec);
    rvec = _mm256_rsqrt_ps(rvec);
    __m256 tmpvec = _mm256_mul_ps(rvec, rvec);
    tmpvec = _mm256_mul_ps(tmpvec, rvec);
    tmpvec = _mm256_mul_ps(mvec, tmpvec);
    __m256 fxtmpvec = _mm256_mul_ps(rxvec, tmpvec);
    __m256 fytmpvec = _mm256_mul_ps(ryvec, tmpvec);
    fxtmpvec = _mm256_mul_ps(fxtmpvec, mask);
    fytmpvec = _mm256_mul_ps(fytmpvec, mask);
    fxvec = _mm256_sub_ps(fxvec, fxtmpvec);
    fyvec = _mm256_sub_ps(fyvec, fytmpvec);
    _mm256_store_ps(fx, fxvec);
    _mm256_store_ps(fy, fyvec);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
