#include <omp.h>
#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

#define MAX_SIZE 64
#define VEC 16

// Tripcount identifiers
__constant int c_size = MAX_SIZE;

//static void MAC1()

// __kernel __attribute__((reqd_work_group_size(1, 1, 1)))
// __attribute__ ((xcl_dataflow))
// static
// void activate(__global float *m_, 
// __global float *x_, 
// __global float *z_, 
// int x_offset, 
// int z_offset, 
// float *A, 
// float *AA, 
// float *AAA) {
static void L1(__global float *m_,
__global float *x_,
int x_offset,
float *A,
float *z) {

	float local_in[784];// __attribute__((xcl_array_partition(cyclic,17,1)));
  	int i, j;
/* Copyright (C) 1991-2020 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */
/* This header is separate from features.h so that the compiler can
   include it implicitly at the start of every compilation.  It must
   not itself include <features.h> or any other header that includes
   <features.h> because the implicit include comes before any feature
   test macros that may be defined in a source file before it first
   explicitly includes a system header.  GCC knows the name of this
   header in order to preinclude it.  */
/* glibc's intent is to support the IEC 559 math functionality, real
   and complex.  If the GCC (4.9 and later) predefined macros
   specifying compiler intent are available, use them to determine
   whether the overall intent is to support these features; otherwise,
   presume an older compiler has intent to support these features and
   define these macros by default.  */
/* wchar_t uses Unicode 10.0.0.  Version 10.0 of the Unicode Standard is
   synchronized with ISO/IEC 10646:2017, fifth edition, plus
   the following additions from Amendment 1 to the fifth edition:
   - 56 emoji characters
   - 285 hentaigana
   - 3 additional Zanabazar Square characters */
  int t1, t2, t3, t4, t5;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
/* Start of CLooG code */
lbp=0;
ubp=24;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5)
for (t2=lbp;t2<=ubp;t2++) {
  lbv=32*t2;
  ubv=min(99,32*t2+31);
#pragma ivdep
#pragma vector always
  for (t3=lbv;t3<=ubv;t3++) {
    local_in[t3] = x_[x_offset+t3];;
    z[t3] = m_[78400+t3];;
  }
  lbv=max(100,32*t2);
  ubv=min(783,32*t2+31);
#pragma ivdep
#pragma vector always
  for (t3=lbv;t3<=ubv;t3++) {
    local_in[t3] = x_[x_offset+t3];;
  }
}
lbp=0;
ubp=3;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5)
for (t2=lbp;t2<=ubp;t2++) {
  for (t3=0;t3<=24;t3++) {
    for (t4=32*t2;t4<=min(99,32*t2+31);t4++) {
      for (t5=32*t3;t5<=min(783,32*t3+31);t5++) {
        z[t4] += A[t4 * 784 + t5] * local_in[t5];;
      }
    }
  }
}
/* End of CLooG code */
}

static void L2(__global float *m_,
float *AA,
float *z,
float *zz) {

	int i, j;

/* Copyright (C) 1991-2020 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */
/* This header is separate from features.h so that the compiler can
   include it implicitly at the start of every compilation.  It must
   not itself include <features.h> or any other header that includes
   <features.h> because the implicit include comes before any feature
   test macros that may be defined in a source file before it first
   explicitly includes a system header.  GCC knows the name of this
   header in order to preinclude it.  */
/* glibc's intent is to support the IEC 559 math functionality, real
   and complex.  If the GCC (4.9 and later) predefined macros
   specifying compiler intent are available, use them to determine
   whether the overall intent is to support these features; otherwise,
   presume an older compiler has intent to support these features and
   define these macros by default.  */
/* wchar_t uses Unicode 10.0.0.  Version 10.0 of the Unicode Standard is
   synchronized with ISO/IEC 10646:2017, fifth edition, plus
   the following additions from Amendment 1 to the fifth edition:
   - 56 emoji characters
   - 285 hentaigana
   - 3 additional Zanabazar Square characters */
  int t1, t2, t3, t4, t5;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
/* Start of CLooG code */
lbp=0;
ubp=3;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5)
for (t2=lbp;t2<=ubp;t2++) {
  lbv=32*t2;
  ubv=min(99,32*t2+31);
#pragma ivdep
#pragma vector always
  for (t3=lbv;t3<=ubv;t3++) {
    zz[t3] = m_[88500+t3];;
  }
}
lbp=0;
ubp=3;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5)
for (t2=lbp;t2<=ubp;t2++) {
  for (t3=0;t3<=3;t3++) {
    for (t4=32*t2;t4<=min(99,32*t2+31);t4++) {
      for (t5=32*t3;t5<=min(99,32*t3+31);t5++) {
        zz[t4] += AA[t4 * 100 + t5] * z[t5];;
      }
    }
  }
}
/* End of CLooG code */
}

static void L3(__global float *m_,
__global float *z_,
int z_offset,
float *AAA,
float *zz) {

	float zzz[10], max=INT_MAX, sum;// __attribute__((xcl_array_partition(complete,1)));;
	float exp_z[10], exp_max=(float)exp(max);
	int i, j;

/* Copyright (C) 1991-2020 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */
/* This header is separate from features.h so that the compiler can
   include it implicitly at the start of every compilation.  It must
   not itself include <features.h> or any other header that includes
   <features.h> because the implicit include comes before any feature
   test macros that may be defined in a source file before it first
   explicitly includes a system header.  GCC knows the name of this
   header in order to preinclude it.  */
/* glibc's intent is to support the IEC 559 math functionality, real
   and complex.  If the GCC (4.9 and later) predefined macros
   specifying compiler intent are available, use them to determine
   whether the overall intent is to support these features; otherwise,
   presume an older compiler has intent to support these features and
   define these macros by default.  */
/* wchar_t uses Unicode 10.0.0.  Version 10.0 of the Unicode Standard is
   synchronized with ISO/IEC 10646:2017, fifth edition, plus
   the following additions from Amendment 1 to the fifth edition:
   - 56 emoji characters
   - 285 hentaigana
   - 3 additional Zanabazar Square characters */
  int t1, t2, t3, t4, t5, t6;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
/* Start of CLooG code */
sum=0.0;;
lbv=0;
ubv=9;
#pragma ivdep
#pragma vector always
for (t3=lbv;t3<=ubv;t3++) {
  zzz[t3] = m_[t3+89600];;
}
for (t3=0;t3<=3;t3++) {
  for (t4=0;t4<=9;t4++) {
    for (t5=32*t3;t5<=min(99,32*t3+31);t5++) {
      zzz[t4] += AAA[t4 * 100 + t5] * zz[t5];;
    }
  }
}
for (t2=0;t2<=9;t2++) {
  exp_z[t2] = (float)exp(zzz[t2]);;
  sum += exp_z[t2];;
}
sum /= exp_max;;
lbv=0;
ubv=9;
#pragma ivdep
#pragma vector always
for (t3=lbv;t3<=ubv;t3++) {
  zzz[t3] = exp_z[t3]/sum;;
  z_[z_offset+t3] = zzz[t3];;
}
/* End of CLooG code */
}


// __kernel __attribute__((reqd_work_group_size(1, 1, 1)))
// // __attribute__ ((xcl_dataflow))
// void run_activate(__global float *m_, __global float *x_, __global float *z_,
// int x_size, int z_size, int test_n) {
	
// 	int x_offset=0, z_offset=0;
//   	// __local 
// 	float A[78400] __attribute__((xcl_array_partition(cyclic,64,1)));
//   	__attribute__((xcl_pipeline_loop(1)))
//   	RD_A: for(int j = 0; j < 78400; j++){
//     	A[j] = m_[j];
//   	}

//   	// __local 
// 	float AA[10000] __attribute__((xcl_array_partition(block,100,1)));
//  	__attribute__((xcl_pipeline_loop(1)))
//   	RD_AA: for(int j = 0; j < 10000; j++){
//     	// AA[j] = m_[j+314000];
//   		AA[j] = m_[j+78500];
//   	}

//   	// __local 
// 	float AAA[1000] __attribute__((xcl_array_partition(block,10,1)));
//   	__attribute__((xcl_pipeline_loop(1)))
//   	RD_AAA: for(int j = 0; j < 1000; j++){
//   		// AAA[j] = m_[j+354400];
//   		AAA[j] = m_[j+88600];
//   	}
  	
// 	// __attribute__((xcl_pipeline_loop(1)))
// 	__attribute__((opencl_unroll_hint(1)))
// 	ACT: for(int i=0; i<test_n; i++){
// 		x_offset = x_size * i;
// 		z_offset = z_size * i;

// 		float z[100] __attribute__((xcl_array_partition(cyclic,17,1)));
// 		float zz[100] __attribute__((xcl_array_partition(cyclic,17,1)));

// 		// activate(m_, x_, z_, x_offset, z_offset, A, AA, AAA);
// 		L1(m_, x_, x_offset, A, z);
// 		L2(m_, AA, z, zz);
// 		L3(m_, z_, z_offset, AAA, zz);
// 	}
// }
