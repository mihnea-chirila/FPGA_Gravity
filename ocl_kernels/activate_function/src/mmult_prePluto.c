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

  	//__attribute__((xcl_pipeline_loop(1)))
  	RD_X: for(int j = 0; j < 784; j++){
    	local_in[j] = x_[x_offset+j];
  	}

  	int i, j, k;
	float part_s[VEC];// __attribute__((xcl_array_partition(complete,1)));
	float temp[784];// __attribute__((xcl_array_partition(cyclic,64,1)));

	#pragma scop
  	//__attribute__((xcl_pipeline_loop))
  	MAC1_1: for (i=0; i<100; ++i) {
		z[i] = m_[78400+i];

		//__attribute__((opencl_unroll_hint(16)))
		PAR_SUM_1: for (k=0; k<VEC; k++) {
			part_s[k] = 0;
		}
		float *subA = &A[784*i];

		//__attribute__((xcl_pipeline_loop))
    	MAC1_1_0: for (j=0; j<784; j++) {
			temp[j] = subA[j] * local_in[j];
		}

    	//__attribute__((xcl_pipeline_loop))
		//__attribute__((opencl_unroll_hint(1)))
    	MAC1_1_1: for (j=0; j<784; j+=VEC) {
			//__attribute__((opencl_unroll_hint(16)))
			PAR_SUM_1_1: for (k=0; k<VEC; k++) {
				part_s[k] += temp[j+k];
			}	
    	}
		float out = 0;

		//__attribute__((opencl_unroll_hint(16)))
		PAR_SUM_1_2: for (k=0; k<VEC; k++) {
				out += part_s[k];
		}
		z[i] += out;

		/*ADD + RELU*/
		if (z[i] <= 0.0) 
			z[i] = 0.0;
  	}
	#pragma endscop
}

static void L2(__global float *m_,
float *AA,
float *z,
float *zz) {

	int i, j;

	#pragma scop
	// __attribute__((xcl_pipeline_loop(1)))
  	MAC1_ADD_2: for (i=0; i<100; ++i)
  		zz[i] = m_[88500+i];

  	// __attribute__((xcl_pipeline_loop))
  	MAC1_2: for (j=0; j<100; ++j) {
  		// __attribute__((opencl_unroll_hint(17)))
  		MAC1_2_1: for (i=0; i<100; ++i) {
  			zz[i] += AA[i * 100 + j] * z[j];
  		}
	}

	//ADD + RELU
	// __attribute__((opencl_unroll_hint(17)))
	MAC1_2_2: for (i=0; i<100; i++)
		if (zz[i] <= 0.0) 
			zz[i] = 0.0;
	#pragma endscop
}

static void L3(__global float *m_,
__global float *z_,
int z_offset,
float *AAA,
float *zz) {

	float zzz[10];// __attribute__((xcl_array_partition(complete,1)));;
	int i, j;

	#pragma scop
	// __attribute__((xcl_pipeline_loop(1)))
  	MAC1_ADD_3: for (i=0; i<10; ++i)
  		//ADD
  		zzz[i] = m_[i+89600];

  	// __attribute__((xcl_pipeline_loop))
  	MAC1_3: for (j=0; j<100; ++j) {
  		// __attribute__((opencl_unroll_hint(10)))
  		MAC1_3_1: for (i=0; i<10; ++i) {
  			zzz[i] += AAA[i * 100 + j] * zz[j];
  		}
  	}

  	/* SOFTMAX */
  	float max=zzz[0], sum=0.0;
  	// __attribute__((xcl_pipeline_loop))
  	SOFTMAX_1: for (i=1; i<10; ++i) {
  		if (max < zzz[i]) {
  			max = zzz[i];
  		}
  	}

	float exp_z[10], exp_max=(float)exp(max);
  	
	// __attribute__((opencl_unroll_hint(10)))
  	EXP_Z: for (i=0; i<10; ++i) {
		exp_z[i] = (float)exp(zzz[i]);
  	}

  	// __attribute__((xcl_pipeline_loop))
  	SOFTMAX_2: for (i=0; i<10; ++i) {
		sum += exp_z[i];
  	}

	sum /= exp_max;

  	//__attribute__((opencl_unroll_hint(10)))

	// __attribute__((nounroll))
	// __attribute__((xcl_pipeline_loop))
  	WB_3:for (i=0; i<10; ++i) {
  		zzz[i] = exp_z[i]/sum;
  	}

  	// __attribute__((xcl_pipeline_loop(1)))
	WB_Z:for (i=0; i<10; ++i) {
		z_[z_offset+i] = zzz[i];
  	}
	#pragma endscop
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