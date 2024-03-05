#define MAX_SIZE 64

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
	// __local 
	float local_in[784] __attribute__((xcl_array_partition(cyclic,2,1)));


  	// __global float *p = m_+179220;

  	__attribute__((xcl_pipeline_loop(1)))
  	RD_X: for(int j = 0; j < 784; j++){
    	local_in[j] = x_[x_offset+j];
    	//*(m_+j/*+716880*/) = local_in[j];
    	//*(m_+j+179220) = local_in[j];
    	// p[j] = local_in[j];
  	}

  	/* MAC1 */
  	// __local 
	// float z[100] __attribute__((xcl_array_partition(cyclic,50,1)));

  	int i, j, flag=0;

  	__attribute__((xcl_pipeline_loop(1)))
  	MAC1_ADD_1: for (i=0; i<100; ++i) 
    	z[i] = m_[78400+i];

  	__attribute__((xcl_pipeline_loop))
  	MAC1_1: for (j=0; j<784; ++j) {
    	__attribute__((opencl_unroll_hint(2)))
    	MAC1_1_1: for (i=0; i<100; ++i) {
			// if ((j == 202) && (i == 0)) {
			// 	printf("pre_z[%d]: %f\t",i,z[i]);
			// 	printf("A[%d]: %f\t",i*784+j,A[i * 784 + j]);
			// 	printf("local_in[%d]: %f\t",j,local_in[j]);
			// }
      		z[i] += A[i * 784 + j] * local_in[j];
			// if ((local_in[j] != 0) && (flag == 0)) {
			// if ((i == 0) && (j == 202)) {
			// 	printf("post_z[%d]: %f\n",i,z[i]);
			// 	flag++;
			// }
    	}
  	}

	// for (i=0; i<100; ++i) 
	// 	printf("z[%d]: %f\t",i,z[i]);

	/*ADD + RELU*/
	__attribute__((opencl_unroll_hint(2)))
	MAC1_1_2: for (i=0; i<100; ++i) {
		if (z[i] <= 0.0) 
			z[i] = 0.0;
		// printf("z[%d]: %f\t",i,z[i]);
	}
}
  	// __attribute__((xcl_pipeline_loop(1)))
  	// WB_1: for(int j = 0; j < 100; j++)
    // 	// m_[720016+j] = z[j];
  	// 	m_[180004+j] = z[j];

static void L2(__global float *m_,
float *AA,
float *z,
float *zz) {
  	/* MAC1 */
  	// __local 
	// float zz[100] __attribute__((xcl_array_partition(cyclic,25,1)));
  	// __attribute__((xcl_pipeline_loop))
  	// for(int j = 0; j < 100; j++){
  		// zz[j] = m_[j+720816];
  		//	zz[j] = m_[j+180204];
  	//}
	int i, j;

	// for (i=0; i<100; ++i) 
	// 	printf("z[%d]: %f\t",i,z[i]);

	__attribute__((xcl_pipeline_loop(1)))
  	MAC1_ADD_2: for (i=0; i<100; ++i)
  		zz[i] = m_[88500+i];

  	__attribute__((xcl_pipeline_loop))
  	MAC1_2: for (j=0; j<100; ++j) {
  		__attribute__((opencl_unroll_hint(2)))
  		MAC1_2_1: for (i=0; i<100; ++i) {
			// if ((i == 0) && (j == 0)) {
			// 	printf("pre_zz[%d]: %f\n",i,zz[i]);
			// 	printf("AA[%d]: %f\n",i*100+j,AA[i * 100 + j]);
			// 	printf("z[%d]: %f\n",j,z[j]);
			// }
  			zz[i] += AA[i * 100 + j] * z[j];
			// if ((i == 0) && (j == 0)) {
			// 	printf("post_zz[%d]: %f\n",i,zz[i]);
			// }
  		}
	}
	//ADD + RELU
	__attribute__((opencl_unroll_hint(2)))
	MAC1_2_2: for (i=0; i<100; i++) {
		// printf("zz[%d]: %f\t",i,zz[i]);
		if (zz[i] <= 0.0) 
			zz[i] = 0.0;
	}
}

	// __attribute__((xcl_pipeline_loop(1)))
	// WB_2: for(int j = 0; j < 100; j++)
  	// 	// m_[720816+j] = zz[j];
  	// 	m_[180204+j] = zz[j];

static void L3(__global float *m_,
__global float *z_,
int z_offset,
float *AAA,
float *zz) {
  	// MAC1
  	// __local 
	float zzz[10] __attribute__((xcl_array_partition(complete,1)));
	int i, j;

	__attribute__((xcl_pipeline_loop(1)))
  	MAC1_ADD_3: for (i=0; i<10; ++i)
  		//ADD
  		zzz[i] = m_[i+89600];

  	__attribute__((xcl_pipeline_loop))
  	MAC1_3: for (j=0; j<100; ++j) {
  		__attribute__((opencl_unroll_hint(2)))
  		MAC1_3_1: for (i=0; i<10; ++i) {
  			zzz[i] += AAA[i * 100 + j] * zz[j];
  			// zzz[i] += m_[88600+ i * 100 + j] * zz[j];
  		}
  	}

  /* ADD */
  // __attribute__((xcl_pipeline_loop))
  // for(int j = 0; j < 10; j++){
  //   // B[j] = m_[j+358400];
  //   B[j] = m_[j+89600];
  // }
  // __attribute__((xcl_pipeline_loop))
  // for (i=0; i<10; ++i) {
  //   zzz[i] += B[i];
  // }

  	/* SOFTMAX */
  	float max=zzz[0], sum=0.0;
  	__attribute__((xcl_pipeline_loop))
  	SOFTMAX_1: for (i=1; i<10; ++i) {
		// printf("zzz[%d]: %f\t",i,zzz[i]);
  		if (max < zzz[i]) {
  			max = zzz[i];
  		}
  	}

	float exp_z[10], exp_max=(float)exp(max);

	__attribute__((opencl_unroll_hint(2)))
  	EXP_Z_PRE: for (i=0; i<10; ++i) {
		exp_z[i] = -max;
  	}
  	
	__attribute__((opencl_unroll_hint(2)))
  	EXP_Z: for (i=0; i<10; ++i) {
		exp_z[i] += (float)exp(zzz[i]);
  	}

  	__attribute__((xcl_pipeline_loop))
  	SOFTMAX_2: for (i=0; i<10; ++i) {
  		//zzz[i] -= max;
  		// sum += (float)exp(zzz[i]-max);
		sum += exp_z[i];
  	}
	//printf("Sum, exp_max:%f %f\n", sum, exp_max);
	sum /= exp_max;

  	//__attribute__((opencl_unroll_hint(10)))
	//__attribute__((nounroll))
	__attribute__((xcl_pipeline_loop))
  	WB_3:for (i=0; i<10; ++i) {
  		zzz[i] = exp_z[i]/sum;
		// zzz[i] = (float)exp(zzz[i]) / sum;
  		// m_[180404+i] = (float)exp(zzz[i]) / sum;
  	}

  	__attribute__((xcl_pipeline_loop(1)))
	WB_Z:for (i=0; i<10; ++i) {
		// m_[180404+i] = zzz[i];
		z_[z_offset+i] = zzz[i];
  	}
}


__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
// __attribute__ ((xcl_dataflow))
void run_activate(__global float *m_, __global float *x_, __global float *z_,
int x_size, int z_size, int test_n) {
	
	int x_offset=0, z_offset=0;
  	// __local 
	float A[78400] __attribute__((xcl_array_partition(block,784,1)));
  	__attribute__((xcl_pipeline_loop(1)))
  	RD_A: for(int j = 0; j < 78400; j++){
    	A[j] = m_[j];
  	}

  	// __local 
	float AA[10000] __attribute__((xcl_array_partition(block,100,1)));
 	__attribute__((xcl_pipeline_loop(1)))
  	RD_AA: for(int j = 0; j < 10000; j++){
    	// AA[j] = m_[j+314000];
  		AA[j] = m_[j+78500];
  	}

  	// __local 
	float AAA[1000] __attribute__((xcl_array_partition(block,100,1)));
  	__attribute__((xcl_pipeline_loop(1)))
  	RD_AAA: for(int j = 0; j < 1000; j++){
  		// AAA[j] = m_[j+354400];
  		AAA[j] = m_[j+88600];
  	}
  	
	// __attribute__((xcl_pipeline_loop(1)))
	__attribute__((opencl_unroll_hint(1)))
	ACT: for(int i=0; i<test_n; i++){
		x_offset = x_size * i;
		z_offset = z_size * i;

		float z[100] __attribute__((xcl_array_partition(cyclic,34,1)));
		float zz[100] __attribute__((xcl_array_partition(cyclic,34,1)));

		// activate(m_, x_, z_, x_offset, z_offset, A, AA, AAA);
		L1(m_, x_, x_offset, A, z);
		L2(m_, AA, z, zz);
		L3(m_, z_, z_offset, AAA, zz);
	}
}