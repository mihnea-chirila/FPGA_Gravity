#define MAX_SIZE 64
#define VEC 16
#define PAR_FACTOR 4
// #define FIXED_BITS 3

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
static void L1(__global int *m_,
__global int *x_,
int x_offset,
int *A,
int *z,
int FIXED_BITS/*,
int *g_min,
int *g_max*/) {
	// __local 
	int local_in[784] __attribute__((xcl_array_partition(cyclic,PAR_FACTOR,1)));


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

  	int i, j, k, flag=0;

	long res[PAR_FACTOR];

	// long part_s[VEC] __attribute__((xcl_array_partition(complete,1)));
	// long temp[784] __attribute__((xcl_array_partition(cyclic,64,1)));

  	__attribute__((xcl_pipeline_loop(1)))
  	MAC1_ADD_1: for (i=0; i<100; ++i) 
    	z[i] = m_[78400+i];

	// for (i=0; i<100; ++i) 
	// 	printf("z[%d]: %d\t",i,z[i]);

  	// __attribute__((xcl_pipeline_loop))
  	// MAC1_1: for (i=0; i<100; ++i) {
	// 	__attribute__((opencl_unroll_hint(16)))
	// 	PAR_SUM_1: for (k=0; k<VEC; k++) {
	// 		part_s[k] = 0;
	// 	}
	// 	int *subA = &A[784*i];
	// 	__attribute__((xcl_pipeline_loop))
    // 	MAC1_1_0: for (j=0; j<784; j++) {
	// 		temp[j] = subA[j] * local_in[j];
	// 		// temp[j] /= 1000000;
	// 	}
    // 	//__attribute__((xcl_pipeline_loop))
	// 	__attribute__((opencl_unroll_hint(1)))
    // 	MAC1_1_1: for (j=0; j<784; j+=VEC) {
	// 		__attribute__((opencl_unroll_hint(16)))
	// 		PAR_SUM_1_1: for (k=0; k<VEC; k++) {
	// 			//part_s[k] += A[i * 784 + j + k] * local_in[j + k];
	// 			part_s[k] += temp[j+k];
	// 		}	
    // 	}
	// 	long out = 0;
	// 	__attribute__((opencl_unroll_hint(16)))
	// 	PAR_SUM_1_2: for (k=0; k<VEC; k++) {
	// 			out += part_s[k];
	// 	}
	// 	z[i] += out;
  	// }

  	__attribute__((xcl_pipeline_loop))
  	MAC1_1: for (j=0; j<784; ++j) {
  		__attribute__((xcl_pipeline_loop))
    	MAC1_1_1: for (i=0; i<100; i+=PAR_FACTOR) {
			__attribute__((opencl_unroll_hint(PAR_FACTOR)))
			MAC1_1_1_1: for (k=0; (k<PAR_FACTOR)&&((i+k)<100); k++)
			{
				// if ((j == 202) && (i == 0)) {
				// 	printf("pre_z[%d]: %d\t",i,z[i]);
				// 	printf("A[%d]: %d\t",i*784+j,A[i * 784 + j]);
				// 	printf("local_in[%d]: %d\t",j,local_in[j]);
				// }
				res[k] = A[(i+k) * 784 + j] * local_in[j];
				// if ((i == 0) && (res != 0)) {
				// 	printf("at j: %d, res: %ld\t", j, res);
				// }
				// res /= 1000000;
				z[i+k] += res[k] >> FIXED_BITS;
				// if(z[i] > *g_max) *g_max = z[i];
				// if(z[i] < *g_min) *g_min = z[i];
				// if ((local_in[j] != 0) && (flag == 0)) {
				// if ((i == 0) && (j == 202)) {
				// 	printf("res: %ld\t", res);
				// 	printf("post_z[%d]: %d\n",i,z[i]);
				// 	flag++;
				// }
			}
    	}
	}

	/*ADD + RELU*/
	__attribute__((opencl_unroll_hint(PAR_FACTOR*2)))
	MAC1_1_2: for (i=0; i<100; ++i)	{
		// z[i] /= 1000000;
		if (z[i] <= 0) 
			z[i] = 0;
		// printf("z[%d]: %d\t",i,z[i]);
	}
}
  	// __attribute__((xcl_pipeline_loop(1)))
  	// WB_1: for(int j = 0; j < 100; j++)
    // 	// m_[720016+j] = z[j];
  	// 	m_[180004+j] = z[j];

static void L2(__global int *m_,
int *AA,
int *z,
int *zz,
int FIXED_BITS/*,
int *g_min,
int *g_max*/) {
  	/* MAC1 */
  	// __local 
	// float zz[100] __attribute__((xcl_array_partition(cyclic,25,1)));
  	// __attribute__((xcl_pipeline_loop))
  	// for(int j = 0; j < 100; j++){
  		// zz[j] = m_[j+720816];
  		//	zz[j] = m_[j+180204];
  	//}
	int i, j, k;
	long temp[PAR_FACTOR];

	// for (i=0; i<100; ++i) 
	// 	printf("z[%d]: %d\t",i,z[i]);

	__attribute__((xcl_pipeline_loop(1)))
  	MAC1_ADD_2: for (i=0; i<100; ++i)
  		zz[i] = m_[88500+i];

	// for (i=0; i<100; ++i) 
	// 	printf("zz[%d]: %d\t",i,zz[i]);

  	__attribute__((xcl_pipeline_loop))
  	MAC1_2: for (j=0; j<100; ++j) {
  		__attribute__((xcl_pipeline_loop))
  		MAC1_2_1: for (i=0; i<100; i+=PAR_FACTOR) {
			__attribute__((opencl_unroll_hint(PAR_FACTOR)))
			MAC1_2_1_1: for (k=0; (k<PAR_FACTOR)&&((i+k)<100); k++)
			{
				// if ((i == 0) && (j == 0)) {
				// 	printf("pre_zz[%d]: %d\n",i,zz[i]);
				// 	printf("AA[%d]: %d\n",i*100+j,AA[i * 100 + j]);
				// 	printf("z[%d]: %d\n",j,z[j]);
				// }
				temp[k] = AA[(i+k) * 100 + j] * z[j];// / 1000000;
				zz[i+k] += temp[k] >> FIXED_BITS;// / 1000000;
				// if(zz[i] > *g_max) *g_max = zz[i];
				// if(zz[i] < *g_min) *g_min = zz[i];
				// if ((i == 0) && (j == 0)) {
				// 	printf("temp: %ld\n", temp);
				// 	printf("post_zz[%d]: %d\n",i,zz[i]);
				// }
			}
  		}
	}
	//ADD + RELU
	__attribute__((opencl_unroll_hint(PAR_FACTOR)))
	MAC1_2_2: for (i=0; i<100; ++i)	{
		// zz[i] /= 1000000;
		// printf("zz[%d]: %d\t",i,zz[i]);
		if (zz[i] <= 0) 
			zz[i] = 0;
	}
}

	// __attribute__((xcl_pipeline_loop(1)))
	// WB_2: for(int j = 0; j < 100; j++)
  	// 	// m_[720816+j] = zz[j];
  	// 	m_[180204+j] = zz[j];

static void L3(__global int *m_,
__global int *z_,
int z_offset,
int *AAA,
int *zz,
int FIXED_BITS/*,
int *g_min,
int *g_max*/) {
  	// MAC1
  	// __local 
	int zzz[10]; //__attribute__((xcl_array_partition(complete,1)));
	int i, j, k;
	long temp[PAR_FACTOR];

	__attribute__((xcl_pipeline_loop(1)))
  	MAC1_ADD_3: for (i=0; i<10; ++i)
  		//ADD
  		zzz[i] = m_[i+89600];

	// for (i=0; i<10; ++i) 
	// 	printf("zzz[%d]: %d\t",i,zzz[i]);

  	__attribute__((xcl_pipeline_loop))
  	MAC1_3: for (j=0; j<100; ++j) {
  		__attribute__((xcl_pipeline_loop))
  		MAC1_3_1: for (i=0; i<10; i+=PAR_FACTOR) {
			__attribute__((opencl_unroll_hint(PAR_FACTOR)))
			MAC1_3_1_1: for (k=0; (k<PAR_FACTOR)&&((i+k)<10); k++)
			{
				temp[k] = AAA[(i+k) * 100 + j] * zz[j];// / 1000000;
				zzz[i+k] += temp[k] >> FIXED_BITS;// / 1000000;
				// if(zzz[i] > *g_max) *g_max = zzz[i];
				// if(zzz[i] < *g_min) *g_min = zzz[i];
				// zzz[i] += m_[88600+ i * 100 + j] * zz[j];
			}
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
  	int max, sum=0;
	max = zzz[0];// / 1000000;
  	__attribute__((xcl_pipeline_loop))
  	SOFTMAX_1: for (i=0; i<10; ++i) {
		// zzz[i] /= 1000000;
		// printf("zzz[%d]: %d\t",i,zzz[i]);
  		if (max < zzz[i]) {
  			max = zzz[i];
  		}
  	}

	max = max >> FIXED_BITS;
	int aux[10]; //__attribute__((xcl_array_partition(complete,1)));
	int exp_z[10]; //__attribute__((xcl_array_partition(complete,1)));
	int exp_max= max>21?INT_MAX:exp(max);
  	
	__attribute__((opencl_unroll_hint(PAR_FACTOR)))
  	EXP_Z: for (i=0; i<10; ++i) {
		aux[i] = zzz[i] >> FIXED_BITS;
		if (aux[i] > 21){
			exp_z[i] = INT_MAX;
		}else{
			exp_z[i] = exp(aux[i]);
		}
		
  	}

  	__attribute__((xcl_pipeline_loop))
  	SOFTMAX_2: for (i=0; i<10; ++i) {
  		//zzz[i] -= max;
  		// sum += (float)exp(zzz[i]-max);
		sum += exp_z[i];
  	}

	sum /= exp_max;

  	//__attribute__((opencl_unroll_hint(10)))
	__attribute__((nounroll))
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
void run_activate(__global int *m_, __global int *x_, __global int *z_,
int x_size, int z_size, int test_n, int FIXED_BITS) {
	
	// int g_min = INT_MAX;
	// int g_max = INT_MIN;
	int x_offset=0, z_offset=0;
  	// __local 
	int A[78400] __attribute__((xcl_array_partition(block,100,1)));
  	__attribute__((xcl_pipeline_loop(1)))
  	RD_A: for(int j = 0; j < 78400; j++){
    	A[j] = m_[j];
  	}

  	// __local 
	int AA[10000] __attribute__((xcl_array_partition(block,100,1)));
 	__attribute__((xcl_pipeline_loop(1)))
  	RD_AA: for(int j = 0; j < 10000; j++){
    	// AA[j] = m_[j+314000];
  		AA[j] = m_[j+78500];
  	}

  	// __local 
	int AAA[1000] __attribute__((xcl_array_partition(block,100,1)));
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

		int z[100] __attribute__((xcl_array_partition(cyclic,PAR_FACTOR*3,1)));
		int zz[100] __attribute__((xcl_array_partition(cyclic,PAR_FACTOR*3,1)));
		// if (i == 162){
		// activate(m_, x_, z_, x_offset, z_offset, A, AA, AAA);
		L1(m_, x_, x_offset, A, z, FIXED_BITS/*, &g_min, &g_max*/);
		// printf("#%d:L1 finished ", i);
		L2(m_, AA, z, zz, FIXED_BITS/*, &g_min, &g_max*/);
		// printf("#%d:L2 finished ", i);
		L3(m_, z_, z_offset, AAA, zz, FIXED_BITS/*, &g_min, &g_max*/);
		// printf("#%d:L3 finished ", i);
		// }
	}
	// printf("g_min, g_max:%d, %d\n", g_min, g_max);
}