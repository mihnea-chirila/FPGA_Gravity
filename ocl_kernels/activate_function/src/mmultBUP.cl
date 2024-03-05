#define MAX_SIZE 64

// Tripcount identifiers
__constant int c_size = MAX_SIZE;

//static void MAC1()

// __kernel __attribute__((reqd_work_group_size(1, 1, 1)))
//__attribute__ ((xcl_dataflow))
static
void activate(__global float *m_, __global float *x_, __global float *z_, int x_offset, int z_offset) {

	// __local 
	float local_in[784];//__attribute__((xcl_array_partition(cyclic,8,1)));

  	// __local 
	float A[78400];//__attribute__((xcl_array_partition(cyclic,8,1)));
  	__attribute__((xcl_pipeline_loop))
  	for(int j = 0; j < 78400; j++){
    	A[j] = m_[j];
  	}
  	__global float *p = m_+179220;

  	__attribute__((xcl_pipeline_loop))
  	COPY_X: for(int j = 0; j < 784; j++){
    	local_in[j] = x_[x_offset+j];
    	//*(m_+j/*+716880*/) = local_in[j];
    	//*(m_+j+179220) = local_in[j];
    	p[j] = local_in[j];
  	}

  	/* MAC1 */
  	// __local 
	float z[100];//__attribute__((xcl_array_partition(cyclic,8,1)));

  // __local float A[78400];//__attribute__((xcl_array_partition(cyclic,8,1)));
  // __attribute__((xcl_pipeline_loop))
  // for(int j = 0; j < 78400; j++){
  //   A[j] = m_[j];
  // }

  	int i, j;

  	__attribute__((xcl_pipeline_loop))
  	for (i=0; i<100; ++i) {
    	z[i] = 0.0;
    	__attribute__((opencl_unroll_hint(2)))
    	for (j=0; j<784; ++j) {
      		z[i] += A[i * 784 + j] * local_in[j];
    	}
    	/*ADD + RELU*/
    	z[i] += m_[78400+i];
    	if (z[i] <= 0.0) {
      		z[i] = 0.0;
    	}
  	}

	// for (i=0; i<100; ++i) 
	// 	printf("z[%d]: %f\t",i,z[i]);

  /* ADD + RELU */
  //__local float B[100];//__attribute__((xcl_array_partition(cyclic,4,1)));
  // __attribute__((xcl_pipeline_loop))
  // for(int j = 0; j < 100; j++){
  //   // B[j] = m_[313600+j];
  //   B[j] = m_[78400+j];
  // }

/*  __attribute__((opencl_unroll_hint(4)))
  for (i=0; i<100; ++i) {
    // z[i] += B[i];
    z[i] += m_[78400+i];
    if (z[i] <= 0.0) {
      z[i] = 0.0;
    }
  }
 */
  	__attribute__((xcl_pipeline_loop))
  	for(int j = 0; j < 100; j++)
    	// m_[720016+j] = z[j];
  		m_[180004+j] = z[j];


  	/* MAC1 */
  	// __local 
	float zz[100];//__attribute__((xcl_array_partition(cyclic,4,1)));
  	// __attribute__((xcl_pipeline_loop))
  	// for(int j = 0; j < 100; j++){
  		// zz[j] = m_[j+720816];
  		//	zz[j] = m_[j+180204];
  	//}
  	// __local 
	float AA[10000];//__attribute__((xcl_array_partition(cyclic,4,1)));
 	__attribute__((xcl_pipeline_loop))
  	for(int j = 0; j < 10000; j++){
    	// AA[j] = m_[j+314000];
  		AA[j] = m_[j+78500];
  	}

  	__attribute__((xcl_pipeline_loop))
  	for (i=0; i<100; ++i) {
  		zz[i] = 0.0;
  		__attribute__((opencl_unroll_hint(1)))
  		for (j=0; j<100; ++j) {
  			zz[i] += AA[i * 100 + j] * z[j];
  		}
  		//ADD + RELU
  		zz[i] += m_[88500+i];
  		if (zz[i] <= 0.0) {
  			zz[i] = 0.0;
  		}
  	}

  /* ADD + RELU */
  // __attribute__((xcl_pipeline_loop))
  // for(int j = 0; j < 100; j++){
  //   // B[j] = m_[354000+j];
  //   B[j] = m_[88500+j];
  // }
/*  __attribute__((opencl_unroll_hint(4)))
  for (i=0; i<100; ++i) {
    // zz[i] += B[i];
    zz[i] += m_[88500+i];
    if (zz[i] <= 0.0) {
      zz[i] = 0.0;
    }
  }
*/
	__attribute__((xcl_pipeline_loop))
	for(int j = 0; j < 100; j++)
  		// m_[720816+j] = zz[j];
  		m_[180204+j] = zz[j];

  	// MAC1
  	// __local 
	float zzz[10];//__attribute__((xcl_array_partition(cyclic,10,1)));;
  // __attribute__((xcl_pipeline_loop))
  // for(int j = 0; j < 10; j++){
  //   // zzz[j] = m_[j+721616];
  //   zzz[j] = m_[j+180404];
  // }
  	// __local 
	float AAA[1000];//__attribute__((xcl_array_partition(cyclic,10,1)));
  	__attribute__((xcl_pipeline_loop))
  	for(int j = 0; j < 1000; j++){
  		// AAA[j] = m_[j+354400];
  		AAA[j] = m_[j+88600];
  	}

  	__attribute__((opencl_unroll_hint(10)))
  	for (i=0; i<10; ++i) {
  		zzz[i] = 0.0;
  		__attribute__((xcl_pipeline_loop))
  		for (j=0; j<100; ++j) {
  			//zzz[i] += AAA[i * 100 + j] * zz[j];
  			zzz[i] += m_[88600+ i * 100 + j] * zz[j];
  		}
  		//ADD
  		zzz[i] += m_[i+89600];
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
  	for (i=1; i<10; ++i) {
  		if (max < zzz[i]) {
  			max = zzz[i];
  		}
  	}
  	__attribute__((xcl_pipeline_loop))
  	for (i=0; i<10; ++i) {
  		//zzz[i] -= max;
  		sum += (float)exp(zzz[i]-max);
  	}
  	__attribute__((xcl_pipeline_loop))
  	for (i=0; i<10; ++i) {
  		// zzz[i] = (float)exp(zzz[i]) / sum;
  		m_[180404+i] = (float)exp(zzz[i]) / sum;
  		// printf("z[%d]: %f\t",i,zzz[i]);
		z_[z_offset+i] = m_[180404+i];
  	}

	// __attribute__((xcl_pipeline_loop))
	// for(int j = 0; j < 10; j++)
	// 	// m_[721616+j] = zzz[j];
	// 	// m_[180404+j] = zzz[j];
	// 	printf("z_[%d]: %f\t",z_offset+j,z_[z_offset+j]);
	// 	// printf("m[%d]: %f\t",180404+j,m_[180404+j]);
	// printf("\n");
}


__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
__attribute__ ((xcl_dataflow))
void run_activate(__global float *m_, __global float *x_, __global float *z_,
int x_size, int z_size, int test_n) {
	
	int x_offset=0, z_offset=0;

	for(int i=0; i<test_n; i++){
		x_offset = x_size * i;
		z_offset = z_size * i;

		activate(m_, x_, z_, x_offset, z_offset);
	}
}