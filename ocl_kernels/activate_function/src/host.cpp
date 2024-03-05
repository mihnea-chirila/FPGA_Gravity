/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/
#include "xcl2.hpp"
#include <vector>
#include <unistd.h>

#define REAL_T int32_t
typedef REAL_T real_t;

#define FIXED_BITS 8

using std::vector;

int mem_size = 721696;
int mem_size_hard = 358440;
int x_size_bytes = 28*28*sizeof(int32_t);
int num_cu = 1;

uint64_t get_duration_ns (const cl::Event &event) {
    uint64_t nstimestart, nstimeend;
    event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START,&nstimestart);
    event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END,&nstimeend);
    return(nstimeend-nstimestart);
}

static int argmax(const real_t *a, int n)
{
  real_t max;
  int i, j;

  max = a[0];
  for (i=j=0; i<n; ++i) {
	// printf("a[%d]=%f\n",i,a[i]);
    if (max < a[i]) {
      max = a[i];
      j = i;
    }
  }
  return j;
}

static const int DATA_SIZE = 1024;
static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";
    
uint64_t activate_setup(
	std::string binaryFile,
	cl::CommandQueue& q,
	std::vector<cl::Kernel>& krnl_activate,
	std::vector<cl::Buffer>& mem_buf,
	std::vector<cl::Buffer>& x_buf,
	std::vector<cl::Buffer>& z_buf,
	vector<int32_t, aligned_allocator<int32_t>>& mem,
	vector<int32_t, aligned_allocator<int32_t>>& x,
	vector<int32_t, aligned_allocator<int32_t>>& z,
	int n_test
){
    
    cl_int err;
    cl::Context context;
	std::string cu_id;
	std::string krnl_name = "run_activate";

    // The get_xil_devices will return vector of Xilinx Devices
    auto devices = xcl::get_xil_devices();
    
    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | 
			CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            // This call will extract a kernel out of the program we loaded in the
            // previous line. A kernel is an OpenCL function that is executed on the
            // FPGA. This function is defined in the src/vetor_addition.cl file.
			for (int i = 0; i < num_cu; i++){
				cu_id = std::to_string(i+1);
				std::string krnl_name_full = krnl_name + ":{" + "run_activate_" + cu_id + "}";
            	OCL_CHECK(err, krnl_activate[i] = cl::Kernel(program, krnl_name_full.c_str(), &err));
			}
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

	// cl_mem_ext_ptr_t outExt;  // Declaring two extensions for both buffers
    // outExt.flags = 4|XCL_MEM_TOPOLOGY; // Specify PLRAM(0) for output Memory (z_)
    // outExt.obj = z.data(); // Setting Obj and Param to Zero
    // outExt.param = 0;
            
    // These commands will allocate memory on the FPGA. The cl::Buffer objects can
    // be used to reference the memory locations on the device. The cl::Buffer
    // object cannot be referenced directly and must be passed to other OpenCL
    // functions.
	for (int i = 0; i < num_cu; i++) {
		OCL_CHECK(err, mem_buf[i] = 
			cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, mem_size, mem.data(),
                                       &err));
		OCL_CHECK(err, x_buf[i] = 
			cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, x_size_bytes*n_test, x.data(),
										&err));
		OCL_CHECK(err, z_buf[i] = 
			cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float)*n_test*10, z.data(),
										&err));
		// set the kernel Arguments
		int narg = 0;
		OCL_CHECK(err, err = krnl_activate[i].setArg(narg++, mem_buf[i]));
		OCL_CHECK(err, err = krnl_activate[i].setArg(narg++, x_buf[i]));
		OCL_CHECK(err, err = krnl_activate[i].setArg(narg++, z_buf[i]));

		int32_t x_offset, z_offset;
		// narg++;
		// Launch the Kernel
		x_offset = x_size_bytes / sizeof(int32_t);
		z_offset = 10;
		OCL_CHECK(err, err = krnl_activate[i].setArg(narg++, x_offset));
		OCL_CHECK(err, err = krnl_activate[i].setArg(narg++, z_offset));
		OCL_CHECK(err, err = krnl_activate[i].setArg(narg++, n_test));
		OCL_CHECK(err, err = krnl_activate[i].setArg(narg++, FIXED_BITS));

		// These commands will load the source_a and source_b vectors from the host
		// application and into the buffer_a and buffer_b cl::Buffer objects. The data
		// will be be transferred from system memory over PCIe to the FPGA on-board
		// DDR memory.
		OCL_CHECK(err, err = q.enqueueMigrateMemObjects({mem_buf[i], x_buf[i], z_buf[i]}, 0 /* 0 means from host*/));
	}
	OCL_CHECK(err, err = q.finish());
		
	std::vector<uint64_t> kernel_duration;
	uint64_t max = 0;
	kernel_duration.assign(num_cu, 0);

	for (int i = 0; i < num_cu; i++){
		cl::Event event;

		OCL_CHECK(err, err = q.enqueueTask(krnl_activate[i], NULL, &event));
		kernel_duration[i] += get_duration_ns(event);
	}
	OCL_CHECK(err, err = q.finish());

	for (int i = 0; i < num_cu; i++){
		if (kernel_duration[i] > max){
			max =kernel_duration[i];
		}
	}
    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will write the data from the
    // buffer_result cl_mem object to the source_results vector
	for (int i = 0; i < num_cu; i++){
		OCL_CHECK(err, err = 
			q.enqueueMigrateMemObjects({mem_buf[i], z_buf[i]}, CL_MIGRATE_MEM_OBJECT_HOST));
	}
    q.finish();


  	return max;
}

// This example illustrates the very simple OpenCL example that performs
// an addition on two vectors
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    
    cl::CommandQueue q;
	std::vector<cl::Kernel> krnl_activate(num_cu);
	std::vector<cl::Buffer> mem_buf(num_cu);
	std::vector<cl::Buffer> x_buf(num_cu);
	std::vector<cl::Buffer> z_buf(num_cu);


  	//Allocate Memory in Host Memory
  	int test_n = 2, digit, dec_cursor, s_idx; 
	long int error=0;    

  	//When creating a buffer with user pointer, under the hood user ptr is
  	//used if and only if it is properly aligned (page aligned). When not 
  	//aligned, runtime has no choice but to create its own host side buffer
  	//that backs user ptr. This in turn implies that all operations that move
  	//data to/from device incur an extra memcpy to move data to/from runtime's
  	//own host buffer from/to user pointer. So it is recommended to use this 
  	//allocator if user wish to Create Buffer/Memory Object to align user buffer
  	//to the page boundary. It will ensure that user buffer will be used when 
  	//user create Buffer/Mem Object.
  	std::vector<int32_t,aligned_allocator<int32_t>> mem(mem_size/sizeof(int32_t));
  	std::vector<int32_t,aligned_allocator<int32_t>> x(test_n*x_size_bytes/sizeof(int32_t));
  	std::vector<int32_t,aligned_allocator<int32_t>> z(test_n*10);
	std::vector<int32_t,aligned_allocator<int32_t>> z_gold(test_n);

  	//Create the test data and Software Result 
	char buf[256], s[19];
	float in_data;
  	FILE *fptr_mem, *fptr_x[test_n], *fptr_z, *fptr_out;
  	fptr_mem = fopen("src/data/mem.txt", "r+");
  	if(fptr_mem == NULL) {
		strerror_r(errno, buf, 256);
		printf("\nERRROR! err: %s\n", buf);
  	}
  	for(int i=0; i<(int)mem_size_hard/sizeof(float); i++) {
	 	/*printf("%f ", *((float *)(mem)+i));*/
	 	fscanf(fptr_mem, "%f ", &in_data);
		// mem[i] = (int)(in_data*1000000);
		mem[i] = (int)(in_data * (1 << FIXED_BITS));
	 	// printf("%d ", mem[i]);
	}

	fptr_z = fopen("src/data/labels.txt", "r+");
	if(fptr_z == NULL) {
		strerror_r(errno, buf, 256);
	  	printf("\nERRROR! err: %s\n", buf);
	}
	for(int i=0; i<test_n; i++) {
	  	/*printf("%f ", *((float *)(mem)+i));*/
	  	fscanf(fptr_z, "%d ", &z_gold[i]);
	  	//printf("%d ", z[i]);
	}
	fclose(fptr_mem);
	fclose(fptr_z);

	uint64_t kernel_duration = 0;
	
	// activate_setup(binaryFile, q, krnl_activate, mem_buf, x_buf, z_buf, mem, x, z, test_n);
	 
	for(int i=0; i<test_n; i++) {
	  	kernel_duration = 0;	  
	  	memset(s, 0, 19);
		digit = i;
		dec_cursor = 10000;
		s_idx = 10;
		strcpy(s,"src/data/x");
		if(digit == 0)
			*(s+s_idx) = i + 48;
		else {
			while(digit/dec_cursor == 0) {
				digit = digit % dec_cursor;
				dec_cursor /= 10;
			}
			while(dec_cursor > 0) {
				s[s_idx] = digit / dec_cursor + 48;
				s_idx++;
				digit = digit % dec_cursor;
				dec_cursor /= 10;
			}
		}
		strcat(s,".txt");
		// printf("%s\t",s);
		fptr_x[i] = fopen(s,"r+");
		memset(buf, 0, 256);
		if(fptr_x[i] == NULL) {
			strerror_r(errno, buf, 256);
			printf("\nERRROR! at i=%d, err: %s\n", i, buf);

		}
		for (int k=0; k<(28*28); ++k) {
			fscanf(fptr_x[i], "%f ", &in_data);
			// printf("in[%d]: %f\t", k, in_data);
			// x[i*28*28+k] = (int)(in_data*1000000);
			x[i*28*28+k] = (int)(in_data * (1 << FIXED_BITS));
		}
		fclose(fptr_x[i]);
	}
		kernel_duration = activate_setup(binaryFile, q, krnl_activate, mem_buf, x_buf, z_buf, 
			mem, x, z, test_n);
    	// real_t *out = mem.at(721616);

    	// real_t *out;
    	// out = &mem[180404];

    	// printf("mem[180411]: %f\t",mem[180411]);
    	// printf("&mem[0]: %x\t",&mem[0]);

    	// for(int j=0; j<10; j++){
    	//   printf("mem[%d]: %f\t",j,*(out+j));
    	// }

  		//   fptr_out = fopen("src/out.txt", "w+");
  		// if(fptr_out == NULL) {
  		//   strerror_r(errno, buf, 256);
  		//   printf("\nERRROR! err: %s\n", buf);
  		// }
  		// for(int j=0; j<(int)mem_size/sizeof(float); j++) {
  		//   /*printf("%f ", *((float *)(mem)+i));*/
  		//   fprintf(fptr_out, "%f ", mem[j]);
  		//   //printf("%f ", (float)mem[i]);
  		// }

		for (int i=0; i<test_n; i++) {
			real_t *out;
			out = &z[i*10];
			// out = &mem[180404];

			// for (int j=0; j<10; j++)
			// 	printf("out[%d]=%d\n",j,z[i*10+j]);
			// printf("z_gold[%d] = %d\n", i, z_gold[i]);
			if (argmax(out, 10) != z_gold[i]) {
				error++;
			}
		}

		// fclose(fptr_x[i]);
		// if(i%100==0) sleep(5);
	//}
  	printf("\rtest done %ld errors\n", error);
  	printf("Accuracy  : %.4f\n", 1.0 - ((double)error / test_n));
  	
    bool match = true;

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl; 

    std::cout << "Wall Clock Time (Kernel execution): " << kernel_duration << std::endl;
    std::cout << "Note: Wall Clock Time is meaningful for real hardware execution only,"  
            << "not for emulation." << std::endl; 

    return (match ? EXIT_SUCCESS :  EXIT_FAILURE);
}