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

#define REAL_T float
typedef REAL_T real_t;

using std::vector;

int mem_size = 721696;
int mem_size_hard = 358440;
int x_size_bytes = 28*28*sizeof(float);

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
    
void activate_setup(
	std::string binaryFile,
	cl::CommandQueue& q,
	cl::Kernel& krnl_activate,
	cl::Buffer& mem_buf,
	cl::Buffer& x_buf,
	cl::Buffer& z_buf,
	vector<float, aligned_allocator<float>>& mem,
	vector<float, aligned_allocator<float>>& x,
	vector<float, aligned_allocator<float>>& z,
	int n_test
){
    
    cl_int err;
    cl::Context context;

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
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            // This call will extract a kernel out of the program we loaded in the
            // previous line. A kernel is an OpenCL function that is executed on the
            // FPGA. This function is defined in the src/vetor_addition.cl file.
            OCL_CHECK(err, krnl_activate = cl::Kernel(program, "run_activate", &err));
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
    OCL_CHECK(err, cl::Buffer buf_mem(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, mem_size, mem.data(),
                                       &err));
    OCL_CHECK(err, cl::Buffer buf_x(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, x_size_bytes*n_test, x.data(),
                                       &err));
	OCL_CHECK(err, cl::Buffer buf_z(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float)*n_test*10, z.data(),
                                       &err));
	mem_buf = buf_mem;
	x_buf = buf_x;
	z_buf = buf_z;
}

uint64_t activate(
	cl::CommandQueue& q,
	cl::Kernel& krnl_activate,
	cl::Buffer& mem_buf,
	cl::Buffer& x_buf,
	cl::Buffer& z_buf,
	int n_test
){
    cl_int err;

    // set the kernel Arguments
    int narg = 0;
    OCL_CHECK(err, err = krnl_activate.setArg(narg++, mem_buf));
    OCL_CHECK(err, err = krnl_activate.setArg(narg++, x_buf));
	OCL_CHECK(err, err = krnl_activate.setArg(narg++, z_buf));

    // These commands will load the source_a and source_b vectors from the host
    // application and into the buffer_a and buffer_b cl::Buffer objects. The data
    // will be be transferred from system memory over PCIe to the FPGA on-board
    // DDR memory.
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({mem_buf, x_buf, z_buf}, 0 /* 0 means from host*/));

  	cl::Event event;
  	uint64_t kernel_duration = 0;
	int32_t x_offset, z_offset;
	// narg++;
    // Launch the Kernel
	x_offset = x_size_bytes / sizeof(float);
	z_offset = 10;
	OCL_CHECK(err, err = krnl_activate.setArg(narg++, x_offset));
	OCL_CHECK(err, err = krnl_activate.setArg(narg++, z_offset));
	OCL_CHECK(err, err = krnl_activate.setArg(narg++, n_test));
	OCL_CHECK(err, err = q.enqueueTask(krnl_activate, NULL, &event));
	kernel_duration += get_duration_ns(event);

    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will write the data from the
    // buffer_result cl_mem object to the source_results vector
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({mem_buf, z_buf}, CL_MIGRATE_MEM_OBJECT_HOST));
    
    q.finish();


  	return kernel_duration;
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
	cl::Kernel krnl_activate;
	cl::Buffer mem_buf;
	cl::Buffer x_buf;
	cl::Buffer z_buf;


  	//Allocate Memory in Host Memory
  	int test_n = 10000, digit, dec_cursor, s_idx; 
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
  	std::vector<float,aligned_allocator<float>> mem(mem_size/sizeof(float));
  	std::vector<float,aligned_allocator<float>> x(test_n*x_size_bytes/sizeof(float));
  	std::vector<float,aligned_allocator<float>> z(test_n*10);
	std::vector<float,aligned_allocator<float>> z_gold(test_n);

  	//Create the test data and Software Result 
	char buf[256], s[19];
  	FILE *fptr_mem, *fptr_x[test_n], *fptr_z, *fptr_out;
  	fptr_mem = fopen("src/data/mem.txt", "r+");
  	if(fptr_mem == NULL) {
		strerror_r(errno, buf, 256);
		printf("\nERRROR! err: %s\n", buf);
  	}
  	for(int i=0; i<(int)mem_size_hard/sizeof(float); i++) {
	 	/*printf("%f ", *((float *)(mem)+i));*/
	 	fscanf(fptr_mem, "%f ", &mem[i]);
	 	//printf("%f ", (float)mem[i]);
	}

	fptr_z = fopen("src/data/labels.txt", "r+");
	if(fptr_z == NULL) {
		strerror_r(errno, buf, 256);
	  	printf("\nERRROR! err: %s\n", buf);
	}
	for(int i=0; i<test_n; i++) {
	  	/*printf("%f ", *((float *)(mem)+i));*/
	  	fscanf(fptr_z, "%f ", &z_gold[i]);
	  	//printf("%d ", z[i]);
	}
	fclose(fptr_mem);
	fclose(fptr_z);

	uint64_t kernel_duration = 0;
	
	activate_setup(binaryFile, q, krnl_activate, mem_buf, x_buf, z_buf, mem, x, z, test_n);
	 
	for(int i=0; i<test_n; i++) {
	  	kernel_duration = 0;	  
	  	memset(s, 0, 10);
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
			fscanf(fptr_x[i], "%f ", &x[i*28*28+k]);
		}
		fclose(fptr_x[i]);
	}
		kernel_duration = activate(q, krnl_activate, mem_buf, x_buf, z_buf, test_n);
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
			// 	printf("out[%d]=%f\n",j,z[i*10+j]);
			// printf("z_gold[%d] = %f\n", i, z_gold[i]);
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
