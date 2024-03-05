Loop Dependency Inter (C)
=========================

This Example demonstrates the HLS pragma 'DEPENDENCE'.Using 'DEPENDENCE' pragma, user can provide additional dependency details to the compiler by specifying if the dependency in consecutive loop iterations on buffer is true/false, which allows the compiler to perform unrolling/pipelining to get better performance.

**KEY CONCEPTS:** `Inter Dependence <https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Managing-Pipeline-Dependencies>`__

**KEYWORDS:** `DEPENDENCE <https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/pragma-HLS-dependence>`__, `inter <https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/pragma-HLS-dependence>`__, `WAR <https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/pragma-HLS-dependence>`__

.. raw:: html

 <details>

.. raw:: html

 <summary> 

 <b>EXCLUDED PLATFORMS:</b>

.. raw:: html

 </summary>
|
..

 - Embedded ZC702
 - All NoDMA Platforms, i.e u50 nodma etc

.. raw:: html

 </details>

.. raw:: html

DESIGN FILES
------------

Application code is located in the src directory. Accelerator binary files will be compiled to the xclbin directory. The xclbin directory is required by the Makefile and its contents will be filled during compilation. A listing of all the files in this example is shown below

::

   src/host.cpp
   src/vconv.cpp
   src/vconv.h
   
COMMAND LINE ARGUMENTS
----------------------

Once the environment has been configured, the application can be executed by

::

   ./dependence_inter <vconv XCLBIN>

DETAILS
-------

This example demonstrates how extra information about dependency can be
specified for variables in consecutive loop iterations using
``#pragma HLS DEPENDENCE``. It allows the ``V++`` to recognize
dependencies and optimize the loop executions.

For example here ``linebuf`` has ``false inter dependency`` which means
each iteration of loop is independent for ``linebuf`` access.

``#pragma HLS DEPENDENCE`` allows compiler to ignore dependency of
linebuf and generate a pipeline with lower II count. If user does not
provide this pragma, by default compiler assumes dependency exists on
linebuf between iterations. As a result, compiler schedules unoptimized
design with larger II factor which in turn leads to slow performance.

.. code:: cpp

   #pragma HLS ARRAY_PARTITION variable = linebuf dim = 1 complete
   VConvH:
       for (int col = 0; col < height; ++col) {
       VConvW:
           for (int row = 0; row < width; ++row) {
              #pragma HLS DEPENDENCE variable=linebuf inter false
               int in_val = inStream.read();
               int out_val = 0;
           VConv:
               for (int i = 0; i < K; i++) {
                   int vwin_val = i < (K - 1) ? linebuf[i][row] : in_val;
                   out_val += vwin_val * vcoeff[i];

                   if (i > 0)
                       linebuf[i - 1][row] = vwin_val;
               }
               outStream << out_val;
           }
       }
   }

For more comprehensive documentation, `click here <http://xilinx.github.io/Vitis_Accel_Examples>`__.