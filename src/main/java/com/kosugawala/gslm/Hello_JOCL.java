package com.kosugawala.gslm;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

import ij.IJ;
import ij.plugin.PlugIn;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Random;

import static java.lang.System.*;
import static com.jogamp.opencl.CLMemory.Mem.*;
import static java.lang.Math.*;

/**
 * Hello Java OpenCL example. Adds all elements of buffer A to buffer B
 * and stores the result in buffer C.<br/>
 * Sample was inspired by the Nvidia VectorAdd example written in C/C++
 * which is bundled in the Nvidia OpenCL SDK.
 * @author Michael Bien
 */
public class Hello_JOCL implements PlugIn {

    private static void fillBuffer(FloatBuffer buffer, int seed) {
        Random rnd = new Random(seed);
        while(buffer.remaining() != 0)
            buffer.put(rnd.nextFloat()*100);
        buffer.rewind();
    }

    private static int roundUp(int groupSize, int globalSize) {
        int r = globalSize % groupSize;
        if (r == 0) {
            return globalSize;
        } else {
            return globalSize + groupSize - r;
        }
    }

	@Override
	public void run(String arg) {
		// set up (uses default CLPlatform and creates context for all devices)
        CLContext context = CLContext.create();
        IJ.log("created "+context);
        
        // always make sure to release the context under all circumstances
        // not needed for this particular sample but recommented
        try{
            
            // select fastest device
            CLDevice device = context.getMaxFlopsDevice();
            IJ.log("using "+device);

            // create command queue on device.
            CLCommandQueue queue = device.createCommandQueue();

            int elementCount = 1444477;                                  // Length of arrays to process
            int localWorkSize = min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
            int globalWorkSize = roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize

            // load sources, create and build program
            InputStream ist = Hello_JOCL.class.getResourceAsStream("/opencl/VectorAdd.cl");
            CLProgram program = context.createProgram(ist).build();

            // A, B are input buffers, C is for the result
            CLBuffer<FloatBuffer> clBufferA = context.createFloatBuffer(globalWorkSize, READ_ONLY);
            CLBuffer<FloatBuffer> clBufferB = context.createFloatBuffer(globalWorkSize, READ_ONLY);
            CLBuffer<FloatBuffer> clBufferC = context.createFloatBuffer(globalWorkSize, WRITE_ONLY);

            IJ.log("used device memory: "
                + (clBufferA.getCLSize()+clBufferB.getCLSize()+clBufferC.getCLSize())/1000000 +"MB");

            // fill input buffers with random numbers
            // (just to have test data; seed is fixed -> results will not change between runs).
            fillBuffer(clBufferA.getBuffer(), 12345);
            fillBuffer(clBufferB.getBuffer(), 67890);

            // get a reference to the kernel function with the name 'VectorAdd'
            // and map the buffers to its input parameters.
            CLKernel kernel = program.createCLKernel("VectorAdd");
            kernel.putArgs(clBufferA, clBufferB, clBufferC).putArg(elementCount);

            // asynchronous write of data to GPU device,
            // followed by blocking read to get the computed results back.
            long time = nanoTime();
            queue.putWriteBuffer(clBufferA, false)
                 .putWriteBuffer(clBufferB, false)
                 .put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize)
                 .putReadBuffer(clBufferC, true);
            time = nanoTime() - time;

            // print first few elements of the resulting buffer to the console.
            IJ.log("a+b=c results snapshot: ");
            for(int i = 0; i < 10; i++)
                IJ.log(clBufferC.getBuffer().get() + ", ");
            IJ.log("...; " + clBufferC.getBuffer().remaining() + " more");

            IJ.log("computation took: "+(time/1000000)+"ms");
            
        } catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}finally{
            // cleanup all resources associated with this context.
            context.release();
        }
	}

}
