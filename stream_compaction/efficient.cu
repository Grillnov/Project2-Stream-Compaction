#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		//No-longer const int* idata, this time we do it in-place
		__global__ void upSweeping(int n, int stride, int *idata)
		{
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx >= n)
			{
				return;
			}

			int stride2times = stride * 2;
			if (idx % stride2times == 0)
			{
				idata[idx + stride2times - 1] += idata[idx + stride - 1];
			}
		}
		__global__ void downSweeping(int n, int stride, int *idata)
		{
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx >= n)
			{
				return;
			}

			int stride2times = stride * 2;
			if (idx % stride2times == 0)
			{
				int indexCorresponding = idx + stride - 1;
				int indexCorresponding2times = idx + stride2times - 1;
				int delta = idata[indexCorresponding];
				idata[indexCorresponding] += idata[indexCorresponding2times];
				idata[indexCorresponding2times] += delta;
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int layer = ilog2ceil(n);

            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
