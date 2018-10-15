
#include "Core/Tensor.h"

using namespace EDX;
using namespace DeepLearning;
using namespace Algorithm;

void TestCUDA()
{
	{
		Tensorf A = { 1,2,3,4,5,8,3,1,4 };
		Tensorf B = { 1,2,3,4,5 };

		B.Reshape(5, 1);

		Tensorf C = A + B + A + A * B;

		float pHostC[45] = { 0 };
		cudaMemcpy((void*)pHostC, (void*)C.Data(), C.LinearSize() * sizeof(float), cudaMemcpyDeviceToHost);
	}

	{
		Tensorf A = { 9,2,3,4,5 };
		Tensorf B = { 9,2,3,4,5 };

		A *= Tensorf::Exp(B);

		float pHostA[5] = { 0 };
		cudaMemcpy((void*)pHostA, (void*)A.Data(), A.LinearSize() * sizeof(float), cudaMemcpyDeviceToHost);
	}

	{
		Tensorf A = Tensorf::LinSpace(0, 40960, 40960);
		Tensorf sum = Tensorf::StandardDeviation(A);

		float pHost[1] = { 0 };
		cudaMemcpy((void*)pHost, (void*)sum.Data(), sum.LinearSize() * sizeof(float), cudaMemcpyDeviceToHost);
	}
}