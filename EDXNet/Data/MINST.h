#pragma once

#include "../Core/NeuralNet.h"

#include "Windows/FileStream.h"

namespace EDX
{
	namespace DeepLearning
	{
		namespace DataSet
		{
			class MNIST
			{
			private:
				Tensorf mTrainData;
				Labels mTrainLabels;

				Tensorf mTestData;
				Labels mTestLabels;

			public:

				void Load(const String& directory)
				{
					mTrainLabels = ParseLabels(directory + "/train-labels.idx1-ubyte");
					mTrainData = ParseImages(directory + "/train-images.idx3-ubyte");

					mTestLabels = ParseLabels(directory + "/t10k-labels.idx1-ubyte");
					mTestData = ParseImages(directory + "/t10k-images.idx3-ubyte");
				}

				const Tensorf& GetTrainingData() const
				{
					return mTrainData;
				}
				Tensorf& GetTrainingData()
				{
					return mTrainData;
				}

				const Labels& GetTrainingLabels() const
				{
					return mTrainLabels;
				}
				Labels& GetTrainingLabels()
				{
					return mTrainLabels;
				}

				const Tensorf& GetTestData() const
				{
					return mTestData;
				}
				Tensorf& GetTestData()
				{
					return mTestData;
				}

				const Labels& GetTestLabels() const
				{
					return mTestLabels;
				}
				Labels& GetTestLabels()
				{
					return mTestLabels;
				}


			private:
				Labels ParseLabels(const String& path)
				{
					Labels ret;

					FileStream stream(path.GetCString(), FileMode::Open);

					int magicNum, numItems;

					stream >> magicNum;
					stream >> numItems;

					ReverseEndian(&magicNum);
					ReverseEndian(&numItems);

					Assertf(magicNum == 0x00000801, "MNIST label-file format error");

					Tensor<uint8> temp;
					temp.Resize((int)numItems);
					stream.ByteOrderRead(temp.Data(), numItems);

					ret.Resize((int)numItems);
					for (auto i = 0; i < numItems; i++)
					{
						ret[i] = temp[i];
					}

					return ret;
				}

				Tensorf ParseImages(const String& path)
				{
					Tensorf ret;

					FileStream stream(path.GetCString(), FileMode::Open);

					int magicNum, numItems;
					int width, height;

					stream >> magicNum;
					stream >> numItems;
					stream >> width;
					stream >> height;
					ReverseEndian(&magicNum);
					ReverseEndian(&numItems);
					ReverseEndian(&width);
					ReverseEndian(&height);

					Assertf(magicNum == 0x00000803, "MNIST label-file format error");

					Tensor<uint8> temp;
					temp.Resize((int)numItems, 1, height, width);
					stream.ByteOrderRead(temp.Data(), width * height * numItems);

					ret.Resize((int)numItems, 1, height, width);
					for (auto i = 0; i < temp.LinearSize(); i++)
					{
						ret[i] = temp[i] / float(256.0f);
					}

					return ret;
				}

				template <typename T>
				T* ReverseEndian(T* value)
				{
					Algorithm::Reverse(reinterpret_cast<char*>(value), sizeof(T));

					return value;
				}


				static StaticArray<TCHAR*, 10> GetLabelString()
				{
					return
					{
						"0",
						"1",
						"2",
						"3",
						"4",
						"5",
						"6",
						"7",
						"8",
						"9",
					};
				}
			};
		}
	}
}