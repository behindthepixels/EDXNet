#pragma once

#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{
		class ContentLoss : public SymbolBase<2, 2>
		{
		public:
			ContentLoss(Symbol* pImage, Symbol* pContent)
			{
				mInputs[0] = pImage;
				mInputs[1] = pContent;
			}

			void Evaluate() override
			{
				const Tensorf image = mInputs[0]->GetOutput().GetSlice(0);
				const Tensorf content = mInputs[1]->GetOutput().GetSlice(0);

				Tensorf contentDiff = image - content;
				Tensorf contentDiffSqr = contentDiff * contentDiff;

				Tensorf& loss = GetOutput();
				loss = Scalar(0.5f) * Tensorf::Sum(contentDiffSqr);

				GetOutput(1) = contentDiff;
			}

			Symbol* Gradient(Symbol* pUpperGrads) const override;
		};

		class ContentLossGradient : public SymbolBase<4, 1>
		{
		public:
			ContentLossGradient(Symbol* pContentLoss, Symbol* pImage, Symbol* pContent, Symbol* pUpperGradients)
			{
				mInputs[0] = pContentLoss;
				mInputs[1] = pImage;
				mInputs[2] = pContent;
				mInputs[3] = pUpperGradients;
			}

			void Evaluate() override
			{
				const Symbol* pContentLoss = mInputs[0];
				const Tensorf image = mInputs[1]->GetOutput();
				const Tensorf content = mInputs[2]->GetOutput();
				const Tensorf& upperGrads = mInputs[3]->GetOutput(pContentLoss->GetGradientIndex());

				// TODO: Avoid recomputing this term
				Tensorf& output = GetOutput();
				output = pContentLoss->GetOutput(1);

				// Multiply with upper stream gradients
				output = upperGrads * output;

				//for (int i = 0; i < image.LinearSize(); i++)
				//{
				//	if (image[i] < 0.0f)
				//		output[i] = 0.0f;
				//}

				mInputs[1]->SetGradientIndex(0);
			}
		};


		class StyleLoss : public SymbolBase<2, 2>
		{
		public:
			StyleLoss(Symbol* pImage, Symbol* pStyle)
			{
				mInputs[0] = pImage;
				mInputs[1] = pStyle;
			}

			void Evaluate() override
			{
				const Tensorf image = mInputs[0]->GetOutput().GetSlice(0);
				const Tensorf style = mInputs[1]->GetOutput().GetSlice(0);

				Tensorf gramMatrixImage = GramMatrix(image);
				Tensorf gramMatrixStyle = GramMatrix(style);

				Tensorf gramMatrixDiff = gramMatrixImage - gramMatrixStyle;
				Tensorf gramMatrixDiffSqr = gramMatrixDiff * gramMatrixDiff;


				const int featureSize = image.LinearSize();
				Tensorf& loss = GetOutput();
				loss = Tensorf::Sum(gramMatrixDiffSqr) / Scalar(4.0f * featureSize * featureSize);

				GetOutput(1) = gramMatrixDiff;
			}

			Symbol* Gradient(Symbol* pUpperGrads) const override;

		private:
			Tensorf GramMatrix(const Tensorf& features)
			{
				const int N = features.Shape(0);
				const int dim = features.LinearSize() / N;

				Tensorf featureAlias = features.GetWithShape(N, dim);

				return Tensorf::Dot(featureAlias, featureAlias.GetTransposed());
			}
		};

		class StyleLossGradient : public SymbolBase<4, 1>
		{
		public:
			StyleLossGradient(Symbol* pStyleLoss, Symbol* pImage, Symbol* pStyle, Symbol* pUpperGradients)
			{
				mInputs[0] = pStyleLoss;
				mInputs[1] = pImage;
				mInputs[2] = pStyle;
				mInputs[3] = pUpperGradients;
			}

			void Evaluate() override
			{
				const Symbol* pStyleLoss = mInputs[0];
				const Tensorf image = mInputs[1]->GetOutput().GetSlice(0);
				const Tensorf style = mInputs[2]->GetOutput().GetSlice(0);
				const Tensorf& upperGrads = mInputs[3]->GetOutput(pStyleLoss->GetGradientIndex());

				Tensorf gramMatrixImage = GramMatrix(image);
				Tensorf gramMatrixStyle = GramMatrix(style);

				Tensorf gramMatrixDiff = pStyleLoss->GetOutput(1);

				const int N = image.Shape(0);
				const int featureSize = image.LinearSize();
				const int dim = featureSize / N;
				const Tensorf imageAlias = image.GetWithShape(N, dim);

				Tensorf& output = GetOutput();
				output = Tensorf::Dot(gramMatrixDiff, imageAlias) / Scalar(1.0f * featureSize * featureSize);

				// Multiply with upper stream gradients
				output = upperGrads * output;
				output.Reshape(mInputs[1]->GetOutput().Shape());

				//for (int i = 0; i < image.LinearSize(); i++)
				//{
				//	if (image[i] < 0.0f)
				//		output[i] = 0.0f;
				//}

				mInputs[1]->SetGradientIndex(0);
			}

		private:
			Tensorf GramMatrix(const Tensorf& features)
			{
				const int N = features.Shape(0);
				const int dim = features.LinearSize() / N;

				Tensorf featureAlias = features.GetWithShape(N, dim);

				return Tensorf::Dot(featureAlias, featureAlias.GetTransposed());
			}
		};

		class TotalVariationLoss : public SymbolBase<1, 4>
		{
		public:
			TotalVariationLoss(Symbol* pImage)
			{
				mInputs[0] = pImage;
			}

			void Evaluate() override
			{
				const Tensorf image = mInputs[0]->GetOutput().GetSlice(0);

				const int width = image.Shape(2);
				const int height = image.Shape(1);
				const int channel = image.Shape(0);

				Tensorf& gradNormX = GetOutput(1);
				Tensorf& gradNormY = GetOutput(2);
				Tensorf& gradientLength = GetOutput(3);
				gradNormX.Resize(channel, height, width);
				gradNormY.Resize(channel, height, width);
				gradientLength.Resize(channel, height, width);

				for (int c = 0; c < channel; c++)
				{
					for (int y = 0; y < height; y++)
					{
						for (int x = 0; x < width; x++)
						{
							int x1 = Math::Min(x + 1, width - 1);
							int y1 = Math::Min(y + 1, height - 1);

							float center = image(c, y, x);
							float gradX = image(c, y, x1) - center;
							float gradY = image(c, y1, x) - center;
							float gradLen = 1e-4f/* epsilon */ + Math::Sqrt(gradX * gradX + gradY * gradY);


							gradNormX(c, y, x) = gradX / gradLen;
							gradNormY(c, y, x) = gradY / gradLen;
							gradientLength(c, y, x) = gradLen;
						}
					}
				}

				Tensorf& loss = GetOutput(0);
				loss = Tensorf::Sum(gradientLength);
			}

			Symbol* Gradient(Symbol* pUpperGrads) const override;
		};

		class TotalVariationGradient : public SymbolBase<3, 1>
		{
		public:
			TotalVariationGradient(Symbol* pLoss, Symbol* pImage, Symbol* pUpperGrads)
			{
				mInputs[0] = pLoss;
				mInputs[1] = pImage;
				mInputs[2] = pUpperGrads;
			}

			void Evaluate() override
			{
				const Tensorf& gradNormX = mInputs[0]->GetOutput(1);
				const Tensorf& gradNormY = mInputs[0]->GetOutput(2);
				const Tensorf& image = mInputs[1]->GetOutput(0);
				const Tensorf& upperGrads = mInputs[2]->GetOutput(0);

				const int width = gradNormX.Shape(2);
				const int height = gradNormX.Shape(1);
				const int channel = gradNormX.Shape(0);

				Tensorf& divGrad = GetOutput(0);
				divGrad.Resize(channel, height, width);

				for (int c = 0; c < channel; c++)
				{
					for (int y = 0; y < height; y++)
					{
						for (int x = 0; x < width; x++)
						{
							int x1 = Math::Max(x - 1, 0);
							int y1 = Math::Max(y - 1, 0);

							float divX = gradNormX(c, y, x) - gradNormX(c, y, x1);
							float divY = gradNormY(c, y, x) - gradNormY(c, y1, x);

							divGrad(c, y, x) = -(divX + divY);
						}
					}
				}

				divGrad *= upperGrads;
				divGrad.Reshape(image.Shape());
				mInputs[1]->SetGradientIndex(0);
			}
		};

		class MattingLaplacian
		{
		public:
			static SparseMatrixf CalcLaplacianMatrix(const Tensorf& image, const int r = 1)
			{
				Assertf(image.Dim() == 3, "Dimension mismatch for matting laplacian.");

				const int numChannel = image.Shape(0);
				const int imgWidth = image.Shape(2);
				const int imgHeight = image.Shape(1);
				const int N = (imgWidth - 2 * r) * (imgHeight - 2 * r);
				const int windowDim = (2 * r + 1);
				const int windowSize = windowDim * windowDim;
				const float eps = 1e-7f;
				const int numPixels = image.LinearSize() / numChannel;

				Tensorf meanVec;
				meanVec.Resize(1, numChannel);

				Tensorf covMat;
				covMat.Resize(numChannel, numChannel);

				Tensorf winI;
				winI.Resize(1, numChannel);

				DynamicSparseMatrixf sparseMat;
				sparseMat.Resize(numPixels);

				TensorIndex imgIndices;
				imgIndices.Resize(imgHeight, imgWidth);

				for (int n = 0; n < numPixels; n++)
				{
					StaticArray<int, 4> index = image.StaticIndex(n);
					if (index[1] < r || index[1] >= imgHeight - r ||
						index[2] < r || index[2] >= imgWidth - r)
					{
						continue;
					}

					// Calculate per channel mean over the patch
					Tensorf imageWindow;
					imageWindow.Resize(windowSize, numChannel);
					for (int c = 0; c < numChannel; c++)
					{
						int tmpIdx = 0;
						for (int i = -r; i <= r; i++)
						{
							for (int j = -r; j <= r; j++)
							{
								const int x = index[2] + j;
								const int y = index[1] + i;

								imageWindow(tmpIdx++, c) = image(c, y, x);
							}
						}
					}

					Tensorf mean = Tensorf::Mean(imageWindow, { 0 });
					mean.Reshape(1, numChannel);

					// Calculate covariance matrix over the patch
					Tensorf covMat = Tensorf::Dot(imageWindow.GetTransposed(), imageWindow) / Scalar(float(windowSize))
						- Tensorf::Dot(mean.GetTransposed(), mean)
						+ Tensorf::Identity(numChannel) * Scalar(eps / float(windowSize));

					Tensorf invCovMat = Tensorf::Inverse(covMat);

					imageWindow -= mean;
					Tensorf val = (Scalar(1.0f) + Tensorf::Dot(Tensorf::Dot(imageWindow, invCovMat), imageWindow.GetTransposed())) / Scalar(windowSize);

					Array<int> linearIndices;
					linearIndices.Reserve(windowSize);
					for (int i = -r; i <= r; i++)
					{
						for (int j = -r; j <= r; j++)
						{
							const int x = index[2] + j;
							const int y = index[1] + i;

							linearIndices.Add(imgIndices.LinearIndex(y, x));
						}
					}

					for (int i = 0; i < linearIndices.Size(); i++)
					{
						for (int j = 0; j < linearIndices.Size(); j++)
						{
							const int x = linearIndices[j];
							const int y = linearIndices[i];
							float incrementValue = (x == y ? 1.0f : 0.0f) - val(i, j);
							sparseMat.AddToElement(linearIndices[i], linearIndices[j], incrementValue);
						}
					}
				}

				SparseMatrixf ret;
				ret.ConstructFromMatrix(sparseMat);

				return ret;
			}
		};

		class PhotorealismLoss : public SymbolBase<1, 1>
		{
		public:
			PhotorealismLoss(Symbol* pImage, SparseMatrixf& laplacianMatrix)
				: mLaplacianMatrix(laplacianMatrix)
			{
				mInputs[0] = pImage;
			}

			void Evaluate() override
			{
				Tensorf image = mInputs[0]->GetOutput().GetSlice(0);

				const int width = image.Shape(2);
				const int height = image.Shape(1);
				const int numChannel = image.Shape(0);
				const int numPixels = width * height;

				image.Reshape(numChannel, numPixels);

				Tensorf& output = GetOutput(0);
				output = Tensorf::Sum(Tensorf::Dot(image, Tensorf::Dot(mLaplacianMatrix, Tensorf::Transpose(image))));
			}

			Symbol* Gradient(Symbol* pUpperGrads) const override;

		private:
			SparseMatrixf& mLaplacianMatrix;
		};

		class PhotorealismGradient : public SymbolBase<2, 1>
		{
		public:
			PhotorealismGradient(Symbol* pImage, Symbol* pUpperGrads, SparseMatrixf& laplacianMatrix)
				: mLaplacianMatrix(laplacianMatrix)
			{
				mInputs[0] = pImage;
				mInputs[1] = pUpperGrads;
			}

			void Evaluate() override
			{
				Tensorf image = mInputs[0]->GetOutput().GetSlice(0);
				const Tensorf& upperGrads = mInputs[1]->GetOutput(0);

				const int width = image.Shape(2);
				const int height = image.Shape(1);
				const int numChannel = image.Shape(0);
				const int numPixels = width * height;

				image.Reshape(numChannel, numPixels);

				Tensorf& gradient = GetOutput(0);
				gradient = Scalar(2.0f) * Tensorf::Dot(mLaplacianMatrix, Tensorf::Transpose(image)) * upperGrads;
			}

		private:
			SparseMatrixf& mLaplacianMatrix;
		};

		class StyleTransferLoss : public SymbolBase<3, 1>
		{
		public:
			StyleTransferLoss(Symbol* pContentLoss, Symbol* pStyleLoss, Symbol* pTVLoss, const float contentWeight, const float styleWeight, const float tvWeight)
			{
				mInputs[0] = pContentLoss;
				mInputs[1] = pStyleLoss;
				mInputs[2] = pTVLoss;

				mContentWeight = contentWeight;
				mStyleWeight = styleWeight;
				mTotalVariartionWeight = tvWeight;
			}

			void Evaluate() override
			{
				const Tensorf& contentLoss = mInputs[0]->GetOutput();
				const Tensorf& styleLoss = mInputs[1]->GetOutput();

				Tensorf& output = GetOutput();
				output = Scalar(mContentWeight) * contentLoss + Scalar(mStyleWeight) * styleLoss;

				if (mInputs[2])
				{
					const Tensorf& totalVariationLoss = mInputs[2]->GetOutput();
					output += Scalar(mTotalVariartionWeight) * totalVariationLoss;
				}
			}

			Symbol* Gradient(Symbol* pUpperGrads) const override;

		private:
			float mContentWeight;
			float mStyleWeight;
			float mTotalVariartionWeight;
		};

		class StyleTransferLossGradient : public SymbolBase<4, 3>
		{
		public:
			StyleTransferLossGradient(Symbol* pContentLoss, Symbol* pStyleLoss, Symbol* pTVLoss, Symbol* pUpperGradients, const float contentWeight, const float styleWeight, const float tvWeight)
			{
				mInputs[0] = pContentLoss;
				mInputs[1] = pStyleLoss;
				mInputs[2] = pTVLoss;
				mInputs[3] = pUpperGradients;

				mContentWeight = contentWeight;
				mStyleWeight = styleWeight;
				mTotalVariartionWeight = tvWeight;
			}

			void Evaluate() override
			{
				const Tensorf& upperGrads = mInputs[3]->GetOutput();

				Tensorf& contentGrads = GetOutput(0);
				contentGrads = Scalar(mContentWeight) * upperGrads;

				Tensorf& styleGrads = GetOutput(1);
				styleGrads = Scalar(mStyleWeight) * upperGrads;

				mInputs[0]->SetGradientIndex(0);
				mInputs[1]->SetGradientIndex(1);

				if (mInputs[2])
				{
					Tensorf& tvGrads = GetOutput(2);
					tvGrads = Scalar(mTotalVariartionWeight) * upperGrads;

					mInputs[2]->SetGradientIndex(2);
				}
			}

		private:
			float mContentWeight;
			float mStyleWeight;
			float mTotalVariartionWeight;
		};

	}
}