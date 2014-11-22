using System;
using AForge.Math.Metrics;
using NeuralTuringMachine.Misc;

namespace NeuralTuringMachine.Memory.Head
{
    public abstract class Head
    {
        protected readonly MemorySettings MemorySettings;
        public double[] LastWeights { get; protected set; }
        protected AddressingData ActualAddressingData;

        private static readonly ISimilarity Similarity = new EuclideanSimilarity();

        protected Head(MemorySettings memorySettings)
        {
            MemorySettings = memorySettings;
            LastWeights = new double[MemorySettings.MemoryCellCount];
        }

        public double[] GetWeightVector(NtmMemory memory)
        {
            double[] contentAddressingVector = GetContentAddressingVector(ActualAddressingData.KeyVector, ActualAddressingData.KeyStrengthBeta, memory);
            FocusByLocation(contentAddressingVector, ActualAddressingData.InterpolationGate, LastWeights);
            double[] convolutedAddressingVector = DoConvolutialShift(contentAddressingVector, ActualAddressingData.ShiftWeighting);
            double[] sharpenedVector = SharpenVector(convolutedAddressingVector, ActualAddressingData.Sharpening);
            ArrayHelper.NormalizeVector(sharpenedVector);
            
            return sharpenedVector;
        }

        private double[] SharpenVector(double[] convolutedAddress, double sharpening)
        {
            int memoryCellCount = MemorySettings.MemoryCellCount;
            double[] addressingVector = new double[memoryCellCount];

            double sharpenAll = 0;

            for (int i = 0; i < memoryCellCount; i++)
            {
                sharpenAll += Math.Pow(convolutedAddress[i], sharpening);
            }

            for (int i = 0; i < memoryCellCount; i++)
            {
                addressingVector[i] = Math.Pow(convolutedAddress[i], sharpening) / sharpenAll;
            }

            return addressingVector;
        }

        private double[] DoConvolutialShift(double[] addressingVector, double[] shiftWeighting)
        {
            int memoryCellCount = MemorySettings.MemoryCellCount;
            int maxConvolutionalShift = MemorySettings.MaxConvolutionalShift;

            double[] convolutional = new double[memoryCellCount];
            for (int i = 0; i < memoryCellCount; i++)
            {
                for (int j = -maxConvolutionalShift; j <= maxConvolutionalShift; j++)
                {
                    if ((i + j) >= 0)
                    {
                        convolutional[i] += addressingVector[(i + j) % memoryCellCount] * shiftWeighting[j + maxConvolutionalShift];
                    }
                    else
                    {
                        convolutional[i] += addressingVector[i + j + memoryCellCount] * shiftWeighting[j + maxConvolutionalShift];
                    }
                }
            }

            return convolutional;
        }

        private void FocusByLocation(double[] addressingVector, double interpolationGate, double[] lastWeightVector)
        {
            int memoryCellCount = MemorySettings.MemoryCellCount;
            for (int i = 0; i < memoryCellCount; i++)
            {
                addressingVector[i] = (interpolationGate * addressingVector[i]) + ( (1 - interpolationGate) * lastWeightVector[i]);
            }
        }

        private double[] GetContentAddressingVector(double[] keyVector, double keyStrengthBeta, NtmMemory memory)
        {
            int memoryCellCount = MemorySettings.MemoryCellCount;
            double[] addressingVector = new double[memoryCellCount];

            double[] similarityVector = GetSimilarityVector(keyVector, keyStrengthBeta, memory);

            double similarityAll = 0;
            for (int i = 0; i < memoryCellCount; i++)
            {
                similarityAll += similarityVector[i];
            }

            for (int i = 0; i < memoryCellCount; i++)
            {
                addressingVector[i] = similarityVector[i] / similarityAll;
            }

            return addressingVector;
        }

        private double[] GetSimilarityVector(double[] keyVector, double keyStrengthBeta, NtmMemory memory)
        {
            int memoryCellCount = MemorySettings.MemoryCellCount;
            double[] similarityVector = new double[memoryCellCount];
            for (int i = 0; i < memoryCellCount; i++)
            {
                similarityVector[i] = Math.Exp(keyStrengthBeta*Similarity.GetSimilarityScore(keyVector, memory.GetCellByIndex(i)));
            }
            return similarityVector;
        }

        public void UpdateAddressingData(double[] headOutput)
        {
            ActualAddressingData = new AddressingData(headOutput, MemorySettings);
        }
    }
}
