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
            LastWeights = new double[MemorySettings.MemoryVectorLength];
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
            int memoryVectorLength = MemorySettings.MemoryVectorLength;
            double[] addressingVector = new double[memoryVectorLength];

            double sharpenAll = 0;

            for (int i = 0; i < memoryVectorLength; i++)
            {
                sharpenAll += Math.Pow(convolutedAddress[i], sharpening);
            }

            for (int i = 0; i < memoryVectorLength; i++)
            {
                addressingVector[i] = Math.Pow(convolutedAddress[i], sharpening) / sharpenAll;
            }

            return addressingVector;
        }

        private double[] DoConvolutialShift(double[] addressingVector, double[] shiftWeighting)
        {
            int memoryVectorLength = MemorySettings.MemoryVectorLength;
            int maxConvolutionalShift = MemorySettings.MaxConvolutionalShift;

            double[] convolutional = new double[memoryVectorLength];
            for (int i = 0; i < memoryVectorLength; i++)
            {
                for (int j = -maxConvolutionalShift; j <= maxConvolutionalShift; j++)
                {
                    if ((i + j) >= 0)
                    {
                        convolutional[i] += addressingVector[(i + j) % memoryVectorLength] * shiftWeighting[j + maxConvolutionalShift];
                    }
                    else
                    {
                        convolutional[i] += addressingVector[i + j + memoryVectorLength] * shiftWeighting[j + maxConvolutionalShift];
                    }
                }
            }

            return convolutional;
        }

        private void FocusByLocation(double[] addressingVector, double interpolationGate, double[] lastWeightVector)
        {
            int memoryVectorLength = MemorySettings.MemoryVectorLength;
            for (int i = 0; i < memoryVectorLength; i++)
            {
                addressingVector[i] = (interpolationGate * addressingVector[i]) + ( (1 - interpolationGate) * lastWeightVector[i]);
            }
        }

        private double[] GetContentAddressingVector(double[] keyVector, double keyStrengthBeta, NtmMemory memory)
        {
            int memoryVectorLength = MemorySettings.MemoryVectorLength;
            double[] addressingVector = new double[memoryVectorLength];

            double[] similarityVector = GetSimilarityVector(keyVector, keyStrengthBeta, memory);

            double similarityAll = 0;
            for (int i = 0; i < memoryVectorLength; i++)
            {
                similarityAll += similarityVector[i];
            }

            for (int i = 0; i < memoryVectorLength; i++)
            {
                addressingVector[i] = similarityVector[i] / similarityAll;
            }

            return addressingVector;
        }

        private double[] GetSimilarityVector(double[] keyVector, double keyStrengthBeta, NtmMemory memory)
        {
            int memoryVectorLength = MemorySettings.MemoryVectorLength;
            double[] similarityVector = new double[memoryVectorLength];
            for (int i = 0; i < memoryVectorLength; i++)
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
