using System;
using AForge.Math.Metrics;

namespace NeuralTuringMachine.Memory.Head
{
    public abstract class Head
    {
        public abstract int OutputNeuronCount { get; }
        public double[] LastWeights { get; protected set; }

        protected int AddressingNeuronsCount;
        protected readonly int MemoryLength;
        protected int MemoryCellSize;
        protected AddressingData ActualAddressingData;
        protected readonly int MaxConvolutialShift;

        private readonly ISimilarity _similarity;

        protected Head(int memoryLength, int memoryCellSize, int maxConvolutialShift)
        {
            AddressingNeuronsCount = 4 + (maxConvolutialShift*2);
            _similarity = new CosineSimilarity();
            MemoryLength = memoryLength;
            MemoryCellSize = memoryCellSize;
            LastWeights = new double[memoryLength];
            MaxConvolutialShift = maxConvolutialShift;
        }

        public double[] GetWeightVector(NtmMemory memory)
        {
            double[] contentAddressingVector = GetContentAddressingVector(ActualAddressingData.KeyVector, ActualAddressingData.KeyStrengthBeta, memory);
            FocusByLocation(contentAddressingVector, ActualAddressingData.InterpolationGate, LastWeights);
            double[] convolutedAddressingVector = DoConvolutialShift(contentAddressingVector, ActualAddressingData.ShiftWeighting);
            double[] sharpenedVector = SharpenVector(convolutedAddressingVector, ActualAddressingData.Sharpening);
            return sharpenedVector;
        }

        private double[] SharpenVector(double[] convolutedAddress, double sharpening)
        {
            double[] addressingVector = new double[MemoryLength];

            double sharpenAll = 0;

            for (int i = 0; i < MemoryLength; i++)
            {
                sharpenAll = Math.Pow(convolutedAddress[i], sharpening);
            }

            for (int i = 0; i < MemoryLength; i++)
            {
                addressingVector[i] = Math.Pow(convolutedAddress[i], sharpening) / sharpenAll;
            }

            return addressingVector;
        }

        private double[] DoConvolutialShift(double[] addressingVector, double[] shiftWeighting)
        {
            double[] convolutional = new double[MemoryLength];
            for (int i = 0; i < MemoryLength; i++)
            {
                for (int j = -MaxConvolutialShift; j <= MaxConvolutialShift; j++)
                {
                    if ((i + j) >= 0)
                    {
                        convolutional[i] += addressingVector[(i + j) % MemoryLength] * shiftWeighting[j + MaxConvolutialShift];
                    }
                    else
                    {
                        convolutional[i] += addressingVector[i + j + MemoryLength] * shiftWeighting[j + MaxConvolutialShift];
                    }
                }
            }

            return convolutional;
        }

        private void FocusByLocation(double[] addressingVector, double interpolationGate, double[] lastWeightVector)
        {
            for (int i = 0; i < MemoryLength; i++)
            {
                addressingVector[i] = (interpolationGate * addressingVector[i]) + ( (1 - interpolationGate) * lastWeightVector[i]);
            }
        }

        private double[] GetContentAddressingVector(double[] keyVector, double keyStrengthBeta, NtmMemory memory)
        {
            double[] addressingVector = new double[MemoryLength];

            double[] similarityVector = GetSimilarityVector(keyVector, keyStrengthBeta, memory);

            double similarityAll = 0;
            for (int i = 0; i < MemoryLength; i++)
            {
                similarityAll += similarityVector[i];
            }

            for (int i = 0; i < MemoryLength; i++)
            {
                addressingVector[i] = similarityVector[i] / similarityAll;
            }

            return addressingVector;
        }

        private double[] GetSimilarityVector(double[] keyVector, double keyStrengthBeta, NtmMemory memory)
        {
            double[] similarityVector = new double[MemoryLength];
            for (int i = 0; i < MemoryLength; i++)
            {
                similarityVector[i] = Math.Exp(keyStrengthBeta*_similarity.GetSimilarityScore(keyVector, memory.GetCellByIndex(i)));
            }
            return similarityVector;
        }

        public void UpdateAddressingData(double[] headOutput)
        {
            ActualAddressingData = new AddressingData(headOutput, MemoryCellSize, MaxConvolutialShift);
        }
    }
}
