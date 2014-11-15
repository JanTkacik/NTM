using System;
using AForge.Math.Metrics;

namespace NeuralTuringMachine.Memory
{
    abstract class Head
    {
        public abstract int OutputNeuronCount { get; }
        public abstract int OutputOffset { get; }

        protected int AddressingNeuronsCount;
        private readonly int _memoryLength;
        protected int MemoryCellSize;
        protected AddressingData ActualAddressingData;
        protected double[] LastWeights;
        private readonly int _id;
        private readonly int _maxConvolutialShift;

        private readonly ISimilarity _similarity;

        protected Head(int memoryLength, int memoryCellSize, int id, int maxConvolutialShift)
        {
            AddressingNeuronsCount = 4 + (maxConvolutialShift*2);
            _similarity = new CosineSimilarity();
            _memoryLength = memoryLength;
            MemoryCellSize = memoryCellSize;
            LastWeights = new double[memoryLength];
            _id = id;
            _maxConvolutialShift = maxConvolutialShift;
        }

        public int Id
        {
            get { return _id; }
        }

        protected double[] GetWeightVector(NTMMemory memory)
        {
            double[] contentAddressingVector = GetContentAddressingVector(ActualAddressingData.KeyVector, ActualAddressingData.KeyStrengthBeta, memory);
            FocusByLocation(contentAddressingVector, ActualAddressingData.InterpolationGate, LastWeights);
            double[] convolutedAddressingVector = DoConvolutialShift(contentAddressingVector, ActualAddressingData.ShiftWeighting);
            double[] sharpenedVector = SharpenVector(convolutedAddressingVector, ActualAddressingData.Sharpening);
            return sharpenedVector;
        }

        private double[] SharpenVector(double[] convolutedAddress, double sharpening)
        {
            double[] addressingVector = new double[_memoryLength];
            double sharpenI = 0;
            double sharpenOther = 0;

            for (int i = 0; i < _memoryLength; i++)
            {
                for (int j = 0; j < _memoryLength; j++)
                {
                    if (i != j)
                    {
                        sharpenOther += Math.Pow(convolutedAddress[j], sharpening);
                    }
                    else
                    {
                        sharpenI += Math.Pow(convolutedAddress[j], sharpening);
                    }
                }
                addressingVector[i] = sharpenI / sharpenOther;
            }

            return addressingVector;
        }

        private double[] DoConvolutialShift(double[] addressingVector, double[] shiftWeighting)
        {
            double[] convolutional = new double[_memoryLength];
            for (int i = 0; i < _memoryLength; i++)
            {
                for (int j = -_maxConvolutialShift; j <= _maxConvolutialShift; j++)
                {
                    convolutional[i] += addressingVector[(i+j)%_memoryLength] * shiftWeighting[j + _maxConvolutialShift];
                }
            }

            return convolutional;
        }

        private void FocusByLocation(double[] addressingVector, double interpolationGate, double[] lastWeightVector)
        {
            for (int i = 0; i < _memoryLength; i++)
            {
                addressingVector[i] = (interpolationGate * addressingVector[i]) + ( (1 - interpolationGate) * lastWeightVector[i]);
            }
        }

        private double[] GetContentAddressingVector(double[] keyVector, double keyStrengthBeta, NTMMemory memory)
        {
            double[] addressingVector = new double[_memoryLength];

            double[] similarityVector = GetSimilarityVector(keyVector, keyStrengthBeta, memory);

            double similarityI = 0;
            double similarityOther = 0;

            for (int i = 0; i < _memoryLength; i++)
            {
                for (int j = 0; j < _memoryLength; j++)
                {
                    if (i != j)
                    {
                        similarityOther += similarityVector[j];
                    }
                    else
                    {
                        similarityI = similarityVector[j];
                    }
                }
                addressingVector[i] = similarityI/similarityOther;
            }

            return addressingVector;
        }

        private double[] GetSimilarityVector(double[] keyVector, double keyStrengthBeta, NTMMemory memory)
        {
            double[] similarityVector = new double[_memoryLength];
            for (int i = 0; i < _memoryLength; i++)
            {
                similarityVector[i] = Math.Exp(keyStrengthBeta*_similarity.GetSimilarityScore(keyVector, memory.GetCellByIndex(i)));
            }
            return similarityVector;
        }

        public void UpdateAddressingData(double[] output)
        {
            ActualAddressingData = new AddressingData(output, OutputOffset, OutputNeuronCount, _maxConvolutialShift);
        }
    }
}
