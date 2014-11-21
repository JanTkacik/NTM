using System;

namespace NeuralTuringMachine.Memory
{
    public class NtmMemory
    {
        public int CellCount { get; private set; }
        public int MemoryVectorLength { get; private set; }
        private readonly double[][] _memory;

        public NtmMemory(int memoryCellCount, int memoryVectorLength)
        {
            CellCount = memoryCellCount;
            MemoryVectorLength = memoryVectorLength;
            _memory = new double[memoryCellCount][];
            for (int i = 0; i < CellCount; i++) 
            {
                _memory[i] = new double[MemoryVectorLength];
            }
        }

        private NtmMemory(int memoryCellCount, int memoryVectorLength, double[][] memory) : this(memoryCellCount, memoryVectorLength)
        {
            for (int i = 0; i < CellCount; i++)
            {
                for (int j = 0; j < MemoryVectorLength; j++)
                {
                    _memory[i][j] = memory[i][j];
                }
            }
        }

        //CONVEX COMBINATION
        public double[] Read(double[] weightVector)
        {
            double[] readVector = new double[MemoryVectorLength];

            for (int i = 0; i < CellCount; i++)
            {
                for (int j = 0; j < MemoryVectorLength; j++)
                {
                    readVector[j] = weightVector[i] * _memory[i][j];
                }
            }

            return readVector;
        }

        public void Write(double[] weightVector, double[] eraseVector, double[] addVector)
        {
            Erase(weightVector, eraseVector);
            Add(weightVector, addVector);
        }

        private void Erase(double[] weightVector, double[] eraseVector)
        {
            for (int i = 0; i < CellCount; i++)
            {
                for (int j = 0; j < MemoryVectorLength; j++)
                {
                    _memory[i][j] = _memory[i][j] * (1 - (eraseVector[j] * weightVector[i]));
                }
            }
        }

        private void Add(double[] weightVector, double[] addVector)
        {
            for (int i = 0; i < CellCount; i++)
            {
                for (int j = 0; j < MemoryVectorLength; j++)
                {
                    _memory[i][j] = _memory[i][j] + (weightVector[i] * addVector[j]);
                }
            }
        }

        public double[] GetCellByIndex(int i)
        {
            return _memory[i];
        }

        public NtmMemory Clone()
        {
            return new NtmMemory(CellCount, MemoryVectorLength, _memory);
        }

        public void SetMemoryContent(double[] memoryContent)
        {
            int offset = 0;
            for (int i = 0; i < CellCount; i++)
            {
                Array.Copy(memoryContent, offset, _memory[i], 0, MemoryVectorLength);
                offset += MemoryVectorLength;
            }
        }

        public double[] GetDataAfterWrite(double[] weightVector, double[] eraseVector, double[] addVector)
        {
            double[] returnData = new double[MemoryVectorLength * CellCount];
            
            //ERASE
            for (int i = 0; i < CellCount; i++)
            {
                for (int j = 0; j < MemoryVectorLength; j++)
                {
                    returnData[(i * MemoryVectorLength) + j] = _memory[i][j] * (1 - (eraseVector[j] * weightVector[i]));
                }
            }

            //ADD
            for (int i = 0; i < CellCount; i++)
            {
                for (int j = 0; j < MemoryVectorLength; j++)
                {
                    returnData[(i * MemoryVectorLength) + j] = returnData[(i * MemoryVectorLength) + j] + (weightVector[i] * addVector[j]);
                }
            }

            return returnData;
        }
    }
}
