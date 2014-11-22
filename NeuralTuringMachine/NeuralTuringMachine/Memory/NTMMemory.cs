using System;
using System.Text;
using NeuralTuringMachine.Memory.Head;
using NeuralTuringMachine.Misc;

namespace NeuralTuringMachine.Memory
{
    public class NtmMemory
    {
        public MemorySettings MemorySettings { get; private set; }
        private readonly double[][] _memory;
        private static readonly Random Rand = new Random();

        public NtmMemory(MemorySettings settings)
        {
            MemorySettings = settings;
            int memoryCellCount = MemorySettings.MemoryCellCount;
            int memoryVectorLength = MemorySettings.MemoryVectorLength;

            _memory = new double[memoryCellCount][];
            for (int i = 0; i < memoryCellCount; i++) 
            {
                _memory[i] = new double[memoryVectorLength];
            }
        }

        private NtmMemory(MemorySettings settings, double[][] memory)
        {
            MemorySettings = settings;
            _memory = memory;
        }

        //CONVEX COMBINATION
        public double[] Read(double[] weightVector)
        {
            int memoryCellCount = MemorySettings.MemoryCellCount;
            int memoryVectorLength = MemorySettings.MemoryVectorLength;
            double[] readVector = new double[memoryVectorLength];

            for (int i = 0; i < memoryCellCount; i++)
            {
                for (int j = 0; j < memoryVectorLength; j++)
                {
                    readVector[j] += weightVector[i] * _memory[i][j];
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
            int memoryCellCount = MemorySettings.MemoryCellCount;
            int memoryVectorLength = MemorySettings.MemoryVectorLength;
            for (int i = 0; i < memoryCellCount; i++)
            {
                for (int j = 0; j < memoryVectorLength; j++)
                {
                    _memory[i][j] = _memory[i][j] * (1 - (eraseVector[j] * weightVector[i]));
                }
            }
        }

        private void Add(double[] weightVector, double[] addVector)
        {
            int memoryCellCount = MemorySettings.MemoryCellCount;
            int memoryVectorLength = MemorySettings.MemoryVectorLength;
            for (int i = 0; i < memoryCellCount; i++)
            {
                for (int j = 0; j < memoryVectorLength; j++)
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
            return new NtmMemory(MemorySettings, ArrayHelper.CloneArray(_memory));
        }

        public void SetMemoryContent(double[] memoryContent)
        {
            int memoryCellCount = MemorySettings.MemoryCellCount;
            int memoryVectorLength = MemorySettings.MemoryVectorLength;
            int offset = 0;
            for (int i = 0; i < memoryCellCount; i++)
            {
                Array.Copy(memoryContent, offset, _memory[i], 0, memoryVectorLength);
                offset += memoryVectorLength;
            }
        }

        public double[] GetDataAfterWrite(double[] weightVector, double[] eraseVector, double[] addVector)
        {
            int memoryCellCount = MemorySettings.MemoryCellCount;
            int memoryVectorLength = MemorySettings.MemoryVectorLength;
            double[] returnData = new double[memoryVectorLength * memoryCellCount];
            
            //ERASE
            for (int i = 0; i < memoryCellCount; i++)
            {
                for (int j = 0; j < memoryVectorLength; j++)
                {
                    returnData[(i * memoryVectorLength) + j] = _memory[i][j] * (1 - (eraseVector[j] * weightVector[i]));
                }
            }

            //ADD
            for (int i = 0; i < memoryCellCount; i++)
            {
                for (int j = 0; j < memoryVectorLength; j++)
                {
                    returnData[(i * memoryVectorLength) + j] = returnData[(i * memoryVectorLength) + j] + (weightVector[i] * addVector[j]);
                }
            }

            return returnData;
        }

        public void ResetMemory()
        {
            int memoryCellCount = MemorySettings.MemoryCellCount;
            int memoryVectorLength = MemorySettings.MemoryVectorLength;
            for (int i = 0; i < memoryCellCount; i++)
            {
                for (int j = 0; j < memoryVectorLength; j++)
                {
                    _memory[i][j] = 0;
                }
            }
        }

        public void Randomize()
        {
            int memoryCellCount = MemorySettings.MemoryCellCount;
            int memoryVectorLength = MemorySettings.MemoryVectorLength;
            for (int i = 0; i < memoryCellCount; i++)
            {
                for (int j = 0; j < memoryVectorLength; j++)
                {
                    _memory[i][j] = Rand.NextDouble();
                }
            }
        }

        public override string ToString()
        {
            int memoryCellCount = MemorySettings.MemoryCellCount;
            int memoryVectorLength = MemorySettings.MemoryVectorLength;
            StringBuilder builder = new StringBuilder();
            AppendTableRowSeparator(builder);
            for (int i = 0; i < memoryCellCount; i++)
            {
                builder.Append("|");
                builder.AppendFormat("{0:00}",i);
                builder.Append("|");
                for (int j = 0; j < memoryVectorLength; j++)
                {
                    builder.AppendFormat("{0:0.000}", _memory[i][j]);
                    builder.Append("|");
                }
                builder.Append(Environment.NewLine);
                AppendTableRowSeparator(builder);
            }

            return builder.ToString();
        }

        private void AppendTableRowSeparator(StringBuilder builder)
        {
            int memoryVectorLength = MemorySettings.MemoryVectorLength;
            builder.Append("+--+");
            for (int i = 0; i < memoryVectorLength; i++)
            {
                builder.Append("-----+");
            }
            builder.Append(Environment.NewLine);
        }
    }
}
