namespace NeuralTuringMachine.Memory
{
    class NtmMemory
    {
        private readonly int _memoryCellCount;
        private readonly int _memoryVectorLength;
        private readonly double[][] _memory;

        public NtmMemory(int memoryCellCount, int memoryVectorLength)
        {
            _memoryCellCount = memoryCellCount;
            _memoryVectorLength = memoryVectorLength;
            _memory = new double[memoryCellCount][];
            for (int i = 0; i < _memoryCellCount; i++) 
            {
                _memory[i] = new double[_memoryVectorLength];
            }
        }

        private NtmMemory(int memoryCellCount, int memoryVectorLength, double[][] memory) : this(memoryCellCount, memoryVectorLength)
        {
            for (int i = 0; i < _memoryCellCount; i++)
            {
                for (int j = 0; j < _memoryVectorLength; j++)
                {
                    _memory[i][j] = memory[i][j];
                }
            }
        }

        //CONVEX COMBINATION
        public double[] Read(double[] weightVector)
        {
            double[] readVector = new double[_memoryVectorLength];

            for (int i = 0; i < _memoryCellCount; i++)
            {
                for (int j = 0; j < _memoryVectorLength; j++)
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
            for (int i = 0; i < _memoryCellCount; i++)
            {
                for (int j = 0; j < _memoryVectorLength; j++)
                {
                    _memory[i][j] = _memory[i][j] * (1 - (eraseVector[j] * weightVector[i]));
                }
            }
        }

        private void Add(double[] weightVector, double[] addVector)
        {
            for (int i = 0; i < _memoryCellCount; i++)
            {
                for (int j = 0; j < _memoryVectorLength; j++)
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
            return new NtmMemory(_memoryCellCount, _memoryVectorLength, _memory);
        }
    }
}
