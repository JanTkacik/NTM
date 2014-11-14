namespace NeuralTuringMachine.Memory
{
    class NTMMemory
    {
        private readonly int _memoryCellCount;
        private readonly int _memoryVectorLength;
        private readonly double[,] _memory;

        public NTMMemory(int memoryCellCount, int memoryVectorLength)
        {
            _memoryCellCount = memoryCellCount;
            _memoryVectorLength = memoryVectorLength;
            _memory = new double[memoryCellCount, memoryVectorLength];
        }

        //CONVEX COMBINATION
        public double[] Read(double[] weightVector)
        {
            double[] readVector = new double[_memoryVectorLength];

            for (int i = 0; i < _memoryCellCount; i++)
            {
                for (int j = 0; j < _memoryVectorLength; j++)
                {
                    readVector[j] = weightVector[i] * _memory[i,j];
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
                    _memory[i, j] = _memory[i, j] * (1 - (eraseVector[j] * weightVector[i]));
                }
            }
        }

        private void Add(double[] weightVector, double[] addVector)
        {
            for (int i = 0; i < _memoryCellCount; i++)
            {
                for (int j = 0; j < _memoryVectorLength; j++)
                {
                    _memory[i, j] = _memory[i, j] + (weightVector[i] * addVector[j]);
                }
            }
        }
    }
}
