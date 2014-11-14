namespace NeuralTuringMachine
{
    class NTMMemory
    {
        private readonly int _memoryCellCount;
        private readonly int _memoryVectorLenght;
        private double[,] _memory;

        public NTMMemory(int memoryCellCount, int memoryVectorLenght)
        {
            _memoryCellCount = memoryCellCount;
            _memoryVectorLenght = memoryVectorLenght;
            _memory = new double[memoryCellCount, memoryVectorLenght];
        }
    }
}
