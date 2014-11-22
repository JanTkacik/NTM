using NUnit.Framework;
using NeuralTuringMachine.Memory;
using NeuralTuringMachine.Memory.Head;

namespace NeuralTuringMachine.Tests
{
    [TestFixture]
    public class MemoryAndAddressingTests
    {
        [Test]
        public void ItIsPossibleToFindHeadThatProducesNearZeroOutput()
        {
            MemorySettings memorySettings = new MemorySettings(5, 3, 1, 1, 1);
            NtmMemory memory = new NtmMemory(memorySettings);
            memory.Randomize();

            ReadHead readHead = new ReadHeadWithFixedLastWeights(new[] { 0.2, 0.2, 0.2, 0.2, 0.2 }, memorySettings);
                                                            //KEY     //BETA   //GATE  //CONVOLUTION //GAMA SHARPENING
            readHead.UpdateAddressingData(new double[] { 0, 0, 0,      1,        1,       0, 1, 0,         1 });

            var vectorFromMemory = readHead.GetVectorFromMemory(memory);
        }

        [Test]
        public void ItIsPossibleToFindHeadThatProducesWeightFocusingOneRandomLocation()
        {
            MemorySettings memorySettings = new MemorySettings(5, 3, 1, 1, 1);
            NtmMemory memory = new NtmMemory(memorySettings);
            memory.Randomize();

            ReadHead readHead = new ReadHeadWithFixedLastWeights(new[] { 0.2, 0.2, 0.2, 0.2, 0.2 }, memorySettings);
                                                        //KEY     //BETA   //GATE  //CONVOLUTION //GAMA SHARPENING
            readHead.UpdateAddressingData(new double[] { 0, 0, 0,    1,       1,      0, 1, 0,           1 });

            var vectorFromMemory = readHead.GetWeightVector(memory);
        }

        [Test]
        public void ItIsPossibleToIterateThroughMemory()
        {
            MemorySettings memorySettings = new MemorySettings(5, 3, 1, 1, 1);
            NtmMemory memory = new NtmMemory(memorySettings);
            memory.Randomize();

            ReadHead readHead = new ReadHead(memorySettings);
                                                           //KEY     //BETA   //GATE  //CONVOLUTION //GAMA SHARPENING
            readHead.UpdateAddressingData(new double[] { 0, 0, 0,       1,      1,      1, 0, 0,        1 });
            readHead.GetVectorFromMemory(memory);
            var weight1 = readHead.LastWeights;
            readHead.UpdateAddressingData(new double[] { 0, 0, 0,       1,      0,      1, 0, 0,        1 });
            readHead.GetVectorFromMemory(memory);
            var weight2 = readHead.LastWeights;
            readHead.UpdateAddressingData(new double[] { 0, 0, 0,       1,      0,      1, 0, 0,        1 });
            readHead.GetVectorFromMemory(memory);
            var weight3 = readHead.LastWeights;
        }

        [Test]
        public void ItIsPossibleToFocusOnOneCellWhenMemoryIsEmpty()
        {
            MemorySettings memorySettings = new MemorySettings(5, 3, 1, 1, 1);
            NtmMemory memory = new NtmMemory(memorySettings);
            memory.ResetMemory();

            ReadHead readHead = new ReadHead(memorySettings);
            //KEY     //BETA   //GATE  //CONVOLUTION //GAMA SHARPENING
            readHead.UpdateAddressingData(new double[] { 0, 0, 0, 1, 1, 0, 1, 0, 1 });
            double[] weightVector = readHead.GetWeightVector(memory);
        }
    }
}
