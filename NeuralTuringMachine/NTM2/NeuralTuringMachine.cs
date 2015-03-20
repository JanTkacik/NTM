using System;
using NTM2.Controller;
using NTM2.Learning;
using NTM2.Memory;
using NTM2.Memory.Addressing;

namespace NTM2
{
    public class NeuralTuringMachine
    {
        internal readonly UnitFactory UnitFactory;
        
        private readonly int _memoryColumnsN;
        private readonly int _memoryRowsM;
        private readonly int _weightsCount;

        private readonly double[] _input;
        private readonly ReadData[] _reads;

        internal readonly NTMMemory Memory;
        private readonly IController _controller;
        
        public int WeightsCount
        {
            get { return _weightsCount; }
        }

        public int HeadCount
        {
            get { return ((FeedForwardController)_controller).OutputLayer.HeadsNeurons.Length; }
        }

        public Head[] HeadsNeurons
        {
            get { return ((FeedForwardController)_controller).OutputLayer.HeadsNeurons; }
        }
        
        public NeuralTuringMachine(int inputSize, int outputSize, int controllerSize, int headCount, int memoryColumnsN, int memoryRowsM)
        {
            UnitFactory = new UnitFactory();
            _memoryColumnsN = memoryColumnsN;
            _memoryRowsM = memoryRowsM;
            int headUnitSize = Head.GetUnitSize(memoryRowsM);
            Memory = new NTMMemory(memoryColumnsN, memoryRowsM, headCount, UnitFactory);
            
            _controller = new FeedForwardController(controllerSize, inputSize, outputSize, headCount, memoryRowsM, UnitFactory);
            
            _weightsCount =
                (headCount * memoryColumnsN) +
                (memoryColumnsN * memoryRowsM) +
                (controllerSize * headCount * memoryRowsM) +
                (controllerSize * inputSize) +
                (controllerSize) +
                (outputSize * (controllerSize + 1)) +
                (headCount * headUnitSize * (controllerSize + 1));
        }

        private NeuralTuringMachine(
            int memoryColumnsN,
            int memoryRowsM,
            int weightsCount,
            ReadData[] readDatas,
            double[] input,
            IController controller,
            UnitFactory unitFactory)
        {
            UnitFactory = unitFactory;
            _memoryColumnsN = memoryColumnsN;
            _memoryRowsM = memoryRowsM;
            _weightsCount = weightsCount;
            _reads = readDatas;
            _input = input;
            _controller = controller;
        }
        
        public NeuralTuringMachine Process(ReadData[] readData, double[] input)
        {
            NeuralTuringMachine newController = new NeuralTuringMachine(
                _memoryColumnsN,
                _memoryRowsM,
                _weightsCount,
                readData,
                input,
                _controller.Clone(),
                UnitFactory);

            newController.ForwardPropagation(readData, input);
            return newController;
        }
        
        private void ForwardPropagation(ReadData[] readData, double[] input)
        {
            _controller.ForwardPropagation(input, readData);
        }

        public void UpdateWeights(Action<Unit> updateAction)
        {
            Memory.UpdateWeights(updateAction);
            _controller.UpdateWeights(updateAction);
        }
        
        public void BackwardErrorPropagation(double[] knownOutput)
        {
            _controller.BackwardErrorPropagation(knownOutput, _input, _reads);
        }

        public void UpdateWeights(IWeightUpdater weightUpdater)
        {
            Memory.UpdateWeights(weightUpdater);
            _controller.UpdateWeights(weightUpdater);
        }

        public double[] GetOutput()
        {
            return _controller.GetOutput();
        }
    }
}
