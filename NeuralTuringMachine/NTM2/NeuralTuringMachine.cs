using System;
using NTM2.Controller;
using NTM2.Learning;
using NTM2.Memory;
using NTM2.Memory.Addressing;

namespace NTM2
{
    public class NeuralTuringMachine
    {
        private readonly int _weightsCount;

        private double[] _input;
        private ReadData[] _reads;

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
            int headUnitSize = Head.GetUnitSize(memoryRowsM);
            Memory = new NTMMemory(memoryColumnsN, memoryRowsM, headCount);
            
            _controller = new FeedForwardController(controllerSize, inputSize, outputSize, headCount, memoryRowsM);
            
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
            int weightsCount,
            IController controller)
        {
            _weightsCount = weightsCount;
            _controller = controller;
        }
        
        public NeuralTuringMachine Clone()
        {
            return new NeuralTuringMachine(_weightsCount, _controller.Clone());
        }
        
        internal void ForwardPropagation(ReadData[] readData, double[] input)
        {
            _reads = readData;
            _input = input;
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
