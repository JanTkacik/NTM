using System;
using NTM2.Controller;
using NTM2.Learning;
using NTM2.Memory;

namespace NTM2
{
    public class TrainableNTM
    {
        private readonly NeuralTuringMachine _machine;
        private readonly MemoryState _oldMemoryState;
        private MemoryState _memoryState;

        private double[] _input;

        public TrainableNTM(NeuralTuringMachine machine)
        {
            _machine = machine;
            _memoryState = new MemoryState(_machine.MemoryState.Memory);
            _memoryState.DoInitialReading();
            _oldMemoryState = null;
        }

        public TrainableNTM(TrainableNTM oldMachine)
        {
            _machine = oldMachine._machine.Clone();
            _oldMemoryState = oldMachine._memoryState;
        }

        public TrainableNTM(int inputSize, int outputSize, int controllerSize, int headCount, int memoryColumnsN, int memoryRowsM)
        {
            _machine = new NeuralTuringMachine(inputSize, outputSize, controllerSize, headCount, memoryColumnsN, memoryRowsM);
        }

        public NeuralTuringMachine Machine
        {
            get
            {
                return _machine;
            }
        }

        public void ForwardPropagation(double[] input)
        {
            _input = input;
            
            _machine.ForwardPropagation(_oldMemoryState, input);

            _memoryState = new MemoryState(_machine.Controller.OutputLayer.HeadsNeurons, _oldMemoryState.Memory);
        }
        
        public void BackwardErrorPropagation(double[] knownOutput)
        {
            _memoryState.BackwardErrorPropagation();
            _machine.Controller.BackwardErrorPropagation(knownOutput, _input, _oldMemoryState.ReadData);
        }

        public void BackwardErrorPropagation()
        {
            _memoryState.BackwardErrorPropagation2();
        }

        public double[] GetOutput()
        {
            return _machine.GetOutput();
        }

        //TODO replace with weight initialization in constructor
        public void UpdateWeights(Action<Unit> updateAction)
        {
            _machine.UpdateWeights(updateAction);
        }

        public void UpdateWeights(IWeightUpdater weightUpdater)
        {
            _machine.UpdateWeights(weightUpdater);
        }
    }
}
