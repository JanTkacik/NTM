using NTM2.Memory;

namespace NTM2.Controller
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
            _oldMemoryState = null;
        }

        public TrainableNTM(TrainableNTM oldMachine)
        {
            _machine = oldMachine._machine.Clone();
            _oldMemoryState = oldMachine._memoryState;
        }

        public void ForwardPropagation(double[] input)
        {
            _input = input;
            
            _machine.ForwardPropagation(_oldMemoryState.ReadData, input);

            //TODO refactor
            for (int i = 0; i < _machine.MemoryState.Memory.HeadCount; i++)
            {
                _machine.Controller.OutputLayer.HeadsNeurons[i].OldHeadSettings = _oldMemoryState.HeadSettings[i];
            }

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
    }
}
