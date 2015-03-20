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
            _memoryState = new MemoryState(_machine);
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
            
            _machine.ForwardPropagation(_oldMemoryState.ReadDatas, input);

            for (int i = 0; i < _machine.HeadCount; i++)
            {
                _machine.HeadsNeurons[i].OldHeadSettings = _oldMemoryState.HeadSettings[i];
            }

            _memoryState = new MemoryState(_machine.HeadsNeurons, _oldMemoryState.Memory);
        }
        
        public void BackwardErrorPropagation(double[] knownOutput)
        {
            _memoryState.BackwardErrorPropagation();
            _machine.Controller.BackwardErrorPropagation(knownOutput, _input, _oldMemoryState.ReadDatas);
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
