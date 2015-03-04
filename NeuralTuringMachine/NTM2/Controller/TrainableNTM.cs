using NTM2.Controller;
using NTM2.Memory;

namespace NTM2
{
    public class TrainableNTM
    {
        private readonly NTMController _controller;
        private readonly MemoryState _memoryState;

        public TrainableNTM(NTMController controller, MemoryState memoryState)
        {
            _controller = controller;
            _memoryState = memoryState;
        }

        public TrainableNTM(TrainableNTM oldMachine, double[] input, UnitFactory unitFactory)
        {
            _controller = oldMachine._controller.Process(oldMachine._memoryState.ReadData, input);
            for (int i = 0; i < _controller.HeadCount; i++)
            {
                _controller.Heads[i].OldHeadSettings = oldMachine._memoryState.HeadSettings[i];
            }
            _memoryState = new MemoryState(_controller.Heads, oldMachine._memoryState.Memory, unitFactory);
        }

        public NTMController Controller
        {
            get { return _controller; }
        }

        public void BackwardErrorPropagation()
        {
            _memoryState.BackwardErrorPropagation();
            _controller.BackwardErrorPropagation();
        }
    }
}
