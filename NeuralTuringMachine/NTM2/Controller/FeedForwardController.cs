using NTM2.Memory.Addressing;

namespace NTM2.Controller
{
    class FeedForwardController : IController
    {
        private readonly UnitFactory _unitFactory;

        private readonly Head[] _heads;
        private readonly Unit[] _outputLayer;
        private readonly Unit[] _hiddenLayer1;

        //Weights from controller to head
        private readonly Unit[][][] _wuh1;
        //Weights from controller to output
        private readonly Unit[][] _wyh1;
        //Controller bias weights
        private readonly Unit[] _wh1b;
        //Weights from input to controller
        private readonly Unit[][] _wh1x;
        //Weights from read data to controller
        private readonly Unit[][][] _wh1r;

        public FeedForwardController(int inputSize, int outputSize, int controllerSize, int headCount)
        {

        }
    }
}
