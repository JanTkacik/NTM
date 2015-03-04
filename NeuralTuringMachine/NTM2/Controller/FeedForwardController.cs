using System;
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

        private int ControllerSize { get { return _wh1b.Length; } }

        public FeedForwardController(int inputSize, int outputSize, int controllerSize, int headCount, UnitFactory unitFactory)
        {
            _unitFactory = unitFactory;

            _wh1b = _unitFactory.GetVector(controllerSize);
        }

        public double ForwardPropagation()
        {
            int controllerSize = ControllerSize;
            double sum = 0;
            for (int i = 0; i < controllerSize; i++)
            {
                sum += _wh1b[i].Value;
            }
            return sum;
        }

        public void UpdateWeights(Action<Unit> updateAction)
        {
            Action<Unit[]> vectorUpdateAction = Unit.GetVectorUpdateAction(updateAction);
            vectorUpdateAction(_wh1b);
        }

        public void BackwardErrorPropagation(double[] hiddenGradients)
        {
            int controllerSize = ControllerSize;
            for (int i = 0; i < controllerSize; i++)
            {
                _wh1b[i].Gradient += hiddenGradients[i];
            }
        }
    }
}
