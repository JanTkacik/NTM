using System;

namespace NTM2.Controller
{
    class FeedForwardController : IController
    {
        private readonly UnitFactory _unitFactory;
        
        //Controller hidden layer threshold weights
        private readonly Unit[] _hiddenLayerThresholds;

        //Weights from input to controller
        private readonly Unit[][] _wh1x;

        private int ControllerSize { get { return _hiddenLayerThresholds.Length; } }
        
        public FeedForwardController(int controllerSize, int inputSize, UnitFactory unitFactory)
        {
            _unitFactory = unitFactory;
            
            _wh1x = _unitFactory.GetTensor2(controllerSize, inputSize);
            _hiddenLayerThresholds = _unitFactory.GetVector(controllerSize);
            
        }

        public double ForwardPropagation(double tempSum, int i, double[] input)
        {
            double sum = tempSum;

            //Foreach input
            Unit[] inputWeights = _wh1x[i];
            for (int j = 0; j < inputWeights.Length; j++)
            {
                sum += inputWeights[j].Value * input[j];
            }

            //Plus threshold
            sum += _hiddenLayerThresholds[i].Value;
            return sum;
        }

        public void UpdateWeights(Action<Unit> updateAction)
        {
            Action<Unit[]> vectorUpdateAction = Unit.GetVectorUpdateAction(updateAction);
            Action<Unit[][]> tensor2UpdateAction = Unit.GetTensor2UpdateAction(updateAction);

            tensor2UpdateAction(_wh1x);
            vectorUpdateAction(_hiddenLayerThresholds);
        }

        public void BackwardErrorPropagation(double[] hiddenLayerGradients, double[] input)
        {
            int controllerSize = ControllerSize;

            for (int i = 0; i < controllerSize; i++)
            {
                double hiddenGradient = hiddenLayerGradients[i];
                int inputLength = input.Length;
                Unit[] inputToHiddenNeuronWeights = _wh1x[i];
                for (int j = 0; j < inputLength; j++)
                {
                    inputToHiddenNeuronWeights[j].Gradient += hiddenGradient * input[j];
                }
            }
            
            for (int i = 0; i < controllerSize; i++)
            {
                _hiddenLayerThresholds[i].Gradient += hiddenLayerGradients[i];
            }
        }
    }
}
