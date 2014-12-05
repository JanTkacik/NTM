using AForge.Neuro;

namespace NeuralTuringMachine.Controller
{
    class RectifiedLinearActivationFunction : IActivationFunction 
    {
        private readonly double _alpha;

        public RectifiedLinearActivationFunction(double alpha = 1)
        {
            _alpha = alpha;
        }

        public double Function(double x)
        {
            if (x < 0)
            {
                return 0;
            }
            return _alpha*x;
        }

        public double Derivative(double x)
        {
            if (x < 0)
            {
                return 0;
            }
            return _alpha;
        }

        public double Derivative2(double y)
        {
            if (y < 0)
            {
                return 0;
            }
            return _alpha;
        }
    }
}
