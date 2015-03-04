using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NTM2.Controller
{
    class SigmoidActivationFunction : IDifferentiableFunction
    {
        private readonly double _alpha;

        public SigmoidActivationFunction(double alpha = 1)
        {
            _alpha = alpha;
        }

        public double Value(double x)
        {
            return 1 / (1 + Math.Exp(-x * _alpha));
        }

        public double Derivative(double y)
        {
            return (_alpha * y * (1 - y));
        }
    }
}
