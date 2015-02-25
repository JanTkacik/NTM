using System;

namespace NTM2.Controller
{
    public static class Sigmoid
    {
        public static double GetValue(double x)
        {
            const double alpha = 1;
            return 1 / (1 + Math.Exp(-x * alpha));
        }
    }
}
