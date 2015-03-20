using System;
using NTM2.Controller;

namespace NTM2.Learning
{
    //SEE http://arxiv.org/pdf/1308.0850v5.pdf page 23
    public class RMSPropTeacher
    {
        private readonly NTMController _controller;
        public double GradientMomentum { get; private set; }
        public double DeltaMomentum { get; private set; }
        public double ChangeMultiplier { get; private set; }
        public double ChangeAddConstant { get; private set; }
        private readonly double[] _n;
        private readonly double[] _g;
        private readonly double[] _delta;

        public RMSPropTeacher(NTMController controller, double gradientMomentum = 0.95, double deltaMomentum = 0.5, double changeMultiplier = 0.001, double changeAddConstant = 0.001)
        {
            _controller = controller;
            GradientMomentum = gradientMomentum;
            DeltaMomentum = deltaMomentum;
            ChangeMultiplier = changeMultiplier;
            ChangeAddConstant = changeAddConstant;
            _n = new double[controller.WeightsCount];
            _g = new double[controller.WeightsCount];
            _delta = new double[controller.WeightsCount];
        }

        public TrainableNTM[] Train(double[][] input, double[][] knownOutput)
        {
            TrainableNTM[] machines = _controller.ProcessAndUpdateErrors(input, knownOutput);
            int i = 0;
            _controller.UpdateWeights(unit =>
                {
                    _n[i] = (GradientMomentum * _n[i]) + ((1 - GradientMomentum) * unit.Gradient * unit.Gradient);
                    _g[i] = (GradientMomentum * _g[i]) + ((1 - GradientMomentum) * unit.Gradient);
                    _delta[i] = (DeltaMomentum * _delta[i]) - (ChangeMultiplier * (unit.Gradient / Math.Sqrt(_n[i] - (_g[i] * _g[i]) + ChangeAddConstant)));
                    unit.Value += _delta[i];
                    i++;
                });
            return machines;
        }
    }
}
