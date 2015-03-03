using System;
using NTM2.Controller;

namespace NTM2.Learning
{
    //SEE http://arxiv.org/pdf/1308.0850v5.pdf page 23
    public class RMSPropTeacher
    {
        private readonly NTMController _controller;
        private readonly double[] _n;
        private readonly double[] _g;
        private readonly double[] _delta;

        public RMSPropTeacher(NTMController controller)
        {
            _controller = controller;
            _n = new double[controller.WeightsCount];
            _g = new double[controller.WeightsCount];
            _delta = new double[controller.WeightsCount];
        }

        public Ntm[] Train(
            double[][] input,
            double[][] knownOutput,
            double gradientMomentum,
            double deltaMomentum,
            double changeMultiplier,
            double changeAddConstant)
        {
            Ntm[] machines = _controller.ProcessAndUpdateErrors(input, knownOutput);
            int i = 0;
            _controller.UpdateWeights(unit =>
                {
                    _n[i] = (gradientMomentum * _n[i]) + ((1 - gradientMomentum) * unit.Gradient * unit.Gradient);
                    _g[i] = (gradientMomentum * _g[i]) + ((1 - gradientMomentum) * unit.Gradient);
                    _delta[i] = (deltaMomentum * _delta[i]) - (changeMultiplier * (unit.Gradient / Math.Sqrt(_n[i] - (_g[i] * _g[i]) + changeAddConstant)));
                    unit.Value += _delta[i];
                    i++;
                });
            return machines;
        }
    }
}
