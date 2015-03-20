using NTM2.Controller;

namespace NTM2.Learning
{
    class GradientResetter : WeightUpdaterBase
    {
        public override void Reset()
        {
          
        }

        public override void UpdateWeight(Unit data)
        {
            data.Gradient = 0;
        }

    }
}
