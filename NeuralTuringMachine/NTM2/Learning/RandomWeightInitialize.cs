using System;
using NTM2.Controller;

namespace NTM2.Learning
{
    public class RandomWeightInitializer : WeightUpdaterBase
    {
        private readonly Random _rand;

        public RandomWeightInitializer(Random rand)
        {
            _rand = rand;
        }

        public override void Reset()
        {
            
        }

        public override void UpdateWeight(Unit data)
        {
            data.Value = _rand.NextDouble() - 0.5;
        }
    }
}
