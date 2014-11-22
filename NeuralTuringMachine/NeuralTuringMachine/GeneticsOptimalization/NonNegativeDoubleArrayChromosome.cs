using System;
using AForge.Genetic;
using AForge.Math.Random;

namespace NeuralTuringMachine.GeneticsOptimalization
{
    class NonNegativeDoubleArrayChromosome : DoubleArrayChromosome
    {
        public NonNegativeDoubleArrayChromosome(IRandomNumberGenerator chromosomeGenerator, IRandomNumberGenerator mutationMultiplierGenerator, IRandomNumberGenerator mutationAdditionGenerator, int length)
            : base(chromosomeGenerator, mutationMultiplierGenerator, mutationAdditionGenerator, length)
        {
        }

        public NonNegativeDoubleArrayChromosome(IRandomNumberGenerator chromosomeGenerator, IRandomNumberGenerator mutationMultiplierGenerator, IRandomNumberGenerator mutationAdditionGenerator, double[] values)
            : base(chromosomeGenerator, mutationMultiplierGenerator, mutationAdditionGenerator, values)
        {
        }

        public NonNegativeDoubleArrayChromosome(DoubleArrayChromosome source)
            : base(source)
        {
        }

        public override IChromosome CreateNew()
        {
            return new NonNegativeDoubleArrayChromosome(chromosomeGenerator, mutationMultiplierGenerator, mutationAdditionGenerator, Length);
        }
        
        public override IChromosome Clone()
        {
            return new NonNegativeDoubleArrayChromosome(this);
        }

        public override void Crossover(IChromosome pair)
        {
            NonNegativeDoubleArrayChromosome p = (NonNegativeDoubleArrayChromosome)pair;

            // check for correct pair
            if ((p != null) && (p.Length == Length))
            {
                // crossover point
                int crossOverPoint = rand.Next(Length - 1) + 1;
                // length of chromosome to be crossed
                int crossOverLength = Length - crossOverPoint;
                // temporary array
                double[] temp = new double[crossOverLength];

                // copy part of first (this) chromosome to temp
                Array.Copy(val, crossOverPoint, temp, 0, crossOverLength);
                // copy part of second (pair) chromosome to the first
                Array.Copy(p.Value, crossOverPoint, val, crossOverPoint, crossOverLength);
                // copy temp to the second
                Array.Copy(temp, 0, p.Value, crossOverPoint, crossOverLength);
            }
        }

        public override void Mutate()
        {
            base.Mutate();
            int length = val.Length;
            for (int i = 0; i < length; i++)
            {
                while (val[i] > 1)
                {
                    val[i] = rand.NextDouble();
                }
                while (val[i] < 0)
                {
                    val[i] = rand.NextDouble();
                }
            }
        }
    }
}
