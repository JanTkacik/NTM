using System;
using AForge.Genetic;
using AForge.Math.Random;

namespace NeuralTuringMachine.GeneticsOptimalization
{
    class ControllerInputChromosome : DoubleArrayChromosome
    {
        public ControllerInputChromosome(IRandomNumberGenerator chromosomeGenerator, IRandomNumberGenerator mutationMultiplierGenerator, IRandomNumberGenerator mutationAdditionGenerator, int length)
            : base(chromosomeGenerator, mutationMultiplierGenerator, mutationAdditionGenerator, length)
        {
        }

        public ControllerInputChromosome(IRandomNumberGenerator chromosomeGenerator, IRandomNumberGenerator mutationMultiplierGenerator, IRandomNumberGenerator mutationAdditionGenerator, double[] values)
            : base(chromosomeGenerator, mutationMultiplierGenerator, mutationAdditionGenerator, values)
        {
        }

        public ControllerInputChromosome(DoubleArrayChromosome source)
            : base(source)
        {
        }

        public override IChromosome CreateNew()
        {
            return new ControllerInputChromosome(chromosomeGenerator, mutationMultiplierGenerator, mutationAdditionGenerator, Length);
        }
        
        public override IChromosome Clone()
        {
            return new ControllerInputChromosome(this);
        }


        public override void Crossover(IChromosome pair)
        {
            ControllerInputChromosome p = (ControllerInputChromosome)pair;

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
    }
}
