using System;
using AForge.Genetic;
using AForge.Math.Metrics;
using AForge.Math.Random;
using AForge.Neuro;
using NeuralTuringMachine.GeneticsOptimalization;

namespace NeuralTuringMachine.Learning
{
    public class GeneticTeacher
    {
        private readonly NeuralTuringMachine _originalMachine;

        public GeneticTeacher(NeuralTuringMachine machine)
        {
            _originalMachine = machine;
        }

        public void Run(double[][] inputs, double[][] outputs)
        {
            int chromosomeSize = GetChromosomeSize();
            Population population =
                new Population(
                    1000,
                    new NonNegativeDoubleArrayChromosome(new UniformOneGenerator(), new UniformOneGenerator(), new UniformOneGenerator(), chromosomeSize),
                    new GeneticTeacherFitnessFunction(inputs, outputs, _originalMachine.Clone()),
                    new RouletteWheelSelection());

            double[] controllerSettings = RunGenetic(population);
            ApplySettingsToController(controllerSettings, _originalMachine);
        }

        private double[] RunGenetic(Population population)
        {
            int stagnateStreak = 0;
            double lastMax = 0;
            int i;
            for (i = 0; i < 100000; i++)
            {
                population.RunEpoch();
                double fitnessMax = population.FitnessMax;
                if (fitnessMax > lastMax)
                {
                    lastMax = fitnessMax;
                    stagnateStreak = 0;
                }
                else
                {
                    stagnateStreak++;
                }

                if (stagnateStreak >= 10)
                {
                    population.Regenerate();
                    stagnateStreak = 0;
                }
                Console.WriteLine("Max: -> " + lastMax + " Population max: -> " + population.FitnessMax + " Population average: -> " + population.FitnessAvg);
            }

            NonNegativeDoubleArrayChromosome bestChromosome = (NonNegativeDoubleArrayChromosome)population.BestChromosome;
            return bestChromosome.Value;
        }

        private int GetChromosomeSize()
        {
            int chromosomeSize = 0;
            ActivationNetwork originalController = _originalMachine.Controller;
            int layersCount = originalController.Layers.Length;
            for (int i = 0; i < layersCount; i++)
            {
                Layer layer = originalController.Layers[i];
                int neuronsCount = layer.Neurons.Length;
                for (int j = 0; j < neuronsCount; j++)
                {
                    Neuron neuron = layer.Neurons[j];
                    int weightsCount = neuron.Weights.Length;
                    chromosomeSize += weightsCount + 1;
                }
            }

            return chromosomeSize;
        }

        private static void ApplySettingsToController(double[] controllerSettings, NeuralTuringMachine machine)
        {
            int index = 0;
            ActivationNetwork originalController = machine.Controller;
            int layersCount = originalController.Layers.Length;
            for (int i = 0; i < layersCount; i++)
            {
                Layer layer = originalController.Layers[i];
                int neuronsCount = layer.Neurons.Length;
                for (int j = 0; j < neuronsCount; j++)
                {
                    ActivationNeuron neuron = (ActivationNeuron)layer.Neurons[j];
                    neuron.Threshold = controllerSettings[index];
                    index++;
                    int weightsCount = neuron.Weights.Length;
                    for (int k = 0; k < weightsCount; k++)
                    {
                        neuron.Weights[k] = controllerSettings[index];
                        index++;
                    }
                }
            }
        }

        internal class GeneticTeacherFitnessFunction : IFitnessFunction
        {
            private readonly double[][] _inputs;
            private readonly double[][] _outputs;
            private readonly NeuralTuringMachine _machine;
            private readonly ISimilarity _similarity;

            public GeneticTeacherFitnessFunction(double[][] inputs, double[][] outputs, NeuralTuringMachine machine)
            {
                _inputs = inputs;
                _outputs = outputs;
                _machine = machine;
                _similarity = new EuclideanSimilarity();
            }

            public double Evaluate(IChromosome chromosome)
            {
                DoubleArrayChromosome doubleArrayChromosome = (DoubleArrayChromosome)chromosome;
                double[] controllerSettings = doubleArrayChromosome.Value;

                _machine.Memory.ResetMemory();
                ApplySettingsToController(controllerSettings, _machine);

                double similarity = 0;
                int inputCount = _inputs.Length;
                for (int i = 0; i < inputCount; i++)
                {
                    double[] computedOutput = _machine.Compute(_inputs[i]);
                    similarity += _similarity.GetSimilarityScore(computedOutput, _outputs[i]);
                }

                return similarity;
            }
        }
    }
}
