using System;
using AForge.Genetic;
using AForge.Math.Random;
using AForge.Neuro;
using AForge.Neuro.Learning;
using NeuralTuringMachine.Controller;
using NeuralTuringMachine.GeneticsOptimalization;
using BackPropagationLearning = NeuralTuringMachine.Misc.BackPropagationLearning;

namespace NeuralTuringMachine.Learning
{
    public class BpttTeacher
    {
        private readonly NeuralTuringMachine _originalMachine;
        private readonly double _learningRate;
        private readonly double _momentum;
        private NeuralTuringMachine[] _bpttMachines;
        private double[][] _ntmOutputs;

        public BpttTeacher(NeuralTuringMachine machine, double learningRate = 0.5, double momentum = 0.1)
        {
            _originalMachine = machine;
            _learningRate = learningRate;
            _momentum = momentum;
        }

        public void Run(double[][] inputs, double[][] outputs)
        {
            //Make copies of Neural turing machine for bptt
            int inputCount = inputs.Length;
            _bpttMachines = new NeuralTuringMachine[inputCount];
            _ntmOutputs = new double[inputCount][];
            double[][] _idealOutputs = new double[inputCount][];
            
            //Forward propagation
            _bpttMachines[0] = _originalMachine.Clone();
            for (int i = 0; i < inputCount; i++)
            {
                _ntmOutputs[i] = _bpttMachines[i].Compute(inputs[i]);

                if (i < inputCount - 1)
                {
                    _bpttMachines[i + 1] = _bpttMachines[i].Clone();
                }
            }
            
            //Find ideal outputs
            for (int i = inputCount - 1; i >= 0; i--)
            {
                double[] controllerOutput = _bpttMachines[i].Controller.Output;
                int outputLength = controllerOutput.Length;

                double[] output = new double[outputLength];

                int dataOutputLength = outputs[i].Length;
                Array.Copy(outputs[i], output, dataOutputLength);

                if (i == inputCount - 1)
                {
                    Array.Copy(controllerOutput, dataOutputLength, output, dataOutputLength, outputLength - dataOutputLength);
                }
                else
                {
                    double[] idealReadInput = FindIdealReadInput(inputs[i + 1], _idealOutputs[i + 1], _bpttMachines[i + 1]);
                    
                    //READ HEAD
                    //TODO remove hack - works only for one read head
                    double[] readHeadIdealWeightVector = FindReadHeadIdealWeightVector(idealReadInput, _bpttMachines[i + 1]);
                    double[] readHeadIdealOutput = FindReadHeadIdealOutput(readHeadIdealWeightVector, _bpttMachines[i].GetReadHead(0).LastWeights, _bpttMachines[i + 1]);

                    //WRITE HEAD
                    //TODO remove hack - works only for one write head
                    double[] idealMemoryContent = FindIdealMemoryContent(idealReadInput, _bpttMachines[i + 1].GetReadHead(0).LastWeights, _bpttMachines[i].Memory.CellCount, _bpttMachines[i].Memory.MemoryVectorLength);
                    double[] writeHeadIdealOutput = FindWriteHeadIdealOutput(idealMemoryContent, _bpttMachines[i].GetWriteHead(0).LastWeights, _bpttMachines[i]);

                    int offset = dataOutputLength;
                    Array.Copy(readHeadIdealOutput, 0, output, offset, readHeadIdealOutput.Length);
                    offset += readHeadIdealOutput.Length;
                    Array.Copy(writeHeadIdealOutput, 0, output, offset, writeHeadIdealOutput.Length);
                }

                _idealOutputs[i] = output;
            }

            //Backward error propagation
            for (int i = 0; i < inputCount; i++)
            {
                BackPropagationLearning bpTeacher = new BackPropagationLearning(_bpttMachines[i].Controller)
                {
                    LearningRate = _learningRate,
                    Momentum = _momentum
                };

                ControllerInput input = _bpttMachines[i].GetInputForController(inputs[i], i > 0 ? _bpttMachines[i - 1].LastControllerOutput : null);

                bpTeacher.Run(input.Input, _idealOutputs[i]);
            }

            //Average the networks
            ActivationNetwork originalController = _originalMachine.Controller;
            int layersCount = originalController.Layers.Length;
            for (int i = 0; i < layersCount; i++)
            {
                Layer layer = originalController.Layers[i];
                int neuronsCount = layer.Neurons.Length;
                for (int j = 0; j < neuronsCount; j++)
                {
                    ActivationNeuron neuron = (ActivationNeuron)layer.Neurons[j];
                    int weightsCount = neuron.Weights.Length;
                    for (int k = 0; k < weightsCount; k++)
                    {
                        neuron.Weights[k] = GetAverageNeuronWeight(i,j,k);
                    }
                    neuron.Threshold = GetAverageNeuronThreshold(i, j);
                }
            }
        }

        private double GetAverageNeuronThreshold(int layerIndex, int neuronIndex)
        {
            double average = 0;
            int controllerCount = _bpttMachines.Length;

            foreach (NeuralTuringMachine neuralTuringMachine in _bpttMachines)
            {
                average += ((ActivationNeuron)neuralTuringMachine.Controller.Layers[layerIndex].Neurons[neuronIndex]).Threshold;
            }

            return average / controllerCount;
        }

        private double GetAverageNeuronWeight(int layerIndex, int neuronIndex, int weightIndex)
        {
            double average = 0;
            int controllerCount = _bpttMachines.Length;

            foreach (NeuralTuringMachine neuralTuringMachine in _bpttMachines)
            {
                average += neuralTuringMachine.Controller.Layers[layerIndex].Neurons[neuronIndex].Weights[weightIndex];
            }

            return average/controllerCount;
        }

        private double[] FindWriteHeadIdealOutput(double[] idealMemoryContent, double[] lastWeights, NeuralTuringMachine currentNTM)
        {
            int chromosomeLength = currentNTM.WriteHeadLength;
            Population population =
                new Population(
                    100,
                    new NonNegativeDoubleArrayChromosome(new UniformOneGenerator(), new UniformOneGenerator(), new UniformOneGenerator(), chromosomeLength),
                    new IdealWriteHeadOutputFitnessFunction(idealMemoryContent, lastWeights, currentNTM.MaxConvolutialShift, currentNTM.Memory),
                    new RouletteWheelSelection());

            return RunGenetic(population);
        }

        private double[] FindIdealMemoryContent(double[] idealReadInput, double[] readWeights, int memoryCellCount, int memoryVectorLength)
        {
            int chromosomeLength = memoryCellCount*memoryVectorLength;
            Population population =
                new Population(
                    100,
                    new NonNegativeDoubleArrayChromosome(new UniformOneGenerator(), new UniformOneGenerator(), new UniformOneGenerator(), chromosomeLength),
                    new IdealMemoryContentFitnessFunction(idealReadInput, readWeights, memoryCellCount, memoryVectorLength),
                    new RouletteWheelSelection());

            return RunGenetic(population);
        }
        

        public double[] FindIdealReadInput(double[] input, double[] idealOutput, NeuralTuringMachine ntm)
        {
            int chromosomeLength = ntm.Controller.InputsCount - ntm.InputCount;
            Population population =
                new Population(
                    100,
                    new NonNegativeDoubleArrayChromosome(new UniformOneGenerator(), new UniformOneGenerator(),new UniformOneGenerator(), chromosomeLength), 
                    new IdealInputFitnessFunction(input, idealOutput, ntm.Controller),
                    new RouletteWheelSelection());

            return RunGenetic(population);
        }

        public double[] FindReadHeadIdealWeightVector(double[] nextIdealReadInput, NeuralTuringMachine nextTM)
        {
            int chromosomeLength = nextTM.Memory.CellCount;
            Population population =
                new Population(
                    100,
                    new NonNegativeDoubleArrayChromosome(new UniformOneGenerator(), new UniformOneGenerator(),new UniformOneGenerator(), chromosomeLength),
                    new IdealReadWeightVectorFitnessFunction(nextIdealReadInput, nextTM.Memory),
                    new RouletteWheelSelection());

            return RunGenetic(population);
        }

        private double[] FindReadHeadIdealOutput(double[] idealWeightVector, double[] lastWeightVector, NeuralTuringMachine nextTM)
        {
            int chromosomeLength = nextTM.ReadHeadLength;
            Population population =
                new Population(
                    100,
                    new NonNegativeDoubleArrayChromosome(new UniformOneGenerator(), new UniformOneGenerator(), new UniformOneGenerator(), chromosomeLength),
                    new IdealReadHeadOutputFitnessFunction(idealWeightVector, lastWeightVector, nextTM.MaxConvolutialShift, nextTM.Memory),
                    new RouletteWheelSelection());

            return RunGenetic(population);
        }

        private double[] RunGenetic(Population population)
        {
            int stagnateStreak = 0;
            double lastMax = 0;

            for (int i = 0; i < 100000; i++)
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
                    break;
                }
                //Console.WriteLine("Genetics: iteration - " + i + " fitness avg - " + population.FitnessAvg + " fitness max - " + fitnessMax);
            }

            NonNegativeDoubleArrayChromosome bestChromosome = (NonNegativeDoubleArrayChromosome)population.BestChromosome;
            return bestChromosome.Value;
        }
    }
}
