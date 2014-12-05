using System;
using System.IO;
using AForge.Genetic;
using AForge.Math.Random;
using AForge.Neuro;
using AForge.Neuro.Learning;
using NeuralTuringMachine.Controller;
using NeuralTuringMachine.Memory;
using NeuralTuringMachine.Memory.Head;
using NeuralTuringMachine.Optimization;
using ParticleSwarmOptimization;

namespace NeuralTuringMachine.Learning
{
    public class BpttTeacher
    {
        private readonly NTMFactory _factory;
        private readonly NTM _originalMachine;
        private readonly double _learningRate;
        private NTM[] _bpttMachines;
        private double[][] _ntmOutputs;
        private readonly StreamWriter _idealReadInputLog;

        public BpttTeacher(NTMFactory factory, NTM machine, double learningRate = 0.01)
        {
            _factory = factory;
            _originalMachine = machine;
            _learningRate = learningRate;
            _idealReadInputLog = File.CreateText("IdealReadInputFitness");
        }

        public void Run(double[][] inputs, double[][] outputs)
        {
            //Make copies of Neural turing machine for bptt
            int inputCount = inputs.Length;
            _bpttMachines = new NTM[inputCount];
            _ntmOutputs = new double[inputCount][];
            double[][] idealOutputs = new double[inputCount][];
            
            //Forward propagation
            _bpttMachines[0] = _factory.CloneNTM(_originalMachine);
            for (int i = 0; i < inputCount; i++)
            {
                _ntmOutputs[i] = _bpttMachines[i].Compute(inputs[i]);

                if (i < inputCount - 1)
                {
                    _bpttMachines[i + 1] = _factory.CloneNTM(_bpttMachines[i]);
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
                    double[] idealReadInput = FindIdealReadInput(inputs[i + 1], new ControllerOutput(idealOutputs[i + 1], dataOutputLength, _bpttMachines[i + 1].Memory.MemorySettings), _bpttMachines[i + 1]);
                    
                    //READ HEAD
                    //TODO remove hack - works only for one read head
                    double[] readHeadIdealWeightVector = FindReadHeadIdealWeightVector(idealReadInput, _bpttMachines[i + 1].Memory);
                    double[] readHeadIdealOutput = FindReadHeadIdealOutput(readHeadIdealWeightVector, _bpttMachines[i].GetReadHead(0).LastWeights, _bpttMachines[i + 1].Memory);

                    //WRITE HEAD
                    //TODO remove hack - works only for one write head
                    double[] idealMemoryContent = FindIdealMemoryContent(idealReadInput, _bpttMachines[i + 1].GetReadHead(0).LastWeights, _bpttMachines[i].Memory.MemorySettings);
                    double[] writeHeadIdealOutput = FindWriteHeadIdealOutput(idealMemoryContent, _bpttMachines[i].GetWriteHead(0).LastWeights, _bpttMachines[i].Memory);

                    int offset = dataOutputLength;
                    Array.Copy(readHeadIdealOutput, 0, output, offset, readHeadIdealOutput.Length);
                    offset += readHeadIdealOutput.Length;
                    Array.Copy(writeHeadIdealOutput, 0, output, offset, writeHeadIdealOutput.Length);
                }

                idealOutputs[i] = output;
            }

            //Console.WriteLine("Ideal input error 1: {0}", idealReadInputErrors.Average());
            //Console.WriteLine("Ideal input error 2: {0}", idealReadInputErrors2.Average());

            //Backward error propagation
            for (int i = 0; i < inputCount; i++)
            {
                ResilientBackpropagationLearning bpTeacher = new ResilientBackpropagationLearning(_bpttMachines[i].Controller)
                    {
                        LearningRate = _learningRate
                    };
                //BackPropagationLearning bpTeacher = new BackPropagationLearning(_bpttMachines[i].Controller)
                //{
                //    LearningRate = _learningRate,
                //    Momentum = _momentum
                //};

                ControllerInput input = _bpttMachines[i].GetInputForController(inputs[i], i > 0 ? _bpttMachines[i - 1].LastControllerOutput : null);
                
                bpTeacher.Run(input.Input, idealOutputs[i]);
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

            foreach (NTM neuralTuringMachine in _bpttMachines)
            {
                average += ((ActivationNeuron)neuralTuringMachine.Controller.Layers[layerIndex].Neurons[neuronIndex]).Threshold;
            }

            return average / controllerCount;
        }

        private double GetAverageNeuronWeight(int layerIndex, int neuronIndex, int weightIndex)
        {
            double average = 0;
            int controllerCount = _bpttMachines.Length;

            foreach (NTM neuralTuringMachine in _bpttMachines)
            {
                average += neuralTuringMachine.Controller.Layers[layerIndex].Neurons[neuronIndex].Weights[weightIndex];
            }

            return average/controllerCount;
        }

        private double[] FindWriteHeadIdealOutput(double[] idealMemoryContent, double[] lastWeights, NtmMemory currentMemory)
        {
            int chromosomeLength = currentMemory.MemorySettings.WriteHeadLength;
            Population population =
                new Population(
                    100,
                    new NonNegativeDoubleArrayChromosome(new UniformOneGenerator(), new UniformOneGenerator(), new UniformOneGenerator(), chromosomeLength),
                    new IdealWriteHeadOutputFitnessFunction(idealMemoryContent, lastWeights, currentMemory),
                    new RouletteWheelSelection());

            return RunGenetic(population);
        }

        private double[] FindIdealMemoryContent(double[] idealReadInput, double[] readWeights, MemorySettings settings)
        {
            int chromosomeLength = settings.MemoryCellCount * settings.MemoryVectorLength;
            Population population =
                new Population(
                    100,
                    new NonNegativeDoubleArrayChromosome(new UniformOneGenerator(), new UniformOneGenerator(), new UniformOneGenerator(), chromosomeLength),
                    new IdealMemoryContentFitnessFunction(idealReadInput, readWeights, settings),
                    new RouletteWheelSelection());

            return RunGenetic(population);
        }
        

        public double[] FindIdealReadInput(double[] input, ControllerOutput idealOutput, NTM ntm)
        {
            int chromosomeLength = ntm.Controller.InputsCount - ntm.InputCount;
            Population population =
                new Population(
                    100,
                    new NonNegativeDoubleArrayChromosome(new UniformOneGenerator(DateTime.Now.Millisecond), new UniformOneGenerator(DateTime.Now.Millisecond), new UniformOneGenerator(DateTime.Now.Millisecond), chromosomeLength), 
                    new IdealInputFitnessFunction(input, idealOutput, ntm),
                    new RouletteWheelSelection());

            double[] result = RunGenetic(population);
            double fitnessMax = population.FitnessMax;
            _idealReadInputLog.WriteLine(fitnessMax);
            _idealReadInputLog.Flush();
            return result;
        }

        public double[] FindIdealReadInputPSO(double[] input, ControllerOutput idealOutput, NTM ntm)
        {
            //Swarm swarm = new Swarm();

            int chromosomeLength = ntm.Controller.InputsCount - ntm.InputCount;
            Population population =
                new Population(
                    100,
                    new NonNegativeDoubleArrayChromosome(new UniformOneGenerator(DateTime.Now.Millisecond), new UniformOneGenerator(DateTime.Now.Millisecond), new UniformOneGenerator(DateTime.Now.Millisecond), chromosomeLength),
                    new IdealInputFitnessFunction(input, idealOutput, ntm),
                    new RouletteWheelSelection());

            double[] result = RunGenetic(population);
            double fitnessMax = population.FitnessMax;
            _idealReadInputLog.WriteLine(fitnessMax);
            _idealReadInputLog.Flush();
            return result;
        }

        public double[] FindReadHeadIdealWeightVector(double[] nextIdealReadInput, NtmMemory nextMemory)
        {
            int chromosomeLength = nextMemory.MemorySettings.MemoryCellCount;
            Population population =
                new Population(
                    100,
                    new NonNegativeDoubleArrayChromosome(new UniformOneGenerator(), new UniformOneGenerator(),new UniformOneGenerator(), chromosomeLength),
                    new IdealReadWeightVectorFitnessFunction(nextIdealReadInput, nextMemory),
                    new RouletteWheelSelection());

            return RunGenetic(population);
        }

        private double[] FindReadHeadIdealOutput(double[] idealWeightVector, double[] lastWeightVector, NtmMemory nextMemory)
        {
            int chromosomeLength = nextMemory.MemorySettings.ReadHeadLength;
            Population population =
                new Population(
                    30,
                    new NonNegativeDoubleArrayChromosome(new UniformOneGenerator(), new UniformOneGenerator(), new UniformOneGenerator(), chromosomeLength),
                    new IdealReadHeadOutputFitnessFunction(idealWeightVector, lastWeightVector, nextMemory),
                    new RouletteWheelSelection());

            return RunGenetic(population);
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
                    break;
                }
            }

            NonNegativeDoubleArrayChromosome bestChromosome = (NonNegativeDoubleArrayChromosome)population.BestChromosome;
            return bestChromosome.Value;
        }
    }
}
