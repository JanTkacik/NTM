using System;
using AForge.Math.Metrics;
using NeuralTuringMachine.Memory;
using NeuralTuringMachine.Memory.Head;
using NeuralTuringMachine.Misc;

namespace NeuralTuringMachine.Controller
{
    public class ControllerOutput
    {
        private MemorySettings MemorySettings { get; set; }
        private static readonly ISimilarity Similarity = new EuclideanSimilarity();
        private const double DataSimilarityWeight = 0.5;
        private const double ReadHeadSimilarityWeight = 0.25;
        private const double WriteHeadSimilarityWeight = 0.25;

        public double[] DataOutput { get; private set; }
        public double[][] ReadHeadsOutputs { get; private set; }
        public double[][] WriteHeadsOutputs { get; private set; }

        public ControllerOutput(double[] rawOutput, int dataOutputLength, MemorySettings settings)
        {
            MemorySettings = settings;
            int readHeadCount = MemorySettings.ReadHeadCount;
            int readHeadLength = MemorySettings.ReadHeadLength;
            int writeHeadCount = MemorySettings.WriteHeadCount;
            int writeHeadLength = MemorySettings.WriteHeadLength;

            DataOutput = new double[dataOutputLength];
            Array.Copy(rawOutput, DataOutput, dataOutputLength);

            int offset = dataOutputLength;

            ReadHeadsOutputs = new double[readHeadCount][];
            for (int i = 0; i < readHeadCount; i++)
            {
                ReadHeadsOutputs[i] = new double[readHeadLength];
                Array.Copy(rawOutput, offset, ReadHeadsOutputs[i], 0, readHeadLength);
                offset += readHeadLength;
            }

            WriteHeadsOutputs = new double[writeHeadCount][];
            for (int i = 0; i < writeHeadCount; i++)
            {
                WriteHeadsOutputs[i] = new double[writeHeadLength];
                Array.Copy(rawOutput, offset, WriteHeadsOutputs[i], 0, writeHeadLength);
                offset += writeHeadLength;
            }
        }

        private ControllerOutput(double[] dataOutput, double[][] readHeadOutputs, double[][] writeHeadOutputs)
        {
            DataOutput = dataOutput;
            ReadHeadsOutputs = readHeadOutputs;
            WriteHeadsOutputs = writeHeadOutputs;
        }

        public ControllerOutput Clone()
        {
            return new ControllerOutput(ArrayHelper.CloneArray(DataOutput), ArrayHelper.CloneArray(ReadHeadsOutputs), ArrayHelper.CloneArray(WriteHeadsOutputs));
        }

        public static double GetSimilarityScore(ControllerOutput outputA, ControllerOutput outputB)
        {
            double similarity = 0;
            similarity += (DataSimilarityWeight * Similarity.GetSimilarityScore(outputA.DataOutput, outputB.DataOutput));
            int readHeadCount = outputA.ReadHeadsOutputs.Length;
            for (int i = 0; i < readHeadCount; i++)
            {
                ReadHead headA = new ReadHead(outputA.ReadHeadsOutputs[i], outputA.MemorySettings);
                ReadHead headB = new ReadHead(outputB.ReadHeadsOutputs[i], outputB.MemorySettings);
                similarity += (ReadHeadSimilarityWeight * ReadHead.GetSimilarityScore(headA,headB) / readHeadCount);
            }

            int writeHeadCount = outputA.WriteHeadsOutputs.Length;
            for (int i = 0; i < writeHeadCount; i++)
            {
                WriteHead headA = new WriteHead(outputA.WriteHeadsOutputs[i], outputA.MemorySettings);
                WriteHead headB = new WriteHead(outputB.WriteHeadsOutputs[i], outputB.MemorySettings);
                similarity += (WriteHeadSimilarityWeight * WriteHead.GetSimilarityScore(headA,headB) / writeHeadCount);
            }

            return similarity;
        }
    }
}
