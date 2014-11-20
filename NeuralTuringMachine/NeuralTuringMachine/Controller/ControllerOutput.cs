using System;

namespace NeuralTuringMachine.Controller
{
    public class ControllerOutput
    {
        public double[] DataOutput { get; private set; }
        public double[][] ReadHeadsOutputs { get; private set; }
        public double[][] WriteHeadsOutputs { get; private set; }

        public ControllerOutput(double[] rawOutput, int dataOutputLength, int readHeadCount, int readHeadLength, int writeHeadCount, int writeHeadLength)
        {
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
    }
}
