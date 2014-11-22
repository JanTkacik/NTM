using System;
using System.Collections.Generic;

namespace NeuralTuringMachine.Controller
{
    public class ControllerInput
    {
        public double[] Input { get; private set; }
        
        public ControllerInput(double[] dataInput, double[] readHeadInput, int controllerInputLength)
        {
            int dataInputLength = dataInput.Length;
            Input = new double[controllerInputLength];
            Array.Copy(dataInput, Input, dataInputLength);
            int offset = dataInputLength;
            Array.Copy(readHeadInput, 0, Input, offset, readHeadInput.Length);
        }

        public ControllerInput(double[] dataInput, IEnumerable<double[]> readHeadInput, int controllerInputLength)
        {
            int dataInputLength = dataInput.Length;
            Input = new double[controllerInputLength];
            Array.Copy(dataInput, Input, dataInputLength);
            int offset = dataInputLength;
            foreach (double[] array in readHeadInput)
            {
                Array.Copy(array, 0, Input, offset, array.Length);
                offset += array.Length;
            }
        }

        public ControllerInput(double[] dataInput, int controllerInputLength)
        {
            Random rand = new Random(DateTime.Now.Millisecond);
            Input = new double[controllerInputLength];
            Array.Copy(dataInput, Input, dataInput.Length);
            int offset = dataInput.Length;
            for (int i = offset; i < controllerInputLength; i++)
            {
                Input[i] = rand.NextDouble();
            }
        }
    }
}
