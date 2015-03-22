using System.IO;

namespace NTM2
{
    interface INeuralTuringMachine
    {
        void Process(double[] input);
        double[] GetOutput();
        void Save(Stream stream);
        void Save(string path);
    }
}
