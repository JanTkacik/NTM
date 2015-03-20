using System.IO;

namespace NTM2
{
    interface INeuralTuringMachine
    {
        double[] Process(double[] input);
        void Save(Stream stream);
        void Save(string path);
    }
}
