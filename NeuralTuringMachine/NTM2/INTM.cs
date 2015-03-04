using System.IO;

namespace NTM2
{
    interface INTM
    {
        double[] Process(double[] input);
        void Save(Stream stream);
        void Save(string path);
    }
}
