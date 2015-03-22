using System;
using System.Threading.Tasks;

namespace NTM2
{
    internal static class ParallelSettings
    {
        internal static readonly ParallelOptions Options = new ParallelOptions {MaxDegreeOfParallelism = Environment.ProcessorCount};  
    }
}
