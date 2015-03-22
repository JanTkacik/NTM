using System;
using System.Diagnostics;
using System.Linq;

namespace NTM2
{
    internal static class PerfCounter
    {
        private const int SampleCount = 1000; 
        private static int _index;
        private static readonly long[] Counters = new long[SampleCount];
        private static readonly Stopwatch Stopwatch = new Stopwatch();

        private static void AddData(long ticks)
        {
            Counters[_index] = ticks;
            _index++;
            if (_index == SampleCount)
            {
                _index = 0;
                Console.WriteLine("Counter {0}", Counters.Average());
            }
        }

        internal static void Start()
        {
            Stopwatch.Restart();
        }

        internal static void Stop()
        {
            Stopwatch.Stop();
            AddData(Stopwatch.ElapsedTicks);
        }
    }
}
