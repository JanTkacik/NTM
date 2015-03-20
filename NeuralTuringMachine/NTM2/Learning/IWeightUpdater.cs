using NTM2.Controller;

namespace NTM2.Learning
{
    public interface IWeightUpdater
    {
        void Reset();
        void UpdateWeight(Unit data);
        void UpdateWeight(Unit[] data);
        void UpdateWeight(Unit[][] data);
        void UpdateWeight(Unit[][][] data);
    }
}
