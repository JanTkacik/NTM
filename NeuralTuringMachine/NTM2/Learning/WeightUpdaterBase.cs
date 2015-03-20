using NTM2.Controller;

namespace NTM2.Learning
{
    public abstract class WeightUpdaterBase : IWeightUpdater
    {
        public abstract void Reset();
        public abstract void UpdateWeight(Unit data);

        public void UpdateWeight(Unit[] data)
        {
            foreach (Unit unit in data)
            {
                UpdateWeight(unit);
            }
        }

        public void UpdateWeight(Unit[][] data)
        {
            foreach (Unit[] units in data)
            {
                UpdateWeight(units);
            }
        }

        public void UpdateWeight(Unit[][][] data)
        {
            foreach (Unit[][] units in data)
            {
                UpdateWeight(units);
            }
        }
    }
}
