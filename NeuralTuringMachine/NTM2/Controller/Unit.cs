using ObjectPool;

namespace NTM2.Controller
{
    public class Unit : PoolableObjectBase 
    {
        public double Value;
        public double Gradient;
        
        public Unit(double value = 0)
        {
            Value = value;
        }

        public override string ToString()
        {
            return string.Format("Value: {0:0.000}, Gradient: {1:0.000}", Value, Gradient);
        }

        protected override void OnResetState()
        {
            Value = 0;
            Gradient = 0;
        }

        protected override void OnReleaseResources()
        {

        }
    }
}
