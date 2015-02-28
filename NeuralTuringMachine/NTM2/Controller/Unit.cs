namespace NTM2.Controller
{
    public class Unit
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
    }
}
