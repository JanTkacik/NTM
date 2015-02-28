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

        public static Unit[] GetVector(int vectorSize)
        {
            Unit[] vector = new Unit[vectorSize];
            for (int i = 0; i < vectorSize; i++)
            {
                vector[i] = new Unit();
            }
            return vector;
        }

        public static Unit[][] GetTensor2(int x, int y)
        {
            Unit[][] tensor = new Unit[x][];
            for (int i = 0; i < x; i++)
            {
                tensor[i] = GetVector(y);
            }
            return tensor;
        }

        public static Unit[][][] GetTensor3(int x, int y, int z)
        {
            Unit[][][] tensor = new Unit[x][][];
            for (int i = 0; i < x; i++)
            {
                tensor[i] = GetTensor2(y, z);
            }
            return tensor;
        }
    }
}
