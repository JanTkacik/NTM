namespace NTM2.Controller
{
    public class Unit
    {
        public double Value { get; set; }
        public double Gradient { get; set; }

        public Unit() 
        {
            Value = 0;
            Gradient = 0;
        }

        public Unit(double value)
        {
            Value = value;
            Gradient = 0;
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
