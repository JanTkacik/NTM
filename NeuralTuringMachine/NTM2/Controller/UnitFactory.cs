namespace NTM2.Controller
{
    public class UnitFactory
    {
        public Unit[] GetVector(int vectorSize)
        {
            Unit[] vector = new Unit[vectorSize];
            for (int i = 0; i < vectorSize; i++)
            {
                vector[i] = new Unit();
            }
            return vector;
        }

        public Unit[][] GetTensor2(int x, int y)
        {
            Unit[][] tensor = new Unit[x][];
            for (int i = 0; i < x; i++)
            {
                tensor[i] = GetVector(y);
            }
            return tensor;
        }

        public Unit[][][] GetTensor3(int x, int y, int z)
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
