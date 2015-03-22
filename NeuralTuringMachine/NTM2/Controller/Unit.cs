using System;
using System.Runtime.Serialization;

namespace NTM2.Controller
{
    [DataContract]
    public class Unit 
    {
        [DataMember(Name = "V")]
        public double Value;
        [DataMember(Name = "G")]
        public double Gradient;
        
        public Unit(double value = 0)
        {
            Value = value;
        }

        public override string ToString()
        {
            return string.Format("Value: {0:0.000}, Gradient: {1:0.000}", Value, Gradient);
        }

        internal static Action<Unit[]> GetVectorUpdateAction(Action<Unit> updateAction)
        {
            return units =>
            {
                foreach (Unit unit in units)
                {
                    updateAction(unit);
                }
            };
        }

        internal static Action<Unit[][]> GetTensor2UpdateAction(Action<Unit> updateAction)
        {
            Action<Unit[]> vectorUpdateAction = GetVectorUpdateAction(updateAction);
            return units =>
            {
                foreach (Unit[] unit in units)
                {
                    vectorUpdateAction(unit);
                }
            };
        }

        internal static Action<Unit[][][]> GetTensor3UpdateAction(Action<Unit> updateAction)
        {
            Action<Unit[][]> tensor2UpdateAction = GetTensor2UpdateAction(updateAction);
            return units =>
            {
                foreach (Unit[][] unit in units)
                {
                    tensor2UpdateAction(unit);
                }
            };
        }
    }
}
