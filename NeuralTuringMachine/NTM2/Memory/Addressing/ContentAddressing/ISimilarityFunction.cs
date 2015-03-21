using NTM2.Controller;

namespace NTM2.Memory.Addressing.ContentAddressing
{
    interface ISimilarityFunction
    {
        Unit Calculate(Unit[] u, Unit[] v);
        void Differentiate(Unit similarity, Unit[] u, Unit[] v);
    }
}
