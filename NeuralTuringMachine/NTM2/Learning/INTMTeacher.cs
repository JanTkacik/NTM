namespace NTM2.Learning
{
    interface INTMTeacher
    {
        //TODO change from trainable NTM to NTMController
        TrainableNTM[] Train(double[][] input, double[][] knownOutput);
        TrainableNTM[] Train(double[][][] inputs, double[][][] knownOutputs);
    }
}
