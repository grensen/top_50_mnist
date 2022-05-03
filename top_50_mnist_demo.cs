System.Action<string> print = System.Console.WriteLine;

print("\nBegin fast convolutional neural network demo\n");

string path = @"C:\fast_cnn\";
// 0. load MNIST data
AutoData d = new(path);

// 1. init cnn + nn
int startDimension = 28; // or = (int)sqrt(784)
var isCnn    = true; // true = cnn or false = nn
int[] cnn = { 1, 8, 16 }; // non-RGB = 1 (MNIST) or RGB = 3 (CIFAR-10), cnn input layer dimension
int[] filter = { 5, 3 }; // x * y dim
int[] stride = { 2, 1 }; // replaces pooling with higher strides than 1  
int[] net    = { 784, 300, 300, 300, 10 }; // nn
var lr = 0.005f;
var momentum = 0.5f;
var lr_mlt = 0.99f;
var mom_mlt = 0.5f;

// 2.0 out conv dimensions            
int[] dim = GetCnnDimensions(cnn.Length - 1, startDimension, filter, stride);
// 2.1 convolution steps for layer wise preparation
int[] cStep = GetCnnSteps(cnn, dim);
// 2.2 kernel steps for layer wise preparation
int[] kStep = GetKernelSteps(cnn, filter);

// 3.0 init visual based kernel weights
float[] kernel = InitKernel(cnn, filter);
// 3.1 init neural network weights
float[] weight = NeuralNetWeightInit(net, cStep, isCnn, 12345);

// print info stuff
NetInfo();

// get time
DateTime elapsed = DateTime.Now;

// to check sample probability count
int[] isDone = new int[60000];
// MNIST training (60k) for n epochs
int seed = 12345;
for (int epoch = 0; epoch < 41; epoch++, lr *= lr_mlt, momentum *= mom_mlt)
{
    CnnTraining(d, ref seed, isDone, isCnn, cnn, filter, stride, kernel, dim, cStep, kStep, net, weight, 60000, epoch, lr, momentum);
    if(epoch > 19) CnnTraining(d, ref seed, isDone, isCnn, cnn, filter, stride, kernel, dim, cStep, kStep, net, weight, 10000);
}
Console.WriteLine("Done after " + ((DateTime.Now - elapsed).TotalMilliseconds / 1000.0).ToString("F2") + "s\n");
// MNIST test 10k 
CnnTraining(d, ref seed, isDone, isCnn, cnn, filter, stride, kernel, dim, cStep, kStep, net, weight, 10000);

ConvNetworkSave(cnn, filter, stride, kernel, net, weight, path, (path + "myCNN.txt"));
print("\nLoad CNN again and test it");

// new cnn
int[] cnn2 = null, filter2 = null, stride2 = null, net2 = null, dim2 = null, cStep2 = null, kStep2 = null;
float[] kernel2 = null, weight2 = null;
// load trained cnn
ConvNetworkLoad(ref cnn2, ref filter2, ref stride2, ref dim2, ref cStep2, ref kStep2, ref kernel2, ref net2, ref weight2, path, (path + "myCNN.txt"));
// retest
CnnTraining(d, ref seed, isDone, isCnn, cnn2, filter2, stride2, kernel2, dim2, cStep2, kStep2, net2, weight2, 10000);

print("\nEnd MNIST CNN demo");


// 5.0 core run
static void CnnTrainingNew(AutoData d, ref int seed, int[] isDone, // data stuff
    bool isCnn, int[] cnn, int[] filter, int[] stride, float[] kernel, int[] dim, int[] cSteps, int[] kStep, // cnn stuff
    int[] net, float[] weight,  // nn stuff
    int len, int epoch = -1, float lr = 0, float mom = 0) // optional hyperparameters for training only
{
    static int FastRand(ref int seed) { return ((seed = (214013 * seed + 2531011)) >> 16) & 0x7FFF; } // [0, 32768)
    DateTime elapsed = DateTime.Now;
    int correct = 0, all = 0;
    
    // change from training to test if no learning rate
    bool training = lr == 0 ? false : true;
    
    // cnn stuff
    int cnn_layerLen = cnn.Length - 1;
    int cnn_neuronLen = GetCnnNeuronsLen(28, cnn, dim);
    int cOutput = cnn_neuronLen - cSteps[cnn_layerLen];
    
    // nn stuff
    int weightLen = weight.Length, neuronLen = GetNeuronsLen(net);
    int layer = net.Length - 1, input = net[0], output = net[layer]; // output neurons
    int inputHidden = neuronLen - output; // size of input and hidden neurons

    // correction value for each neural network weight    
    float[] delta = training ? new float[weight.Length] : null;

    float[] kernel_delta = training ? new float[kernel.Length] : null;
    
    // start training each epoch
    for (int x = 1, batch = 1; x < len + 1; x++)
    {
        // drop if count reaches threshold
       // if (training && isDone[x - 1] >= 2) continue;
        // get target label
        int target = d.GetLabel(x - 1, training);
        // create neurons arrays for nn and cnn
        float[] neuron = null, conv = null;
        // feed input sample and set arrays
        if (isCnn) // cnn active
        {
            // feed image and set arrays
            conv = FeedSample(d.GetSample(x - 1, training), cnn_neuronLen);
            neuron = new float[neuronLen];
            // convolution feed forward
            if (!training)
                ConvolutionForward(cnn, dim, cSteps, filter, kStep, stride, conv, kernel);
            else
                ConvolutionForwardDropout(cnn, dim, cSteps, filter, kStep, stride, conv, kernel, ref seed);

          //  ConvolutionForwardDropout(cnn, dim, cSteps, filter, kStep, stride, conv, kernel, ref seed);

            for (int i = 0, ii = cSteps[cnn_layerLen]; i < cOutput; i++)
                neuron[i] = conv[ii + i];
            /*
            // send last cnn layer to first nn layer
            if (!training)
            {
                for (int i = 0, ii = cSteps[cnn_layerLen]; i < cOutput; i++)
                    neuron[i] = conv[ii + i];
            }
            else
                for (int i = 0, ii = cSteps[cnn_layerLen]; i < cOutput; i++)
                {
                    float cv = conv[ii + i];
                    if (cv > 0)
                    { 
                        if(FastRand(ref seed) / 32767.0f > 0.4f)
                            neuron[i] = cv;
                        else
                            neuron[i] = conv[ii + i] = 0;   
                    }
                }
            */
        }
        else // just neural net, no cnn
            neuron = FeedSample(d.GetSample(x - 1, training), neuronLen);

        // neural net feed forward
        FeedForward(net, neuron, weight, layer, output, inputHidden);

        int prediction = ArgMax(output, inputHidden, neuron);
        Sm(inputHidden, output, neuron[inputHidden + prediction], neuron);

        // general network prediction 
        correct += prediction == target ? 1 : 0; all++; // true count
        // test zone ends here
        if (!training) continue; // dirty
        // probability check with count to drop high confident samples over epochs
        if (neuron[inputHidden + prediction] >= 0.99) { isDone[x - 1]++; continue; }
        if (neuron[inputHidden + target] >= 0.99) continue;

        float[] gradient = new float[neuronLen];// nn gradients array
        float[] cnnGradient = new float[cSteps[cnn_layerLen] + cOutput];// cnn gradients array 
    
        // output error (target - output)
        for (int n = inputHidden; n < neuronLen; n++) 
            gradient[n] = target == n - inputHidden ? 1 - neuron[n] : -neuron[n];

        // nn backprop
        Backprop(net, neuron, weight, gradient, delta, layer, inputHidden, neuronLen, weightLen);
        // count batch size
        batch++;
        // cnn backprop

        if (isCnn)
        {
            // sent gradient from first nn input layer to last cnn layer 
            for (int i = 0, ii = cSteps[cnn_layerLen]; i < cOutput; i++, ii++)
                cnnGradient[ii] = gradient[i];

            // convolution backprop with kernel update - TODO: add delta and batch support
            ConvolutionBackprop(cnn, dim, cSteps, filter, kStep, stride, conv, kernel, kernel_delta, cnnGradient, lr);
        }

        // update
        if (prediction == target) continue;
      //  for (int i = 0; i < kernel.Length; i++)
        {
          //  kernel[i] += kernel_delta[i] * lr;
          //  kernel_delta[i] = 0;
        }
        Update(net, weight, delta, layer, (neuronLen / layer * 1.0f) / (batch + 1), lr, mom);
        batch = 0;
    } // runs end

    if (lr == 0) Console.WriteLine("Accuracy on test data = " + (correct * 100.0 / all).ToString("F2")
            + "% after " + ((DateTime.Now - elapsed).TotalMilliseconds / 1000.0).ToString("F2") + "s");
    else Console.WriteLine("epoch = " + epoch.ToString().PadLeft(2) + "  |  acc = " + (correct * 100.0 / all).ToString("F2").PadLeft(6)
            + "%  |  time = " + ((DateTime.Now - elapsed).TotalMilliseconds / 1000.0).ToString("F2") + "s");
    
    static int ArgMax(int output, int start, float[] neuron)
    {
        float max = neuron[start]; // init class 0 prediction      
        int prediction = 0; // init class 0 prediction
        for (int i = 1; i < output; i++)
        {
            float n = neuron[i + start];
            if (n > max) { max = n; prediction = i; } // grab maxout prediction here
        }
        return prediction;
    }
    static void Sm(int inputHidden, int output, float max, float[] neuron)
    {
        float scale = 0; // softmax with max trick
        for (int n = inputHidden, N = inputHidden + output; n != N; n++) scale += neuron[n] = MathF.Exp(neuron[n] - max);
        for (int n = inputHidden, N = inputHidden + output; n != N; n++) neuron[n] = neuron[n] / scale;
    }
    static int GetCnnNeuronsLen(int startDimension, int[] cnn, int[] dim)
    {
        int cnn_layerLen = cnn.Length - 1;
        int cnn_neuronLen = startDimension * startDimension; // add input first
        for (int i = 0; i < cnn_layerLen; i++)
            cnn_neuronLen += cnn[i + 1] * dim[i + 1] * dim[i + 1];
        return cnn_neuronLen;
    }
    static int GetNeuronsLen(int[] net)
    {
        int sum = 0;
        for (int n = 0; n < net.Length; n++) sum += net[n];
        return sum;
    }
}
static void CnnTraining(AutoData d, ref int seed, int[] isDone, // data stuff
    bool isCnn, int[] cnn, int[] filter, int[] stride, float[] kernel, int[] dim, int[] cSteps, int[] kStep, // cnn stuff
    int[] net, float[] weight,  // nn stuff
    int len, int epoch = -1, float lr = 0, float mom = 0) // optional hyperparameters for training only
{
    static int FastRand(ref int seed) { return ((seed = (214013 * seed + 2531011)) >> 16) & 0x7FFF; } // [0, 32768)
    DateTime elapsed = DateTime.Now;
    int correct = 0, all = 0;

    // change from training to test if no learning rate
    bool training = lr == 0 ? false : true;

    // cnn stuff
    int cnn_layerLen = cnn.Length - 1;
    int cnn_neuronLen = GetCnnNeuronsLen(28, cnn, dim);
    int cOutput = cnn_neuronLen - cSteps[cnn_layerLen];

    // nn stuff
    int weightLen = weight.Length, neuronLen = GetNeuronsLen(net);
    int layer = net.Length - 1, input = net[0], output = net[layer]; // output neurons
    int inputHidden = neuronLen - output; // size of input and hidden neurons

    // correction value for each neural network weight    
    float[] delta = training ? new float[weight.Length] : null;
    float[] kernel_delta = training ? new float[kernel.Length] : null;

    // start training each epoch
    for (int x = 1, batch = 1; x < len + 1; x++)
    {
        // drop if count reaches threshold
        // if (training && isDone[x - 1] >= 2) continue;
        // get target label
        int target = d.GetLabel(x - 1, training);
        // create neurons arrays for nn and cnn
        float[] neuron = null, conv = null;
        // feed input sample and set arrays
        if (isCnn) // cnn active
        {
            // feed image and set arrays
            conv = FeedSample(d.GetSample(x - 1, training), cnn_neuronLen);
            neuron = new float[neuronLen];
            // convolution feed forward
            if (!training)
                ConvolutionForward(cnn, dim, cSteps, filter, kStep, stride, conv, kernel);
            else
                ConvolutionForwardDropout(cnn, dim, cSteps, filter, kStep, stride, conv, kernel, ref seed);

            for (int i = 0, ii = cSteps[cnn_layerLen]; i < cOutput; i++)
                neuron[i] = conv[ii + i];
        }
        else // just neural net, no cnn
            neuron = FeedSample(d.GetSample(x - 1, training), neuronLen);

        // neural net feed forward
        FeedForward(net, neuron, weight, layer, output, inputHidden);

        int prediction = ArgMax(output, inputHidden, neuron);
        Sm(inputHidden, output, neuron[inputHidden + prediction], neuron);

        // general network prediction 
        correct += prediction == target ? 1 : 0; all++; // true count
        // test zone ends here
        if (!training) continue; // dirty
        // probability check with count to drop high confident samples over epochs
        if (neuron[inputHidden + prediction] >= 0.99) { isDone[x - 1]++; continue; }
        if (neuron[inputHidden + target] >= 0.99) continue;

        float[] gradient = new float[neuronLen];// nn gradients array
        float[] cnnGradient = new float[cSteps[cnn_layerLen] + cOutput];// cnn gradients array 

        // output error (target - output)
        for (int n = inputHidden; n < neuronLen; n++)
            gradient[n] = target == n - inputHidden ? 1 - neuron[n] : -neuron[n];

        // nn backprop
        Backprop(net, neuron, weight, gradient, delta, layer, inputHidden, neuronLen, weightLen);
        // count batch size
        batch++;
        // cnn backprop
        if (isCnn)
        {
            // sent gradient from first nn input layer to last cnn layer 
            for (int i = 0, ii = cSteps[cnn_layerLen]; i < cOutput; i++, ii++)
                cnnGradient[ii] = gradient[i];

            // convolution backprop with kernel update - TODO: add delta and batch support
            ConvolutionBackprop(cnn, dim, cSteps, filter, kStep, stride, conv, kernel, kernel_delta, cnnGradient,lr);
        }

        // update
        if (prediction == target) continue;

        Update(net, weight, delta, layer, (neuronLen / layer * 1.0f) / (batch + 1), lr, mom);
        batch = 0;
    } // runs end

    if (lr == 0) Console.WriteLine("Accuracy on test data = " + (correct * 100.0 / all).ToString("F2")
            + "% after " + ((DateTime.Now - elapsed).TotalMilliseconds / 1000.0).ToString("F2") + "s");
    else Console.WriteLine("epoch = " + epoch.ToString().PadLeft(2) + "  |  acc = " + (correct * 100.0 / all).ToString("F2").PadLeft(6)
            + "%  |  time = " + ((DateTime.Now - elapsed).TotalMilliseconds / 1000.0).ToString("F2") + "s");

    static int ArgMax(int output, int start, float[] neuron)
    {
        float max = neuron[start]; // init class 0 prediction      
        int prediction = 0; // init class 0 prediction
        for (int i = 1; i < output; i++)
        {
            float n = neuron[i + start];
            if (n > max) { max = n; prediction = i; } // grab maxout prediction here
        }
        return prediction;
    }
    static void Sm(int inputHidden, int output, float max, float[] neuron)
    {
        float scale = 0; // softmax with max trick
        for (int n = inputHidden, N = inputHidden + output; n != N; n++) scale += neuron[n] = MathF.Exp(neuron[n] - max);
        for (int n = inputHidden, N = inputHidden + output; n != N; n++) neuron[n] = neuron[n] / scale;
    }
    static int GetCnnNeuronsLen(int startDimension, int[] cnn, int[] dim)
    {
        int cnn_layerLen = cnn.Length - 1;
        int cnn_neuronLen = startDimension * startDimension; // add input first
        for (int i = 0; i < cnn_layerLen; i++)
            cnn_neuronLen += cnn[i + 1] * dim[i + 1] * dim[i + 1];
        return cnn_neuronLen;
    }
    static int GetNeuronsLen(int[] net)
    {
        int sum = 0;
        for (int n = 0; n < net.Length; n++) sum += net[n];
        return sum;
    }
}

void NetInfo()
{
    if (isCnn) print("Convolution = " + string.Join(",", cnn).Replace(",", "-"));
    if (isCnn) print("Kernel size =   " + string.Join(",", filter).Replace(",", "-"));
    if (isCnn) print("Stride step =   " + string.Join(",", stride).Replace(",", "-"));
    if (isCnn) print("DimensionX  =  " + string.Join(",", dim).Replace(",", "-"));
    if (isCnn) print("Map (Dim²)  = " + string.Join(",", dim.Select((x, index) => x * x).ToArray()).Replace(",", "-"));
    if (isCnn) print("CNN = " + string.Join(",", cnn.Select((x, index) => x * dim[index] * dim[index]).ToArray()).Replace(",", "-"));
    print("NN  =         " + string.Join(",", net).Replace(",", "-"));
    if (isCnn) print("Kernel weights  = " + kernel.Length.ToString());
    print("Network weights = " + weight.Length.ToString());
    print("Learning = " + lr.ToString("F3") + " | MLT = " + lr_mlt.ToString("F2"));
    print("Momentum = " + momentum.ToString("F2") + "  | MLT = " + mom_mlt.ToString("F2"));
    print("\nStarting training");
}

// 2.0 create conv dimensions              
static int[] GetCnnDimensions(int cnn_layerLen, int startDimension, int[] filter, int[] stride)
{
    int[] dim = new int[cnn_layerLen + 1];
    for (int i = 0, c_dim = (dim[0] = startDimension); i < cnn_layerLen; i++)
        dim[i + 1] = c_dim = (c_dim - (filter[i] - 1)) / stride[i];
    return dim;
}
// 2.1 convolution steps 
static int[] GetCnnSteps(int[] cnn, int[] dim)
{
    int cnn_layerLen = cnn.Length - 1;
    int[] cs = new int[cnn_layerLen + 2];
    cs[1] = dim[0] * dim[0]; // startDimension^2
    for (int i = 0, sum = cs[1]; i < cnn_layerLen; i++)
        cs[i + 2] = sum += cnn[i + 1] * dim[i + 1] * dim[i + 1];
    return cs;
}
// 2.2 kernel steps
static int[] GetKernelSteps(int[] cnn, int[] filter)
{
    //  steps in stucture for kernel weights 
    int cnn_layerLen = cnn.Length - 1;
    int[] ks = new int[cnn_layerLen];
    for (int i = 0; i < cnn_layerLen - 1; i++)
        ks[i + 1] += cnn[i + 0] * cnn[i + 1] * filter[i] * filter[i];
    return ks;
}

// 3.0 init visual based kernel weights
static float[] InitKernel(int[] cnn, int[] filter)
{
    int cnn_layerLen = cnn.Length - 1, cnn_weightLen = 0;

    for (int i = 0; i < cnn_layerLen; i++) cnn_weightLen += cnn[i + 0] * cnn[i + 1] * filter[i] * filter[i];
    float[] kernel = new float[cnn_weightLen];
    Erratic rnd = new(1234567);
    for (int i = 0, c = 0; i < cnn_layerLen; i++)
    {
        float sd = MathF.Sqrt(6.0f / ((cnn[i] + cnn[i + 1]) * filter[i] * filter[i]));
        for (int j = 0, f = filter[i]; j < cnn[i + 1]; j++)
            for (int k = 0; k < cnn[i + 0]; k++)
                for (int u = 0; u < f; u++)
                    for (int v = 0; v < f; v++, c++)
                        // kernel[c] = rnd.NextFloat(-1.0f / (f * f / 1.0f), 1.0f / (f * f / 1.0f)); //
                        //  kernel[c] = rnd.NextFloat(-sd * 1.0f, sd * 1.0f) * 1.0f;
                        kernel[c] = rnd.NextFloat(-1.0f, 1.0f) / (filter[i] * filter[i] * 0.5f);
    }
    return kernel;
}
// 3.1 init neural network weights
static float[] NeuralNetWeightInit(int[] net, int[] cStep, bool isCnn, int seed)
{
    // 3.0.1 fit cnn input to nn output
    if (isCnn) SetNeuralNetInputDimension(cStep, net);
    // 3.1 init neural network weights
    return Glorot(net, seed);

    // 3.0.1 fit nn input to cnn output
    static void SetNeuralNetInputDimension(int[] convStep, int[] net)
    {
        net[0] = convStep[^1] - convStep[^2]; // cnn output length
    }
    // 3.1 glorot nn weights init     
    static float[] Glorot(int[] net, int seed)
    {
        int len = 0;
        for (int n = 0; n < net.Length - 1; n++) len += net[n] * net[n + 1];
        float[] weight = new float[len];
        Erratic rnd = new(seed);
        for (int i = 0, w = 0; i < net.Length - 1; i++, w += net[i - 0] * net[i - 1]) // layer
        {
            float sd = (float)Math.Sqrt(6.0f / (net[i] + net[i + 1]));
            for (int m = w; m < w + net[i] * net[i + 1]; m++) // weights
                weight[m] = rnd.NextFloat(-sd * 1.0f, sd * 1.0f);
        }
        return weight;
    }
}

// 4.0 input sample
static float[] FeedSample(Sample s, int neuronLen)
{
    float[] neuron = new float[neuronLen];
    for (int i = 0; i < 784; i++) neuron[i] = s.sample[i];
    return neuron;
}
// 4.1 cnn ff
static void ConvolutionForward(int[] cnn, int[] dim, int[] cs, int[] filter, int[] kstep, int[] stride, float[] conv, float[] kernel)
{
    for (int i = 0; i < cnn.Length - 1; i++)
    {
        int left = cnn[i], right = cnn[i + 1], lDim = dim[i], rDim = dim[i + 1], lStep = cs[i + 0], rStep = cs[i + 1],
            kd = filter[i], ks = kstep[i], st = stride[i], lMap = lDim * lDim, rMap = rDim * rDim, kMap = kd * kd, sDim = st * lDim;

        // convolution
        for (int l = 0, ls = lStep; l < left; l++, ls += lMap) // input channel feature map 
            for (int r = 0, rs = rStep; r < right; r++, rs += rMap) // output channel feature map 
            {
                int k = rs; // output map position 
                for (int y = 0, w = ks + (l * right + r) * kMap; y < rDim; y++) // conv dim y
                    for (int x = 0; x < rDim; x++, k++) // conv dim x
                    {
                        float sum = 0;
                        int j = ls + y * sDim + x * st; // input map position for kernel operation
                        for (int col = 0, fid = 0; col < kd; col++) // filter dim y 
                            for (int row = col * lDim, len = row + kd; row < len; row++, fid++) // filter dim x     
                                sum += conv[j + row] * kernel[w + fid];
                        conv[k] += sum;
                    }
            }

        // relu activation
        for (int r = 0, kN = rStep; r < right; r++, kN += rMap) // output maps 
            for (int k = kN, K = k + rMap; k < K; k++) // conv map
            {
                float sum = conv[k];
                conv[k] = sum > 0 ? sum * left : 0; // relu activation for each neuron
            }
    }
}

static void ConvolutionForwardDropout(int[] cnn, int[] dim, int[] cs, int[] filter, int[] kstep, int[] stride, float[] conv, float[] kernel, ref int seed)
{
    static int FastRand(ref int seed) { return ((seed = (214013 * seed + 2531011)) >> 16) & 0x7FFF; } // [0, 32768)
    for (int i = 0; i < cnn.Length - 1; i++)
    {
        int left = cnn[i], right = cnn[i + 1], lDim = dim[i], rDim = dim[i + 1], lStep = cs[i + 0], rStep = cs[i + 1],
            kd = filter[i], ks = kstep[i], st = stride[i], lMap = lDim * lDim, rMap = rDim * rDim, kMap = kd * kd, sDim = st * lDim;

        // convolution
        for (int l = 0, ls = lStep; l < left; l++, ls += lMap) // input channel feature map 
            for (int r = 0, rs = rStep; r < right; r++, rs += rMap) // output channel feature map 
            {
                int k = rs; // output map position 
                for (int y = 0, w = ks + (l * right + r) * kMap; y < rDim; y++) // conv dim y
                    for (int x = 0; x < rDim; x++, k++) // conv dim x
                    {
                        float sum = 0;
                        int j = ls + y * sDim + x * st; // input map position for kernel operation
                        for (int col = 0, fid = 0; col < kd; col++) // filter dim y 
                            for (int row = col * lDim, len = row + kd; row < len; row++, fid++) // filter dim x     
                                sum += conv[j + row] * kernel[w + fid];
                        conv[k] += sum;
                    }
            }
        // activation
        if (i == cnn.Length - 2)
            for (int r = 0, kN = rStep; r < right; r++, kN += rMap) // output maps 
                for (int k = kN, K = k + rMap; k < K; k++) // conv map
                {
                    float sum = conv[k];
                    conv[k] = sum > 0 && FastRand(ref seed) / 32767.0f > 0.5f ? sum * left : 0; // relu activation for each neuron
                }
        else
            for (int r = 0, kN = rStep; r < right; r++, kN += rMap) // output maps 
                for (int k = kN, K = k + rMap; k < K; k++) // conv map
                {
                    float sum = conv[k];
                    conv[k] = sum > 0 ? sum * left : 0; // relu activation for each neuron
                }
    }
}

// 4.2 nn ff
static void FeedForward(int[] net, float[] neuron, float[] weight, int layer, int output, int inputHidden)
{
    //  feed forward
    for (int i = 0, k = net[0], w = 0, j = 0; i < layer; i++)
    {
        int left = net[i], right = net[i + 1];
        for (int l = 0; l < left; l++)
        {
            float n = neuron[j + l];
            if (n > 0) for (int r = 0; r < right; r++) neuron[k + r] += n * weight[w + r];
            w += right;
        }
        j += left; k += right;
    }

}
// 4.3 nn bp
static void Backprop(int[] net, float[] neuron, float[] weight, float[] gradient, float[] delta,
    int layer, int inputHidden, int neuronLen, int weightLen)
{

    // all gradients mlp
    for (int i = layer - 1, j = inputHidden, k = neuronLen, ww = weightLen; i >= 0; i--)
    {
        int left = net[i], right = net[i + 1]; j -= left; k -= right; ww -= right * left;
        for (int l = 0, w = ww; l < left; l++)
        {
            float gra = 0, n = neuron[j + l];
            if (n > 0) for (int r = 0; r < right; r++)
                    gra += weight[w + r] * gradient[k + r];
            w += right;
            gradient[j + l] = gra;
        }
    }
    // all deltas mlp
    for (int i = layer - 1, j = inputHidden, k = neuronLen, ww = weightLen; i >= 0; i--)
    {
        int left = net[i], right = net[i + 1]; j -= left; k -= right; ww -= right * left;
        for (int l = 0, w = ww; l < left; l++)
        {
            float n = neuron[j + l];
            if (n > 0) for (int r = 0; r < right; r++)
                    delta[w + r] += n * gradient[k + r];
            w += right;
        }
    }
}
// 4.4 cnn bp        
static void ConvolutionBackprop(int[] cnn, int[] dim, int[] cs, int[] filter, int[] kstep, int[] stride, float[] conv, float[] kernel, float[] kernel_delta, float[] cGradient, float lr)
{
    // convolution gradient
    for (int i = cnn.Length - 2; i >= 1; i--)
        for (int left = cnn[i], right = cnn[i + 1], lDim = dim[i], rDim = dim[i + 1], lStep = cs[i + 0], rStep = cs[i + 1],
            kd = filter[i], ks = kstep[i], st = stride[i], lMap = lDim * lDim, rMap = rDim * rDim, kMap = kd * kd, sDim = st * lDim, l = 0, ls = lStep; l < left; l++, ls += lMap) // input channel feature map 
            for (int r = 0, rs = rStep; r < right; r++, rs += rMap) // output channel feature map 
                for (int y = 0, k = rs, w = ks + (l * right + r) * kMap; y < rDim; y++) // conv dim y
                    for (int x = 0; x < rDim; x++, k++) // conv dim x
                        if (conv[k] > 0) // relu derivative
                        {
                            float gra = cGradient[k];
                            int j = ls + y * sDim + x * st; // input map position 
                            for (int col = 0, fid = 0; col < kd; col++) // filter dim y cols
                                for (int row = col * lDim, len = row + kd; row < len; row++, fid++) // filter dim x rows    
                                    cGradient[j + row] += kernel[w + fid] * gra;
                        }

    // kernel delta with kernel weights update 
    for (int i = cnn.Length - 2; i >= 0; i--)
        for (int left = cnn[i], right = cnn[i + 1], lDim = dim[i], rDim = dim[i + 1], lStep = cs[i + 0], rStep = cs[i + 1],
            kd = filter[i], ks = kstep[i], st = stride[i], lMap = lDim * lDim, rMap = rDim * rDim, kMap = kd * kd, sDim = st * lDim, l = 0, ls = lStep;
            l < left; l++, ls += lMap) // input channel feature map 
            for (int r = 0, rs = rStep; r < right; r++, rs += rMap) // output channel feature map 
                for (int y = 0, k = rs, w = ks + (l * right + r) * kMap; y < rDim; y++) // conv dim y
                    for (int x = 0; x < rDim; x++, k++) // conv dim x
                        if (conv[k] > 0) // relu derivative
                        {
                            float gra = cGradient[k];
                            int j = ls + y * sDim + x * st; // input map position 
                            for (int col = 0, fid = 0; col < kd; col++) // filter dim y cols
                                for (int row = col * lDim, len = row + kd; row < len; row++, fid++) // filter dim x rows    
                                    kernel_delta[w + fid] += conv[j + row] * gra * 0.005f;// * 0.5f;
                        }
}
// 4.5 update
static void Update(int[] net, float[] weight, float[] delta, int layer, float mlt, float lr, float mom)
{
    for (int i = 0, mStep = 0; i < layer; i++, mStep += net[i - 0] * net[i - 1]) // layers
    {
        float oneUp = (float)Math.Sqrt(2.0f / (net[i + 1] + net[i])) * mlt;
        for (int m = mStep, mEnd = mStep + net[i] * net[i + 1]; m < mEnd; m++) // weights 
        {
            float del = delta[m], s2 = del * del;
            if (s2 > oneUp) continue; // check overwhelming deltas
            weight[m] += del * lr;
            delta[m] = del * mom;
        }
    }
}

// 6.0 save network
static void ConvNetworkSave(int[] cnn, int[] filter, int[] stride, float[] kernel, int[] net, float[] weight, string path, string name)
{
    // check directory
    if (!Directory.Exists(path)) Directory.CreateDirectory(path);

    // get file len
    int weightLen = kernel.Length + 1 + weight.Length + 1;
    // create string array 
    string[] netString = new string[weightLen];
    // add convolutional network first
    netString[0] = string.Join(",", cnn) + "+" + string.Join(",", filter) + "+" + string.Join(",", stride); // neural network at first line
    // add kernel weights
    for (int i = 1; i < kernel.Length + 1; i++)
        netString[i] = ((decimal)((double)kernel[i - 1])).ToString(); // for precision
    // add neural network 
    netString[kernel.Length + 1] = string.Join(",", net); // neural network at first line
    // add neural network weights
    for (int i = kernel.Length + 2, ii = 0; i < weightLen; i++, ii++)
        netString[i] = ((decimal)((double)weight[ii])).ToString(); // for precision
    // save file
    File.WriteAllLines(name, netString);
}
// 6.1 load network
static void ConvNetworkLoad(ref int[] cnn, ref int[] filter, ref int[] stride, ref int[] dim, ref int[] cStep, ref int[] kStep, ref float[] kernel,
    ref int[] net, ref float[] weight, string path, string name)
{

    FileStream Readfiles = new(name, FileMode.Open, FileAccess.Read);
    string[] backup = File.ReadLines(name).ToArray();

    // split first line with cnn 
    string[] netPartOne = backup[0].Split('+');
    // get convolutional network layer len

    // load cnn
    cnn = netPartOne[0].Split(',').Select(int.Parse).ToArray();
    // load filter
    filter = netPartOne[1].Split(',').Select(int.Parse).ToArray();
    // load stride
    stride = netPartOne[2].Split(',').Select(int.Parse).ToArray();

    // get cnn stuff
    dim = GetCnnDimensions(cnn.Length - 1, 28, filter, stride);
    cStep = GetCnnSteps(cnn, dim);
    kStep = GetKernelSteps(cnn, filter);

    // load kernel weights
    int cnn_layerLen = cnn.Length - 1, cnn_weightLen = 0;
    // get neurons len
    for (int i = 0; i < cnn_layerLen; i++) cnn_weightLen += cnn[i + 0] * cnn[i + 1] * filter[i] * filter[i];
    // resize kernel array 
    kernel = new float[cnn_weightLen];
    for (int n = 1; n < cnn_weightLen + 1; n++)
        kernel[n - 1] = float.Parse(backup[n]);

    // CNN READY !!!

    // get NN after CNN index + info line
    string netPartTwo = backup[cnn_weightLen + 1];
    // load network from info line
    net = netPartTwo.Split(',').Select(int.Parse).ToArray();
    // resize weight array 
    weight = new float[backup.Length - (cnn_weightLen + 1)];
    // load weights

    for (int n = cnn_weightLen + 2, nn = 0; n < backup.Length; n++, nn++)
        weight[nn] = float.Parse(backup[n]);

    // close file
    Readfiles.Close(); // don't forget to close!
}

//
struct Sample
{
    public float[] sample;
    public int label;
}
struct AutoData // https://github.com/grensen/easy_regression#autodata
{
    public string source;
    public byte[] samplesTest, labelsTest;
    public byte[] samplesTraining, labelsTraining;

    public AutoData(string yourPath)
    {
        this.source = yourPath;

        // hardcoded urls from my github
        string trainDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-images.idx3-ubyte";
        string trainLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-labels.idx1-ubyte";
        string testDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-images.idx3-ubyte";
        string testnLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-labels.idx1-ubyte";

        // change easy names 
        string d1 = @"trainData", d2 = @"trainLabel", d3 = @"testData", d4 = @"testLabel";

        if (!File.Exists(yourPath + d1)
            || !File.Exists(yourPath + d2)
              || !File.Exists(yourPath + d3)
                || !File.Exists(yourPath + d4))
        {
            DateTime elapsed = DateTime.Now;
            Console.WriteLine("Data does not exist");
            if (!Directory.Exists(yourPath)) Directory.CreateDirectory(yourPath);

            // padding bits: data = 16, labels = 8
            Console.WriteLine("Download MNIST dataset from GitHub");
            this.samplesTraining = (new System.Net.WebClient().DownloadData(trainDataUrl)).Skip(16).Take(60000 * 784).ToArray();
            this.labelsTraining = (new System.Net.WebClient().DownloadData(trainLabelUrl)).Skip(8).Take(60000).ToArray();
            this.samplesTest = (new System.Net.WebClient().DownloadData(testDataUrl)).Skip(16).Take(10000 * 784).ToArray();
            this.labelsTest = (new System.Net.WebClient().DownloadData(testnLabelUrl)).Skip(8).Take(10000).ToArray();
           
            Console.WriteLine("Downdload complete after " + ((DateTime.Now - elapsed).TotalMilliseconds / 1000.0).ToString("F2") + "s");

            Console.WriteLine("Save cleaned MNIST data into folder " + yourPath + "\n");
            File.WriteAllBytes(yourPath + d1, this.samplesTraining);
            File.WriteAllBytes(yourPath + d2, this.labelsTraining);
            File.WriteAllBytes(yourPath + d3, this.samplesTest);
            File.WriteAllBytes(yourPath + d4, this.labelsTest);
            return;
        }
        // data on the system, just load from yourPath
        System.Console.WriteLine("Load MNIST data from " + yourPath + "\n");
        this.samplesTraining = File.ReadAllBytes(yourPath + d1).Take(60000 * 784).ToArray();
        this.labelsTraining = File.ReadAllBytes(yourPath + d2).Take(60000).ToArray();
        this.samplesTest = File.ReadAllBytes(yourPath + d3).Take(10000 * 784).ToArray();
        this.labelsTest = File.ReadAllBytes(yourPath + d4).Take(10000).ToArray();
    }
    public Sample GetSample(int id, bool isTrain)
    {
        Sample s = new();
        s.sample = new float[784];

        if (isTrain) for (int i = 0; i < 784; i++)
                s.sample[i] = samplesTraining[id * 784 + i] / 255f;
        else for (int i = 0; i < 784; i++)
                s.sample[i] = samplesTest[id * 784 + i] / 255f;

        s.label = isTrain ? labelsTraining[id] : labelsTest[id];
        return s;
    }
    public int GetLabel(int id, bool isTrain)
    {
        return isTrain ? labelsTraining[id] : labelsTest[id];
    }
}
class Erratic // https://jamesmccaffrey.wordpress.com/2019/05/20/a-pseudo-pseudo-random-number-generator/
{
    private float seed;
    public Erratic(float seed2)
    {
        this.seed = this.seed + 0.5f + seed2;  // avoid 0
    }
    public float Next()
    {
        double x = Math.Sin(this.seed) * 1000;
        double result = x - Math.Floor(x);  // [0.0,1.0)
        this.seed = (float)result;  // for next call
        return (float)result;
    }
    public float NextFloat(float lo, float hi)
    {
        float x = this.Next();
        return (hi - lo) * x + lo;
    }
};