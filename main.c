#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#include <math.h>
#include <time.h>

#include <pthread.h>

#include "simple_MLP.h"
#include "activ_func.c"

#ifndef M_PI
#define M_PI (3.14159265358979323846264338327950288)
#endif

#define DEBUG true
#define TRAIN true
#define CUDA false

#define NUM_INPUTS 2
#define NUM_OUTPUTS 1
#define H_LAYERS (int[]){3, 2}

#define NUM_EPOCHS 10
#define NUM_TRAIN_SETS 4
#define ACTIV_FUNC "SIGMOID"
#define LEARN_FACTOR 2

struct outputResult
{
    double output[NUM_OUTPUTS];
    double exp_output[NUM_OUTPUTS];
    double err[NUM_OUTPUTS];
};

int* structure;
double inputs[NUM_INPUTS * NUM_TRAIN_SETS];
double exp_outputs[NUM_OUTPUTS * NUM_TRAIN_SETS];
double* z_tensor;
double* a_tensor;
double* w_tensor;
double* dw_tensor;

int output_node_index;
int num_nodes = 0;

struct outputResult forwardPass(int, int);
double nodeMVM(int, int, int);
void backPropSGD(int, int, double[NUM_OUTPUTS]);
void weightDelta(double, int, int, int, int, int);
int findMaxInt(int[], int);
void* printIntArr(int*, int, const char*);
void* printDoubleArr(double*, int, const char*);
void* trainData_XOR(void*);

int main()
{
    pthread_t train_data_thread;
    if (pthread_create(&train_data_thread, NULL, trainData_XOR, NULL) != 0)
    {
        fprintf(stderr, "Failed to create thread.\n");
        return 1;
    }

    if (pthread_join(train_data_thread, NULL) != 0)
    {
        fprintf(stderr, "Failed to join thread.\n");
        return 1;
    }

    if (DEBUG)
    {
        for (int i_train_set = 0; i_train_set < NUM_TRAIN_SETS; i_train_set++)
        {
            for (int i_input = 0; i_input < NUM_INPUTS; i_input++)
            {
                printf("INPUT_%d: %.2f ", i_input, inputs[i_train_set + i_input]);
            }
            for (int i_output = 0; i_output < NUM_OUTPUTS; i_output++)
            {
                printf("EXP_OUTPUT_%d: %.2f ", i_output, exp_outputs[i_train_set + i_output]);  
            }
            printf("\n");
        }
        printf("\n");
    }

    /* DEFINE NEURAL NETWORK LAYER AND NODE STRUCTURE AND STORE TO ARRAY */
    const int num_layers = (sizeof(H_LAYERS) / sizeof(int)) + 2;
    structure = (int*)malloc(num_layers * sizeof(int));
    for (int i_layer = 0; i_layer < num_layers; i_layer++)
    {
        if (i_layer == 0)
        {
            structure[i_layer] = NUM_INPUTS;
        }
        else if (i_layer == (num_layers - 1))
        {
            structure[i_layer] = NUM_OUTPUTS;
        }
        else
        {
            structure[i_layer] = H_LAYERS[i_layer - 1];
        }
    }

    int num_weights = 0;
    for (int i_layer = 0; i_layer < num_layers; i_layer++)
    {
        num_nodes += structure[i_layer];
        if (i_layer != num_layers - 1)
        {
            num_weights += structure[i_layer] * structure[i_layer + 1];
        }
    }

    srand((unsigned int)time(NULL));

    z_tensor = (double*)malloc((num_nodes) * sizeof(double));
    a_tensor = (double*)malloc((num_nodes) * sizeof(double));
    int index = 0;
    for (int i_layer = 1; i_layer < num_layers; i_layer++)
    {
        for (int i_node = 0; i_node < structure[i_layer]; i_node++) 
        {
            z_tensor[index] = 0;
            a_tensor[index] = 0;
            index++;
        }
    }

    output_node_index = num_nodes - structure[num_layers - 1];

    /* ALLOCATE WEIGHTS ARRAY AND POPULATE WITH RANDOM GAUSSIAN DISTRIBUTION */
    w_tensor = (double*)malloc(num_weights * sizeof(double));
    dw_tensor = (double*)malloc(num_weights * sizeof(double));
    index = 0;
    for (int i_layer = 0; i_layer < num_layers - 1; i_layer++)
    {
        for (int i_node = 0; i_node < (structure[i_layer + 1]); i_node++)
        {
            for (int j_node = 0; j_node < (structure[i_layer]); j_node++)
            {
                w_tensor[index] = (double)rand() / RAND_MAX;
                index++;
            }
        }
    }

    if (DEBUG)
    {
        printIntArr(structure, num_layers, "structure");
        printDoubleArr(z_tensor, num_nodes, "z_tensor");
        printDoubleArr(a_tensor, num_nodes, "a_tensor");
        printDoubleArr(w_tensor, num_weights, "w_tensor");
        printf("\n");
    }

    /* RUN TRAINING EPOCHS */
    struct outputResult epoch_results[NUM_TRAIN_SETS];
    for (int i_epoch = 0; i_epoch < NUM_EPOCHS; i_epoch++)
    {
        if (DEBUG)
        {
            printf("Running Epoch %d:\n", i_epoch);
        }

        for (int i_trainset = 0; i_trainset < NUM_TRAIN_SETS; i_trainset++)
        {
            // POPLATE FIRST LAYER WITH INPUTS
            for (int i_node = 0; i_node < NUM_INPUTS; i_node++)
            {
                z_tensor[i_node] = inputs[(i_trainset * NUM_INPUTS) + i_node];
            }

            epoch_results[i_trainset] = forwardPass(num_layers, i_trainset);

            if (TRAIN)
            {
                backPropSGD(num_layers, num_weights, epoch_results[i_trainset].err);
            }
        }
    }

    return 0;
}

struct outputResult forwardPass(int num_layers, int i_trainset)
{
    if (DEBUG)
    {
        int index = i_trainset * NUM_INPUTS;
        double x1 = inputs[index];
        double x2 = inputs[index + 1];
        double y = exp_outputs[i_trainset];
        printf("Running training set with inputs %f, %f and expected output %f\n", x1, x2, y);
    }

    int index = NUM_INPUTS;
    int w_index = 0;
    int z_index = 0;
    for (int i_layer = 1; i_layer < num_layers; i_layer++)
    {
        for (int i_node = 0; i_node < structure[i_layer]; i_node++)
        {
            z_tensor[index] = nodeMVM(i_layer, w_index, z_index);

            if ((i_layer - 1) != 0 || i_layer != (num_layers - 1))
            {
                if (ACTIV_FUNC == "SIGMOID")
                {
                    a_tensor[index] = sigmoid(z_tensor[index], 0);
                }
                else if (ACTIV_FUNC == "RELU")
                {
                    a_tensor[index] = relu(z_tensor[index], 0);
                }
                else
                {
                    printf("ERROR: Invalid activation function type. simple_MLP.h -> ACTIV_FUNC");
                    exit(-1);
                }
            }

            w_index += structure[i_layer - 1];
            index++;
        }

        z_index += structure[i_layer - 1];
    }

    struct outputResult result;
    for (int i_output = 0; i_output < NUM_OUTPUTS; i_output++)
    {
        result.exp_output[i_output] = exp_outputs[i_trainset + i_output];
        result.output[i_output] = z_tensor[output_node_index + i_output];
        result.err[i_output] = result.exp_output[i_output] - result.output[i_output];
    
        if (DEBUG)
        {
            printDoubleArr(z_tensor, num_nodes, "z_tensor");
            printDoubleArr(a_tensor, num_nodes, "a_tensor");
            printf("Expected Output: %.4f, numerical output: %.4f, error: %.4f\n\n",  result.exp_output[i_output], result.output[i_output], result.err[i_output]);
        }
    }

    return result;
}

double nodeMVM(int layer, int w_index, int z_index)
{
    double sum = 0;
    for (int i_value = 0; i_value < structure[layer - 1]; i_value++)
    {
        double x;
        if (layer == 1)
        {
            x = z_tensor[z_index++];
        }
        else
        {
            x = a_tensor[z_index++];
        }

        double w = w_tensor[w_index++];
        sum += x * w;
    }

    return sum;
}

double nodeMVM_CUDA(int i_layer, int i_node)
{

}

void backPropSGD(int num_layers, int num_weights, double err[NUM_OUTPUTS])
{
    for (int i_output = (NUM_OUTPUTS - 1); i_output >= 0; i_output--)
    {
        int w_index = num_weights - 1;
        int a_index = num_nodes - NUM_OUTPUTS - 1;
        double d_cost = 2 * err[i_output];
        for (int j_node = structure[num_layers - 2] - 1; j_node >=0; j_node--)
        {
            weightDelta(d_cost, (num_layers - 1), i_output, j_node, w_index--, a_index--);
        }

        for (int index = 0; index < num_weights; index++)
        {
            w_tensor[index] -= dw_tensor[index] * LEARN_FACTOR;
        }
    }
}

void weightDelta(double chain, int i_layer, int i_node, int j_node, int w_index, int a_index)
{
    if (i_layer - 1 != 0)
    {
        dw_tensor[w_index] = chain * a_tensor[a_index];
    }
    else
    {
        dw_tensor[w_index] = chain * z_tensor[a_index];
    }

    if (i_layer != 0)
    {

        chain *= w_tensor[w_index];
        if (ACTIV_FUNC == "SIGMOID")
        {
            chain *= sigmoid(z_tensor[a_index], 1);
        }
        else if (ACTIV_FUNC == "RELU")
        {
            chain += relu(z_tensor[a_index], 1);
        }

        w_index -= ((j_node * structure[i_layer]) + i_node + (structure[i_layer - 1] - (j_node + 1)) + 1);

        i_node = j_node;
        a_index -= (i_node + 1);

        i_layer--;

        for (int j_node = structure[i_layer - 1] - 1; j_node >=0; j_node--)
        {
            weightDelta(chain, i_layer, i_node, j_node, w_index, a_index);

            w_index -= structure[i_layer];
            a_index--;
        }
    }
}

int findMaxInt(int arr[], int size)
{
    if (size == 0)
    {
        printf("Error: Array is empty.\n");
        return 0;
    }

    int max = arr[0];

    for (int i = 1; i < size; i++)
    {
        if (arr[i] > max)
        {
            max = arr[i]; 
        }
    }

    return max;
}

void* printIntArr(int* arr, int arr_size, const char* arr_desc)
{
    printf("%s: [", arr_desc);
    for (int index = 0; index < arr_size; index++)
    {
        if ((index + 1) == arr_size)
        {
            printf("%d]\n", arr[index]);
            break;
        }
        printf("%d, ", arr[index]);
    }
}

void* printDoubleArr(double* arr, int arr_size, const char* arr_desc)
{
    printf("%s: [", arr_desc);
    for (int index = 0; index < arr_size; index++)
    {
        if ((index + 1) == arr_size)
        {
            printf("%.3f]\n", arr[index]);
            break;
        }
        printf("%.3f, ", arr[index]);
    }
}

void* trainData_XOR(void* arg)  
{
    // XOR EXAMPLE
    int INPUT_VARS[2] = {0, 1};
    int i_train_set = 0;
    for (int i_input  = 0; i_input < sizeof(INPUT_VARS)/sizeof(INPUT_VARS[0]); i_input++)
    {
        for (int j_input = 0; j_input < sizeof(INPUT_VARS)/sizeof(INPUT_VARS[0]); j_input++)
        {
            inputs[i_train_set * NUM_INPUTS] = INPUT_VARS[i_input];
            inputs[(i_train_set * NUM_INPUTS) + 1] = INPUT_VARS[j_input];
            exp_outputs[i_train_set] = abs(INPUT_VARS[i_input] - INPUT_VARS[j_input]);
            i_train_set++;
        }
    }

    return 0;
}
