#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#include <math.h>
#include <time.h>
#include <unistd.h>

#include <sys/wait.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "simple_MLP.h"
#include "activ_func.c"

struct shmDouble* inputs;
struct shmDouble* exp_outputs;

int* structure;
double* z_tensor;
double* a_tensor;
double* w_tensor;

int output_node_index;

struct outputResult runInference(int, int);
double nodeMulAcc(int, int , int);
void backPropSGD(int, int);
void weightUpdate(double, int, int, int);
int findMaxInt(int[], int);
double randGaussian();

int main()
{
    inputs = initShmDouble("inputs", NUM_INPUTS * NUM_TRAIN_SETS, true);
    exp_outputs = initShmDouble("exp_outputs", NUM_OUTPUTS * NUM_TRAIN_SETS, true);

    pid_t IO_DATA_PID = fork();
    if (IO_DATA_PID == 0)
    {
        char* args[] = {"./io_data", NULL};
        execv("./io_data", args);
    }
    else 
    {
        int io_data_status;
        waitpid(IO_DATA_PID, &io_data_status, 0);
        if (WEXITSTATUS(io_data_status) != 0)
        {
            printf("IO_DATA process failed with exit code %d\n", WEXITSTATUS(io_data_status));
        }
        
        if (DEBUG)
        {
            for (int i_train_set = 0; i_train_set < NUM_TRAIN_SETS; i_train_set++)
            {
                printf("DEBUG: ");
                for (int i_input = 0; i_input < NUM_INPUTS; i_input++)
                {
                    printf("INPUT_%f: %f ", i_input, inputs->mem[i_train_set + i_input]);
                }
                for (int i_output = 0; i_output < NUM_OUTPUTS; i_output++)
                {
                    printf("EXP_OUTPUT_%f: %f ", i_output, exp_outputs->mem[i_train_set + i_output]);  
                }
                printf("\n");
            }
        }
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

    int num_nodes = 0;
    int num_weights = 0;
    for (int i_layer = 0; i_layer < num_layers; i_layer++)
    {
        num_nodes += structure[i_layer];
        if (i_layer != num_layers - 1)
        {
            num_weights += structure[i_layer] * structure[i_layer + 1];
        }
    }

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
    srand((unsigned int)time(NULL));

    w_tensor = (double*)malloc(num_weights * sizeof(double));
    index = 0;
    for (int i_layer = 0; i_layer < num_layers - 1; i_layer++)
    {
        for (int i_node = 0; i_node < (structure[i_layer + 1]); i_node++)
        {
            for (int j_node = 0; j_node < (structure[i_layer]); j_node++)
            {
                w_tensor[index] = randGaussian();
                index++;
            }
        }
    }

    /* RUN TRAINING EPOCHS */
    struct outputResult epoch_results[NUM_TRAIN_SETS];
    for (int i_epoch = 0; i_epoch < NUM_EPOCHS; i_epoch++)
    {
        for (int i_input = 0; i_input < NUM_TRAIN_SETS; i_input++)
        {
            // POPLATE FIRST LAYER WITH INPUTS
            for (int i_node = 0; i_node < NUM_INPUTS; i_node++)
            {
                z_tensor[i_node] = inputs->mem[(i_input * NUM_INPUTS) + i_node];
            }

            epoch_results[i_input] = runInference(num_layers, i_input);

            if (TRAIN)
            {
                backPropSGD(i_input, num_layers);
            }
        }
    }

    return 0;
}

struct outputResult runInference(int num_layers, int i_trainset)
{
    if (DEBUG)
    {
        int index = i_trainset * NUM_INPUTS;
        double x1 = inputs->mem[index];
        double x2 = inputs->mem[index + 1];
        int y = exp_outputs->mem[i_trainset];
        printf("DEBUG: Running training set with inputs %f, %f and expected output %f\n", x1, x2, y);
    }

    int index = 0;
    for (int i_layer = 1; i_layer < num_layers; i_layer++)
    {
        for (int i_node = 0; i_node < structure[i_layer]; i_layer++)
        {
            z_tensor[index] = nodeMulAcc(num_layers, i_layer, i_node);

            if (i_layer != (num_layers - 1))
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

            index++;
        }
    }

    struct outputResult result;
    for (int i_output = 0; i_output < NUM_OUTPUTS; i_output++)
    {
        result.exp_output[i_output] = exp_outputs->mem[i_trainset + i_output];
        result.output[i_output] = a_tensor[output_node_index + i_output];
        result.cost[i_output] = (result.exp_output[i_output] - result.output[i_output]);
    }

    return result;
}

double nodeMulAcc(int num_layers, int layer, int node)
{
    int x_index = 0;
    int w_index = 0;
    for (int i_layer = 0; i_layer < (layer - 1); i_layer++)
    {
        x_index += structure[i_layer];
        w_index += structure[i_layer] * structure[i_layer + 1];
    }
    w_index += node * structure[layer - 1];

    double sum = 0;
    for (int i_value = 0; i_value < structure[layer - 1]; i_value++)
    {
        double x;
        if (layer == 1)
        {
            x = z_tensor[x_index++];
        }
        else
        {
            x = a_tensor[x_index++];
        }

        double w = w_tensor[w_index++];

        sum += x * w;
    }

    return sum;
}

double nodeMulAccCUDA(int i_layer, int i_node)
{

}

void backPropSGD(int i_input, int num_layers)
{
    double cost_diff = 0;

    // cost_diff = pow((VALUES[num_layers - 1][i_output] - EXP_OUTPUTS[num_layers - 1][i_output]), 2);
    for (int i_layer = num_layers - 1; i_layer > 0; i_layer--)
    {
        for (int j_node = 0; j_node < structure[i_layer]; j_node++)
        {
            for (int k_node = 0; k_node < structure[i_layer]; k_node++)
            {
                weightUpdate(cost_diff, i_layer, j_node, k_node);
            }
        }
    }
}

void weightUpdate(double cost_diff, int i_layer, int j, int k)
{
    double delta = 0;
    for (int i_node = 0; i_node < structure[i_layer]; i_node++)
    {
        
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

double randGaussian()
{
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;

    // Apply the Box-Muller transform to get two independent standard normal variables
    double z0 = fabs(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));

    return z0;
}
