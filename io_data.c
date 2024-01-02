/* This process is designed to generate the input and expected output data */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <unistd.h>

#include <sys/wait.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "simple_MLP.h"


int main(int argc, char* argv[])  
{
    struct shmDouble* inputs = initShmDouble("inputs", NUM_INPUTS * NUM_TRAIN_SETS, false);
    struct shmDouble* exp_outputs = initShmDouble("exp_outputs", NUM_OUTPUTS * NUM_TRAIN_SETS, false);

    /************** POPULATE INPUT AND EXP_OUTPUT ARRAYS BELOW **************/

    // XOR EXAMPLE
    int INPUT_VARS[2] = {0, 1};
    int i_train_set = 0;
    for (int i_input  = 0; i_input < sizeof(INPUT_VARS)/sizeof(INPUT_VARS[0]); i_input++)
    {
        for (int j_input = 0; j_input < sizeof(INPUT_VARS)/sizeof(INPUT_VARS[0]); j_input++)
        {
            inputs->mem[i_train_set] = INPUT_VARS[i_input];
            inputs->mem[i_train_set + 1] = INPUT_VARS[j_input];
            exp_outputs->mem[i_train_set] = abs(INPUT_VARS[i_input] - INPUT_VARS[j_input]);
            i_train_set += 2;
        }
    }
    
    /************** POPULATE INPUT AND EXP_OUTPUT ARRAYS ABOVE **************/

    return 0;
}
