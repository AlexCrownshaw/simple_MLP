/* This process is designed to generate the input and expected output data */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "simple_MLP.h"


void* trainData_XOR(void* arg)  
{

    /************** POPULATE INPUT AND EXP_OUTPUT ARRAYS BELOW **************/

    // XOR EXAMPLE
    int INPUT_VARS[2] = {0, 1};
    int i_train_set = 0;
    for (int i_input  = 0; i_input < sizeof(INPUT_VARS)/sizeof(INPUT_VARS[0]); i_input++)
    {
        for (int j_input = 0; j_input < sizeof(INPUT_VARS)/sizeof(INPUT_VARS[0]); j_input++)
        {
            inputs[i_train_set] = INPUT_VARS[i_input];
            inputs[i_train_set + 1] = INPUT_VARS[j_input];
            exp_outputs[i_train_set] = abs(INPUT_VARS[i_input] - INPUT_VARS[j_input]);
            i_train_set += 2;
        }
    }
    
    /************** POPULATE INPUT AND EXP_OUTPUT ARRAYS ABOVE **************/

    return 0;
}
