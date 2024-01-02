#include <math.h>


double sigmoid(double z, int diff)
{
    double a = 1 / (1 + exp(-z));

    if (diff == 1)	{
        return a * (1 - a);
    }
    
    return a;
}

double relu(double z, int diff)
{
    int a = 0;
    if (z > 0)
    {
        if (diff == 1)
        {
            a = 1;
        }
        else
        {
            a = z;
        }
    }

    return a;
}
