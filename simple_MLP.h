#include <stdbool.h>

#ifndef M_PI
#define M_PI (3.14159265358979323846264338327950288)
#endif

#define DEBUG true
#define TRAIN true
#define CUDA false

#define NUM_INPUTS 2
#define NUM_OUTPUTS 1
#define H_LAYERS (int[]){3, 2}

#define NUM_EPOCHS 3
#define NUM_TRAIN_SETS 3
#define ACTIV_FUNC "SIGMOID"

struct outputResult
{
    double output[NUM_OUTPUTS];
    double exp_output[NUM_OUTPUTS];
    double cost[NUM_OUTPUTS];
};

struct shmDouble {
    int fd;
    double* mem;
};

struct shmDouble* initShmDouble(const char* fd_name, int size, bool create)
{
    int fd;
    if (create)
    {
        fd = shm_open(fd_name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    }
    else
    {
        fd = shm_open(fd_name, O_RDWR, S_IRUSR | S_IWUSR);
    }

    if (fd == -1)
    {
        printf("shm_open failed (%s)\n", fd_name);
        exit(EXIT_FAILURE);
    }

    if (ftruncate(fd, size * sizeof(double)) == -1)
    {
        printf("ftruncate failed (%s)\n", fd_name);
        exit(EXIT_FAILURE);
    }

    double* mem = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mem == MAP_FAILED)
    {
        printf("mmap failed (%s)\n", fd_name);
        exit(EXIT_FAILURE);
    }

    struct shmDouble shm;
    shm.fd = fd;
    shm.mem = mem;

    return &shm;
}
