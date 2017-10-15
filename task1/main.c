#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

typedef int destination;
#define A 0
#define B 1

typedef struct {
  destination dest;
  int walk_count;
} random_walk_result;

void argparse(int argc, char **argv, int *a, int *b, int *x, int *N, double *p, int *P) {
  assert(argc >= 7);
  *a = atoi(argv[1]);
  *b = atoi(argv[2]);
  *x = atoi(argv[3]);
  *N = atoi(argv[4]);
  *p = atof(argv[5]);
  *P = atoi(argv[6]);
}

/*
 * Returns true if B is reached
 */
random_walk_result walk(int a, int b, int x, double p, unsigned int seed) {

  // Result will be here
  random_walk_result res = {A, 0};

  // Implementation of random walk
  while (x != a && x != b) {
    if ((double) rand_r(&seed) <= (double) RAND_MAX * p)
      x += 1;
    else
      x -= 1;
    res.walk_count++;
  }
  res.dest = (x == a) ? A : B;
  return res;
}

int main(int argc, char **argv) {

  // Parsing arguments
  int a, b, x, N, P;
  double p;
  argparse(argc, argv, &a, &b, &x, &N, &p, &P);
  int reached_b = 0;    // Number of particles that reached b
  int total_moves = 0;  // Total moves

  omp_set_num_threads(P);

  struct timeval t0, t1;
  assert(gettimeofday(&t0, NULL) == 0);

  srand(time(NULL));
  unsigned int* seeds = (unsigned int*) malloc(sizeof(int) * N);
  for (int i = 0; i < N; i++) {
    seeds[i] = rand();
  }

  #pragma omp parallel for reduction(+:total_moves,reached_b)
  for (size_t i = 0; i < N; i++) {
    random_walk_result res = walk(a, b, x, p, seeds[i]);
    reached_b += (res.dest == B) ? 1 : 0;
    total_moves += res.walk_count;
  }

  assert(gettimeofday(&t1, NULL) == 0);

  // Gathering data
  double b_probability = ((double) reached_b) / N;
  double average_lifetime = (double) total_moves / N;
  double execution_time = ((t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec) / 1000000.0;

  // Writing data
  FILE *file;
  // file = fopen("stats.txt", "a");
  file = fopen("stats.txt", "a");
  if (file == NULL) {
    printf("Could not create 'stats.txt'.\n");
    exit(1);
  }
  fprintf(file, "%f %f %fs %d %d %d %d %f %d\n", b_probability, average_lifetime, execution_time, a, b, x, N, p, P);
  fclose(file);
  free(seeds);
  return 0;
}
