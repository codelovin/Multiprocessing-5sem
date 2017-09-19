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
random_walk_result walk(int a, int b, int x, double p) {

  // Result will be here
  random_walk_result res = {A, 0};

  // Implementation of random walk
  while (x != a && x != b) {
    double prob = (double) rand() / RAND_MAX;
    if (prob <= p)
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
  clock_t begin = clock();

  srand(time(NULL));

  #pragma omp parallel for reduction(+:total_moves,reached_b)
  for (size_t i = 0; i < N; i++) {
    random_walk_result res = walk(a, b, x, p);
    reached_b += (res.dest == B) ? 1 : 0;
    total_moves += res.walk_count;
  }

  clock_t end = clock();

  // Gathering data
  double b_probability = ((double) reached_b) / N;
  double average_lifetime = (double) total_moves / N;
  double execution_time = (double)(end - begin) / CLOCKS_PER_SEC;

  // Writing data
  FILE *file;
  file = fopen("stats.txt", "a");
  if (file == NULL) {
    printf("Could not create 'stats.txt'.\n");
    exit(1);
  }
  fprintf(file, "%f %f %fs %d %d %d %d %f %d\n", b_probability, average_lifetime, execution_time, a, b, x, N, p, P);
  fclose(file);

  return 0;
}
