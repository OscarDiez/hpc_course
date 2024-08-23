#include <stdio.h>
#include <omp.h>

int main(){
      
#pragma omp parallel sections
{
    #pragma omp section
    {
        printf ("section 1 id = %d, \n", omp_get_thread_num()); 
    }
    #pragma omp section
    {
        printf ("section 2 id = %d, \n", omp_get_thread_num());
    }
    #pragma omp section
    {
        printf ("section 3 id = %d, \n", omp_get_thread_num());
    }
}  return 0;
}