#include <iostream>
#include <omp.h>
#include <cmath>

int main(){
    double sum_i = 0;
    #pragma omp parallel for reduction(+:sum_i)
    for (int i=0; i<12; i++){
        sum_i += i;

    }
    std::cout << "Sum = " << sum_i << std::endl;
    return 0;
}