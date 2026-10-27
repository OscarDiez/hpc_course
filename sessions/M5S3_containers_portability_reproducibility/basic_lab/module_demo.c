#include <stdio.h>

int main(void) {
    /* Use a wide integer because the sum is larger than a 32-bit signed int. */
    long long total = 0;

    for (int number = 1; number <= 100000; number++) {
        total += number;
    }

    /* __VERSION__ identifies the compiler that built this executable. */
    printf("Compiler: %s\n", __VERSION__);
    printf("Sum 1..100000: %lld\n", total);
    return 0;
}
