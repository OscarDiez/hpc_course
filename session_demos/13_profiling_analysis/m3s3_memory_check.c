#include <stdio.h>
#include <stdlib.h>
#include <string.h>
__attribute__((noinline)) static void allocation(int leak){
    int *a=malloc(4096*sizeof(int));if(!a)exit(2);
    for(int i=0;i<4096;++i)a[i]=i;
    printf("MEMORY checksum=%d\n",a[1]+a[4095]);
    if(!leak)free(a); /* Intentional leak only in the named teaching mode. */
}
int main(int argc,char **argv){
    if(argc!=2||(strcmp(argv[1],"leak")&&strcmp(argv[1],"fixed")))return 2;
    allocation(!strcmp(argv[1],"leak"));return 0;
}
