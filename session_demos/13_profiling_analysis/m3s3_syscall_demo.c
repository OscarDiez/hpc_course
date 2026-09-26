#define _POSIX_C_SOURCE 200809L
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void){
    const char *path = "m3s3_trace_demo.tmp";
    const int records = 2000;
    const int record_bytes = 64;

    char buf[64];
    memset(buf, 'A', sizeof(buf));

    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if(fd < 0){ perror("open"); return 1; }

    for(int i=0;i<records;++i){
        if(write(fd, buf, record_bytes) != record_bytes){
            perror("write"); close(fd); return 2;
        }
    }

    if(close(fd) != 0){ perror("close"); return 3; }
    unlink(path);
    printf("TRACE_DEMO records=%d bytes_per_record=%d total_bytes=%d\n",
           records, record_bytes, records*record_bytes);
    return 0;
}
