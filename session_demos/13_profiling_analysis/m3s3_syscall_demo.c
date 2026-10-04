#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char **argv){
    const char *mode=argc>1?argv[1]:"small";
    int records=argc>2?atoi(argv[2]):4000, chunk_records=argc>3?atoi(argv[3]):256;
    if((strcmp(mode,"small")&&strcmp(mode,"buffered"))||records<1||records>100000||chunk_records<1||chunk_records>4096)return 2;
    char path[]="m3s3_io_XXXXXX";int fd=mkstemp(path);if(fd<0){perror("mkstemp");return 2;}
    size_t total=(size_t)records*64, chunk=!strcmp(mode,"small")?64:(size_t)chunk_records*64;
    char *buf=malloc(chunk);if(!buf){close(fd);unlink(path);return 2;}memset(buf,'A',chunk);
    int calls=0;size_t pos=0;double t0=now();
    while(pos<total){size_t target=total-pos<chunk?total-pos:chunk, off=0;
        while(off<target){ssize_t w=write(fd,buf+off,target-off);calls++;
            if(w<0&&errno==EINTR)continue;
            if(w<=0){perror("write");free(buf);close(fd);unlink(path);return 3;}off+=(size_t)w;}
        pos+=target;
    }
    if(close(fd)){perror("close");free(buf);unlink(path);return 3;}
    double secs=now()-t0;
    fd=open(path,O_RDONLY);if(fd<0){free(buf);unlink(path);return 3;}
    uint64_t checksum=0;size_t read_bytes=0;int ok=1;ssize_t got;
    while((got=read(fd,buf,chunk))>0){for(ssize_t i=0;i<got;++i){if(buf[i]!='A')ok=0;checksum+=(unsigned char)buf[i];}read_bytes+=(size_t)got;}
    if(got<0)ok=0;
    close(fd);unlink(path);free(buf);
    if(!ok||read_bytes!=total||checksum!=65ULL*total){fprintf(stderr,"VALIDATION_FAIL io_bytes\n");return 3;}
    printf("IO mode=%s records=%d bytes=%zu chunk_records=%d write_calls=%d seconds=%.9f checksum=%llu\n",mode,records,total,chunk_records,calls,secs,(unsigned long long)checksum);
    printf("VALIDATION name=io_bytes result=PASS\n");return 0;
}
