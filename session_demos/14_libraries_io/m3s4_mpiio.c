/* Every rank owns a disjoint byte interval. Verify EVERY byte after reopen. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(call) do { int rc=(call); if(rc!=MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD,rc); } while(0)
int main(int argc, char **argv) {
    CHECK(MPI_Init(&argc,&argv));
    int rank,size; CHECK(MPI_Comm_rank(MPI_COMM_WORLD,&rank)); CHECK(MPI_Comm_size(MPI_COMM_WORLD,&size));
    int block=argc>1?atoi(argv[1]):1024;
    if(block<1 || block>1048576) MPI_Abort(MPI_COMM_WORLD,2);
    unsigned char *buf=malloc((size_t)block); if(!buf) MPI_Abort(MPI_COMM_WORLD,3);
    for(int j=0;j<block;j++) buf[j]=(unsigned char)((rank*17+j)%251);
    MPI_File fh; MPI_Status status;
    CHECK(MPI_File_open(MPI_COMM_WORLD,"mpi_output.bin",MPI_MODE_CREATE|MPI_MODE_RDWR,MPI_INFO_NULL,&fh));
    CHECK(MPI_File_set_size(fh,(MPI_Offset)size*block));
    CHECK(MPI_Barrier(MPI_COMM_WORLD)); double start=MPI_Wtime();
    /* Default file view has byte etypes, so this offset is in bytes. */
    CHECK(MPI_File_write_at_all(fh,(MPI_Offset)rank*block,buf,block,MPI_BYTE,&status));
    int count; CHECK(MPI_Get_count(&status,MPI_BYTE,&count)); if(count!=block) MPI_Abort(MPI_COMM_WORLD,4);
    CHECK(MPI_File_sync(fh)); CHECK(MPI_File_close(&fh));
    double elapsed=MPI_Wtime()-start,maximum; CHECK(MPI_Reduce(&elapsed,&maximum,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD));
    CHECK(MPI_File_open(MPI_COMM_WORLD,"mpi_output.bin",MPI_MODE_RDONLY,MPI_INFO_NULL,&fh));
    MPI_Offset bytes; CHECK(MPI_File_get_size(fh,&bytes));
    memset(buf,0,(size_t)block);
    CHECK(MPI_File_read_at_all(fh,(MPI_Offset)rank*block,buf,block,MPI_BYTE,&status));
    CHECK(MPI_Get_count(&status,MPI_BYTE,&count));
    int ok=(bytes==(MPI_Offset)size*block && count==block);
    for(int j=0;j<block;j++) if(buf[j]!=(unsigned char)((rank*17+j)%251)) ok=0;
    CHECK(MPI_File_close(&fh)); int all_ok; CHECK(MPI_Allreduce(&ok,&all_ok,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD));
    if(rank==0) printf("{\"passed\":%s,\"ranks\":%d,\"block\":%d,\"bytes\":%lld,\"max_write_sync_close_s\":%.9f}\n",all_ok?"true":"false",size,block,(long long)bytes,maximum);
    free(buf); CHECK(MPI_Finalize()); return all_ok?0:5;
}
