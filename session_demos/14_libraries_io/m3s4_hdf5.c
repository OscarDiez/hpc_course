/* Weather field, editable dimensions, chunking/compression and hyperslab read.
 * A serial writer even when linked with a parallel HDF5 installation. */
#include <hdf5.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define HANDLE(v) do { if((v)<0) return 2; } while(0)
#define CALL(c) do { if((c)<0) return 3; } while(0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9; }
static int attribute(hid_t d,const char *name,const char *value) {
    hid_t s=H5Screate(H5S_SCALAR),t=H5Tcopy(H5T_C_S1);HANDLE(s);HANDLE(t);
    CALL(H5Tset_size(t,strlen(value)+1));
    hid_t a=H5Acreate2(d,name,t,s,H5P_DEFAULT,H5P_DEFAULT);HANDLE(a);CALL(H5Awrite(a,t,value));
    CALL(H5Aclose(a));CALL(H5Tclose(t));CALL(H5Sclose(s));return 0;
}
int main(int argc,char **argv) {
    if(argc!=6) return 1;
    int nt=atoi(argv[1]),ny=atoi(argv[2]),nx=atoi(argv[3]),selected=atoi(argv[4]),zip=atoi(argv[5]);
    if(nt<1||ny<1||nx<1||selected<0||selected>=nt||(zip!=0&&zip!=1)) return 1;
    size_t plane=(size_t)ny*nx,total=(size_t)nt*plane;
    if(total>2000000) return 1;
    double *field=malloc(total*sizeof(double)),*readback=malloc(total*sizeof(double)),*slice=malloc(plane*sizeof(double)),*times=malloc((size_t)nt*sizeof(double));
    if(!field||!readback||!slice||!times) return 1;
    for(int t=0;t<nt;t++) {times[t]=t;for(int y=0;y<ny;y++) for(int x=0;x<nx;x++) field[(size_t)t*plane+(size_t)y*nx+x]=273.15+t+0.1*y+0.01*x;}
    hsize_t dims[3]={(hsize_t)nt,(hsize_t)ny,(hsize_t)nx},chunk[3]={1,(hsize_t)ny,(hsize_t)nx};
    double start=now();
    hid_t f=H5Fcreate("weather.h5",H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);HANDLE(f);
    hid_t s=H5Screate_simple(3,dims,NULL),props=H5Pcreate(H5P_DATASET_CREATE);HANDLE(s);HANDLE(props);
    CALL(H5Pset_chunk(props,3,chunk));
    if(zip) CALL(H5Pset_deflate(props,4));
    hid_t d=H5Dcreate2(f,"temperature",H5T_IEEE_F64LE,s,H5P_DEFAULT,props,H5P_DEFAULT);HANDLE(d);
    CALL(H5Dwrite(d,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,field));
    if(attribute(d,"units","K")||attribute(d,"dimensions","time,y,x")) return 3;
    CALL(H5Dclose(d));CALL(H5Pclose(props));CALL(H5Sclose(s));
    hsize_t time_dim[1]={(hsize_t)nt};s=H5Screate_simple(1,time_dim,NULL);HANDLE(s);
    d=H5Dcreate2(f,"time",H5T_IEEE_F64LE,s,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);HANDLE(d);
    CALL(H5Dwrite(d,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,times));
    if(attribute(d,"units","hours since simulation start")) return 3;
    CALL(H5Dclose(d));CALL(H5Sclose(s));CALL(H5Fclose(f));double write_s=now()-start;
    f=H5Fopen("weather.h5",H5F_ACC_RDONLY,H5P_DEFAULT);HANDLE(f);
    d=H5Dopen2(f,"temperature",H5P_DEFAULT);HANDLE(d);s=H5Dget_space(d);HANDLE(s);
    hsize_t got[3];if(H5Sget_simple_extent_ndims(s)!=3) return 4;CALL(H5Sget_simple_extent_dims(s,got,NULL));
    for(int j=0;j<3;j++) if(got[j]!=dims[j]) return 5;
    CALL(H5Dread(d,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,readback));
    for(size_t i=0;i<total;i++) if(fabs(readback[i]-field[i])>1e-12) return 6;
    hid_t a=H5Aopen(d,"units",H5P_DEFAULT);HANDLE(a);hid_t type=H5Aget_type(a);HANDLE(type);
    if(H5Tget_size(type)!=2) return 7;
    char units[2];CALL(H5Aread(a,type,units));if(units[0]!='K'||units[1]!=0) return 8;
    CALL(H5Aclose(a));CALL(H5Tclose(type));
    hsize_t offset[3]={(hsize_t)selected,0,0},count[3]={1,(hsize_t)ny,(hsize_t)nx},md[2]={(hsize_t)ny,(hsize_t)nx};
    CALL(H5Sselect_hyperslab(s,H5S_SELECT_SET,offset,NULL,count,NULL));
    hid_t memory=H5Screate_simple(2,md,NULL);HANDLE(memory);start=now();
    CALL(H5Dread(d,H5T_NATIVE_DOUBLE,memory,s,H5P_DEFAULT,slice));double read_s=now()-start;
    for(size_t i=0;i<plane;i++) if(fabs(slice[i]-field[(size_t)selected*plane+i])>1e-12) return 9;
    CALL(H5Sclose(memory));CALL(H5Sclose(s));CALL(H5Dclose(d));
    d=H5Dopen2(f,"time",H5P_DEFAULT);HANDLE(d);CALL(H5Dread(d,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,times));
    for(int t=0;t<nt;t++) if(times[t]!=t) return 10;
    CALL(H5Dclose(d));CALL(H5Fclose(f));
    unsigned major,minor,release;CALL(H5get_libversion(&major,&minor,&release));
    printf("{\"passed\":true,\"shape\":[%d,%d,%d],\"chunks\":[1,%d,%d],\"units\":\"K\",\"checked_values\":%zu,\"slice_time\":%d,\"compression\":%s,\"write_close_s\":%.9g,\"read_slice_s\":%.9g,\"hdf5\":\"%u.%u.%u\"}\n",nt,ny,nx,ny,nx,total,selected,zip?"\"gzip\"":"null",write_s,read_s,major,minor,release);
    free(field);free(readback);free(slice);free(times);return 0;
}
