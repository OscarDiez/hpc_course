/* Serial HDF5 weather field: dimensions, units and a selected time slice. */
#include <hdf5.h>
#include <math.h>
#include <stdio.h>
#define HANDLE(v) do { if((v)<0) return 2; } while(0)
#define CALL(c) do { if((c)<0) return 3; } while(0)
int main(void) {
    hsize_t dims[3]={4,8,8}; double field[4][8][8],readback[4][8][8],slice[8][8];
    for(int t=0;t<4;t++) for(int y=0;y<8;y++) for(int x=0;x<8;x++) field[t][y][x]=273.15+t+0.1*y+0.01*x;
    hid_t f=H5Fcreate("weather.h5",H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT); HANDLE(f);
    hid_t s=H5Screate_simple(3,dims,NULL); HANDLE(s);
    hid_t d=H5Dcreate2(f,"temperature",H5T_IEEE_F64LE,s,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT); HANDLE(d);
    CALL(H5Dwrite(d,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,field));
    hid_t a_s=H5Screate(H5S_SCALAR),type=H5Tcopy(H5T_C_S1); HANDLE(a_s); HANDLE(type); CALL(H5Tset_size(type,2));
    hid_t a=H5Acreate2(d,"units",type,a_s,H5P_DEFAULT,H5P_DEFAULT); HANDLE(a); CALL(H5Awrite(a,type,"K"));
    CALL(H5Aclose(a)); CALL(H5Tclose(type)); CALL(H5Sclose(a_s)); CALL(H5Dclose(d)); CALL(H5Sclose(s)); CALL(H5Fclose(f));
    f=H5Fopen("weather.h5",H5F_ACC_RDONLY,H5P_DEFAULT); HANDLE(f);
    d=H5Dopen2(f,"temperature",H5P_DEFAULT); HANDLE(d); s=H5Dget_space(d); HANDLE(s);
    hsize_t got[3]; if(H5Sget_simple_extent_ndims(s)!=3) return 4; CALL(H5Sget_simple_extent_dims(s,got,NULL));
    for(int j=0;j<3;j++) if(got[j]!=dims[j]) return 5;
    CALL(H5Dread(d,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,readback));
    for(int t=0;t<4;t++) for(int y=0;y<8;y++) for(int x=0;x<8;x++) if(fabs(readback[t][y][x]-field[t][y][x])>1e-12) return 6;
    a=H5Aopen(d,"units",H5P_DEFAULT); HANDLE(a); type=H5Aget_type(a); HANDLE(type);
    if(H5Tget_size(type)!=2) return 7;
    char units[2]; CALL(H5Aread(a,type,units)); if(units[0]!='K'||units[1]!=0) return 8;
    CALL(H5Aclose(a)); CALL(H5Tclose(type));
    hsize_t start[3]={2,0,0},count[3]={1,8,8},md[2]={8,8};
    CALL(H5Sselect_hyperslab(s,H5S_SELECT_SET,start,NULL,count,NULL));
    hid_t memory=H5Screate_simple(2,md,NULL); HANDLE(memory);
    CALL(H5Dread(d,H5T_NATIVE_DOUBLE,memory,s,H5P_DEFAULT,slice));
    for(int y=0;y<8;y++) for(int x=0;x<8;x++) if(fabs(slice[y][x]-field[2][y][x])>1e-12) return 9;
    CALL(H5Sclose(memory)); CALL(H5Sclose(s)); CALL(H5Dclose(d)); CALL(H5Fclose(f));
    puts("{\"passed\":true,\"shape\":[4,8,8],\"units\":\"K\",\"checked_values\":256,\"slice_time\":2,\"slice_first\":275.15}"); return 0;
}
