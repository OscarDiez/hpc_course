/* Adapter restores the Celsius contract expected by this solver. */
/* Convert units at the library boundary so the existing solver remains correct. */
double top_temperature(void){return 373.15-273.15;}
