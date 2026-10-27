/* Deliberate incompatible contract: Kelvin. Same symbol, different units. */
/* This is intentionally wrong for the caller: a loadable symbol can still use wrong units. */
double top_temperature(void){return 373.15;}
