# RRL contamination to 21cm Emission
Contains module for calculating the contamination by Radio Recombination Lines to 21cm emission. 

Running the line_contamination function, with given parameters will analytically determine the total RRl contamination for the generated HII region. 
Contamination can be calculated for 
1. the full RRL emission,
2. in a specific frequency band (nu_band_min and nu_band_max) by declaring the boolean **plot_freqContam**, or
3. in an atomic transition band (n_band_min and n_band_max) by declaring the boolean **plot_nContam**.

These two booleans will also plot the contamination of the respective bands on the 21cm signal.
