##  FIPS 204 Lattice Visualization

This project is an implementation of FIPS-204 with a high-dimensional lattice viewing tool. This was done as a dissertation for my final year of my BSc Computer Science degree.

The FIPS-204 standard can be found [here](https://csrc.nist.gov/pubs/fips/204/final). 

The major permutation of this implementation from the reference is in bitarrays; unless otherwise noted,
all bytes objects have been converted to bits, for simplicity. 

The Lattice visualization tool was developed with Dash, with Plotly for visualization tools.

Setup the environment beforehand with `pip install -r requirements.txt`. 
The project was developed with Python 3.13.2, and tested with 3.11 and 3.10. Versions <3 are guaranteed to not work.

Run the project with `python app.py`. The site will run locally, accessed at `http://127.0.0.1:8050/`.
