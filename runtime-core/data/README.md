Canonical distance fields used by visual metrics tests.

Files:
- mandelbrot_distance_1024.npy
- mandelbrot_distance_1024.json
- mandelbrot_distance_1024.bin

These are precomputed signed distance fields used for deterministic testing and runtime sampling. The `.bin` file is the runtime-embedded/binary form used for loading in the core; it must stay in sync with the `.npy` data and `.json` metadata. If you update these files, ensure the accompanying .json metadata is correct (xmin/xmax/ymin/ymax/res) and regenerate the `.bin` as needed.