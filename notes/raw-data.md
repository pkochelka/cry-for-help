# Raw data notes

The non-BMP files in `TRAIN_SET/` are Bruker NanoScope AFM measurement files.

## What is inside one raw file

1. **Text header** at the start of the file
   - starts with `\*File list`
   - contains instrument metadata (`Description: Dimension Icon`, scanner type, date, scan settings)
   - contains one or more `\*Ciao image list` blocks

2. **Binary image blocks** later in the file
   - each image block is described by:
     - `Data offset`
     - `Data length`
     - `Bytes/pixel`
     - `Samps/line`
     - `Number of lines`
     - `Scan Size`
     - `@...:Image Data`
   - in this dataset, blocks are mostly `int16` rasters (`Bytes/pixel: 2`)

3. **Multiple channels per scan**
   - most files contain 4 channels:
     - `ZSensor` = height sensor / topography-related signal
     - `AmplitudeError` = tapping-mode error signal
     - `Phase` = phase contrast signal
     - `Height` = processed height map
   - some diabetes files are irregular and have `Height` twice while lacking `Phase`

## How to read the raster

For each channel:
- width = `Samps/line`
- height = `Number of lines`
- dtype is usually signed 16-bit because `Bytes/pixel = 2`
- binary payload lives at `Data offset : Data offset + Data length`

Typical example:
- `512 x 512 x 2 bytes = 524,288 bytes`
- `1024 x 1024 x 2 bytes = 2,097,152 bytes`

So the raw files are not plain images. They are containers with metadata + several embedded measurement matrices.

## Physical meaning

`Scan Size` gives the real-world field of view in micrometers, so each pixel corresponds to a real physical distance on the tear crystal sample.

Example:
- `Scan Size: 92.5168 92.5168 ~m`
- `Samps/line: 1024`
- lateral pixel size is about `92.5168 / 1024 = 0.0903 um` per pixel

`Z scale` lines describe how raw integer counts map to physical vertical units through the instrument calibration.

## BMP files

The `*_1.bmp` files are fixed-size previews (`704 x 575`) exported by the microscope software. They are useful for quick viewing, but they are not the original full-fidelity channel data.

## Dataset quirks

- `TRAIN_SET/` is about 4.1G and should stay out of git.
- There is one extra preview in `Diabetes` (`37_DM_.bmp`).
- Missing preview files were found for:
  - `PGOV_Glaukom/25_PV_PGOV.017`
  - `SucheOko/35_PM_suche_oko.026`
