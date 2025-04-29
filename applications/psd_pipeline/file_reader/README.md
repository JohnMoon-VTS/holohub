<!--
SPDX-FileCopyrightText: 2025 Valley Tech Systems, Inc.

SPDX-License-Identifier: Apache-2.0
-->
# File Reader Operator

## Overview

The file reader operator is a simple IQ data source for downstream signal
processing applications.

## Description

During initialization, the operator loads raw IQ data from user-provided
files into a MatX tensor. Then, it uses a user-provided sample rate to
send bursts of data on its output.

This operator will loop over the file data continuously.

### Data Format

Currently, the file reader supports ingesting either complex, signed, 16-bit
integer data (i.e. 16-bit I, 16-bit Q) or complex, 32-bit floating-point data
(i.e. 32-bit I, 32-bit Q). Integer data is cast (with scaling) to 32-bit float
values between -1.0 and +1.0 for more precise downstream computations. 
Floating-point values are left as they are. Samples are output as a
2-dimensional tensor with configurable shape.

## Requirements

- [MatX](https://github.com/NVIDIA/MatX) (dependency)

## Example Usage

To make the file reader send at the configured sample rate, use Holoscan's
`PeriodicCondition` while creating the operator:

```cpp
auto file_reader_rate_hz = std::to_string(
    from_config("file_reader.sample_rate_sps").as<uint64_t>()
    / from_config("file_reader.burst_size").as<uint64_t>())
    + std::string("Hz");

auto fileReaderOp = make_operator<ops::FileReader>(
    "fileReader",
    from_config("file_reader"),
    make_condition<PeriodicCondition>("periodic-condition",
                                      Arg("recess_period") = file_reader_rate_hz));
```

For an full example, see the
[`psd_pipeline`](../../applications/psd_pipeline) application.

## Configuration

The file reader takes in a few parameters that determine how data is
sent and what metadata to pass along:

```yaml
file_reader:
  file_names: [
    "IQ_Data_8MSPS_Ch1.dat",
    "IQ_Data_8MSPS_Ch2.dat",
    "IQ_Data_8MSPS_Ch3.dat",
    "IQ_Data_8MSPS_Ch4.dat",
  ]
  num_channels: 4
  burst_size: 1280
  num_bursts: 625
  sample_rate_sps: 8000000
  stream_id: 1
  rf_ref_freq_hz: [
    100000000,
    500000000,
    750000000,
    900000000,
  ]
  bandwidth_hz: 6000000
  ref_level_dbm: 0
  gain_db: -10
  float_input: false
```

- `file_names`: List of names/paths of the files from which to ingest data
- `num_channels`: Number of channels to output
                  (ideally the same as the number of specified files,
                  but files will be repeated if there are excess channels
                  or ignored if there are excess files)
- `burst_size`: Number of samples to send per burst in one invocation of `compute()`
- `num_bursts`: Number of bursts to to send in one invocation of `compute()`
- `sample_rate_sps`: Rate at which the samples should be emitted
                     (also passed along in metadata)
- `stream_id`: Stream ID to pass along in the metadata
- `rf_ref_freq_hz`: Tune frequency for each channel to pass along in the metadata
- `bandwidth_hz`: Bandwidth to pass along in the metadata
- `ref_level_dbm`: Reference level to pass along in the metadata
- `gain_db`: Gain to pass along in the metadata
- `float_input`: True to parse files as 32-bit floats, false to parse files as 16-bit integers.

**Note:** Input files are divided into chunks where each chunk is the number of
samples sent per invocation of `compute()`, i.e. `burst_size * num_bursts` samples.
If the file does not divide evenly into chunks, then the ingested data will be
truncated to the smallest multiple of the chunk size when the file is loaded. If
any input file is smaller than one chunk in size, an error will occur and the
program will immediately terminate.
