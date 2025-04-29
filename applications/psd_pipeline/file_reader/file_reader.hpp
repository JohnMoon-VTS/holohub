// SPDX-FileCopyrightText: 2025 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <fstream>
#include <chrono>
#include <cstdint>
#include <matx.h>
#include "holoscan/holoscan.hpp"
#include "vrt_time.hpp"

using namespace matx;

using complex = cuda::std::complex<float>;
using out_t = std::tuple<tensor_t<complex, 2>, cudaStream_t>;

namespace holoscan::ops {
class FileReader : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(FileReader)

    void initialize() override;
    void setup(OperatorSpec& spec) override;
    void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
    Parameter<std::vector<std::string>> file_names;
    Parameter<int> burst_size;
    Parameter<int> num_bursts;
    Parameter<uint16_t> num_channels;
    Parameter<uint32_t> stream_id;
    Parameter<std::vector<double>> rf_ref_freq_hz;
    Parameter<double> bandwidth_hz;
    Parameter<double> sample_rate_sps;
    Parameter<float> ref_level_dbm;
    Parameter<float> gain_db;
    Parameter<bool> float_input;

    struct Channel {
        size_t file_length;
        int num_chunks;
        int chunk_index;
        int channel_num;
        cudaStream_t stream;
        tensor_t<complex, 2> rf_data;
    };
    std::vector<std::shared_ptr<Channel>> channel_list;

    int next_channel = 0;
    bool change_indicator = true;
    VrtTime vrt_time;

    void load_int16_input();
    void load_float32_input();
    void open_file(const std::string &file_name,
                   std::ifstream &file_stream,
                   size_t &file_length,
                   int &num_chunks,
                   int sample_size);
    std::shared_ptr<Channel> create_channel(size_t file_length, int num_chunks);
};

}  // namespace holoscan::ops
