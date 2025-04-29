// SPDX-FileCopyrightText: 2025 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "file_reader.hpp"

namespace holoscan::ops {

constexpr uint64_t PS_PER_NS = 1e3;

void FileReader::setup(OperatorSpec& spec) {
    spec.output<out_t>("out");
    spec.param(file_names,
               "file_names",
               "Data file"
               "Raw data files to send");
    spec.param(num_channels,
               "num_channels",
               "Number of channels",
               "Number of channels to allocate memory for");
    spec.param(burst_size,
               "burst_size",
               "Burst size"
               "Number of samples to send in each burst");
    spec.param(num_bursts,
               "num_bursts",
               "Number of bursts"
               "Number of sample bursts to process at once");
    spec.param(stream_id,
               "stream_id",
               "Stream ID",
               "VITA 49 stream ID to pass along in metadata");
    spec.param(rf_ref_freq_hz,
               "rf_ref_freq_hz",
               "Tune frequency for each channel"
               "Center tune frequency for each channel to pass along in metadata");
    spec.param(bandwidth_hz,
               "bandwidth_hz",
               "Bandwidth"
               "Bandwidth to pass along in metadata");
    spec.param(sample_rate_sps,
               "sample_rate_sps",
               "Sample rate"
               "Sample rate to derive burst rate and pass along in metadata");
    spec.param(ref_level_dbm,
               "ref_level_dbm",
               "Reference level"
               "Reference level to pass along in metadata");
    spec.param(gain_db,
               "gain_db",
               "Gain"
               "Gain to pass along in metadata");
    spec.param(float_input,
               "float_input",
               "Floating-point input"
               "True to read in 32-bit floats, false to read in 16-bit integers");
}

static std::complex<float> cast_to_complex_float(int16_t real, int16_t imag) {
    // Scale the int16 values to -1.0 thru +1.0 by dividing by 2^15 - 1 (0x7FFF)
    constexpr float scalar = 1.0 / 0x7FFF;
    return std::complex<float>(
        static_cast<float>(real) * scalar,
        static_cast<float>(imag) * scalar);
}

void FileReader::initialize() {
    holoscan::Operator::initialize();

    change_indicator = true;

    if (file_names.get().size() != num_channels.get()) {
        HOLOSCAN_LOG_WARN("Number of channels doesn't match number of supplied files!");
    }

    if (rf_ref_freq_hz.get().size() != num_channels.get()) {
        HOLOSCAN_LOG_WARN(
            "Number of channels doesn't match number of supplied center frequencies!");
    }

    if (float_input.get()) {
        load_float32_input();
    } else {
        load_int16_input();
    }
}

void FileReader::load_int16_input() {
    constexpr size_t SAMPLE_SIZE = sizeof(int16_t) * 2;
    const size_t burst_size_bytes = burst_size.get() * SAMPLE_SIZE;
    std::unique_ptr<int16_t[]> temp_int_array = std::make_unique<int16_t[]>(burst_size.get() * 2);
    if (!temp_int_array && burst_size_bytes) {
        throw std::bad_alloc();
    }

    auto h_rf_data = make_tensor<complex, 1>({burst_size.get()}, MATX_HOST_MEMORY);

    cudaExecutor exec;
    for (const auto &file_name : file_names.get()) {
        std::ifstream file;
        size_t file_length = 0;
        int num_chunks = 0;
        open_file(file_name, file, file_length, num_chunks, SAMPLE_SIZE);

        std::shared_ptr<Channel> file_channel = create_channel(file_length, num_chunks);

        for (int chunk = 0; chunk < num_chunks; chunk++) {
            file.read(reinterpret_cast<char*>(temp_int_array.get()), burst_size_bytes);

            if (file.eof() || file.fail()) {
                throw std::runtime_error("Failed to read file data into tensor");
            }

            for (int samp = 0; samp < burst_size.get(); samp++) {
                h_rf_data(samp) = cast_to_complex_float(
                    temp_int_array[samp * 2],
                    temp_int_array[(samp * 2) + 1]);
            }
            auto rf_slice = slice<1>(file_channel->rf_data, {chunk, 0}, {matxDropDim, matxEnd});
            matx::copy(rf_slice, h_rf_data, exec);
            exec.sync();
        }

        file.close();
    }
}

void FileReader::load_float32_input() {
    constexpr size_t SAMPLE_SIZE = sizeof(float) * 2;
    const size_t burst_size_bytes = burst_size.get() * SAMPLE_SIZE;
    std::unique_ptr<float[]> temp_float_array = std::make_unique<float[]>(burst_size.get() * 2);
    if (!temp_float_array && burst_size_bytes) {
        throw std::bad_alloc();
    }

    auto h_rf_data = make_tensor<complex, 1>({burst_size.get()}, MATX_HOST_MEMORY);

    cudaExecutor exec;
    for (const auto &file_name : file_names.get()) {
        std::ifstream file;
        size_t file_length = 0;
        int num_chunks = 0;
        open_file(file_name, file, file_length, num_chunks, SAMPLE_SIZE);

        std::shared_ptr<Channel> file_channel = create_channel(file_length, num_chunks);

        for (int chunk = 0; chunk < num_chunks; chunk++) {
            file.read(reinterpret_cast<char*>(temp_float_array.get()), burst_size_bytes);

            if (file.eof() || file.fail()) {
                throw std::runtime_error("Failed to read file data into tensor");
            }

            for (int samp = 0; samp < burst_size.get(); samp++) {
                h_rf_data(samp) = std::complex<float>(
                    temp_float_array[samp * 2],
                    temp_float_array[(samp * 2) + 1]);
            }

            auto rf_slice = slice<1>(file_channel->rf_data, {chunk, 0}, {matxDropDim, matxEnd});
            matx::copy(rf_slice, h_rf_data, exec);
            exec.sync();
        }

        file.close();
    }
}

void FileReader::open_file(const std::string &file_name,
                           std::ifstream &file_stream,
                           size_t &file_length,
                           int &num_chunks,
                           int sample_size) {
    file_stream.open(file_name, std::ios::binary);
    if (file_stream.fail()) {
        HOLOSCAN_LOG_ERROR("could not open file {}", file_name);
        exit(1);
    }

    file_stream.seekg(0, file_stream.end);
    file_length = file_stream.tellg();
    file_stream.seekg(0, file_stream.beg);

    num_chunks = file_length / (burst_size.get() * sample_size);

    if (num_chunks < num_bursts.get()) {
        const size_t min_length = burst_size.get() * sample_size * num_bursts.get();
        HOLOSCAN_LOG_ERROR("File not large enough: {}, but needed {}", file_length, min_length);
        file_stream.close();
        exit(1);
    }

    HOLOSCAN_LOG_INFO("File Info: Name: {}, Length: {}", file_name, file_length);
    HOLOSCAN_LOG_INFO("Sample Size: {}, Samples/Burst: {}, Bursts/Batch: {}, Bursts/File: {}",
        sample_size, burst_size.get(), num_bursts.get(), num_chunks);

    // Force num_chunks to be a multiple of num_bursts
    num_chunks = (num_chunks / num_bursts.get()) * num_bursts.get();

    HOLOSCAN_LOG_INFO("Loading {} sample bursts into memory from {}", num_chunks, file_name);
}

std::shared_ptr<FileReader::Channel> FileReader::create_channel(size_t file_length, int num_chunks) {
    std::shared_ptr<Channel> file_channel = std::make_shared<Channel>();
    file_channel->file_length = file_length;
    file_channel->num_chunks = num_chunks;
    file_channel->chunk_index = 0;
    file_channel->channel_num = static_cast<int>(channel_list.size());
    cudaStreamCreateWithFlags(&file_channel->stream, cudaStreamNonBlocking);

    make_tensor(file_channel->rf_data, {num_chunks, burst_size.get()}, MATX_DEVICE_MEMORY);

    channel_list.push_back(file_channel);

    return file_channel;
}

void FileReader::compute(InputContext&, OutputContext& op_output, ExecutionContext&) {
    auto meta = metadata();
    meta->set("integer_timestamp", vrt_time.intTime());
    meta->set("fractional_timestamp", vrt_time.fracTime() * PS_PER_NS);
    meta->set("stream_id", static_cast<uint32_t>(stream_id.get() + next_channel));
    meta->set("change_indicator", static_cast<bool>(change_indicator));
    meta->set("bandwidth_hz", static_cast<double>(bandwidth_hz.get()));
    meta->set("reference_level_dbm", static_cast<double>(ref_level_dbm.get()));
    meta->set("gain_stage_1_db", static_cast<double>(gain_db.get()));
    meta->set("gain_stage_2_db", static_cast<double>(gain_db.get()));
    meta->set("sample_rate_hz", static_cast<double>(sample_rate_sps.get()));
    meta->set("channel_number", static_cast<uint16_t>(next_channel));

    if (rf_ref_freq_hz.get().size() > 0) {
        const int freq_idx = next_channel % rf_ref_freq_hz.get().size();
        const double center_freq_hz = rf_ref_freq_hz.get()[freq_idx];
        meta->set("rf_ref_freq_hz", center_freq_hz);
    }

    if (channel_list.size() > 0) {
        auto channel = channel_list[next_channel % channel_list.size()];

        const int chunk_start = channel->chunk_index;
        const int chunk_end = chunk_start + num_bursts.get();
        op_output.emit(
            out_t{
                slice<2>(channel->rf_data, {chunk_start, 0}, {chunk_end, matxEnd}),
                channel->stream
            },
            "out");

        channel->chunk_index = chunk_end % channel->num_chunks;
    }

    next_channel = (next_channel + 1) % num_channels.get();
    if (next_channel == 0) {
        change_indicator = false;
    }

    const uint32_t sample_count = burst_size.get() * num_bursts.get();
    vrt_time.increment(sample_count, sample_rate_sps.get());
}
}  // namespace holoscan::ops
