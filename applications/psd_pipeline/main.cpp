// SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "advanced_network_connectors/vita49_rx.h"
#include <file_reader.hpp>
#include <fft.hpp>
#include <high_rate_psd.hpp>
#include <low_rate_psd.hpp>
#include <vita49_psd_packetizer.hpp>
#include <data_writer.hpp>

// #define WRITE_DATA

class PsdPipeline : public holoscan::Application {
 public:
    void compose() override {
        using namespace holoscan;

        const bool use_file_reader_input = from_config("use_file_reader_input").as<bool>();

        auto fftOp = make_operator<ops::FFT>(
            "fftOp",
            from_config("fft"));

        auto highRatePsdOp = make_operator<ops::HighRatePSD>(
            "highRatePsdOp",
            from_config("high_rate_psd"));

        auto lowRatePsdOp = make_operator<ops::LowRatePSD>(
            "lowRatePsdOp",
            from_config("low_rate_psd"));

        auto packetizerOp = make_operator<ops::V49PsdPacketizer>(
            "packetizerOp",
            from_config("vita49_psd_packetizer"),
            make_condition<CountCondition>(from_config("num_psds").as<int64_t>()));

        add_operator(fftOp);
        add_operator(highRatePsdOp);
        add_operator(lowRatePsdOp);
        add_operator(packetizerOp);
        add_flow(fftOp, highRatePsdOp);
        add_flow(highRatePsdOp, lowRatePsdOp);
        add_flow(lowRatePsdOp, packetizerOp);

        std::shared_ptr<Operator> inputOp;
        if (use_file_reader_input) {
            const auto sample_rate = from_config("file_reader.sample_rate_sps").as<uint64_t>();
            const auto burst_size = from_config("file_reader.burst_size").as<uint64_t>();
            const auto num_bursts = from_config("file_reader.num_bursts").as<uint64_t>();
            const auto num_channels = from_config("file_reader.num_channels").as<uint64_t>();
            const auto read_rate = sample_rate * num_channels / (burst_size * num_bursts);

            auto file_reader_rate_hz = std::to_string(read_rate) + std::string("Hz");
            HOLOSCAN_LOG_INFO("File reader rate: {}", file_reader_rate_hz);

            auto fileReaderOp = make_operator<ops::FileReader>(
                "fileReader",
                from_config("file_reader"),
                make_condition<PeriodicCondition>("periodic-condition",
                                                    Arg("recess_period") = file_reader_rate_hz));

            add_operator(fileReaderOp);
            inputOp = fileReaderOp;
        } else {
            auto adv_net_config = from_config("advanced_network").as<NetworkConfig>();
            if (adv_net_init(adv_net_config) != Status::SUCCESS) {
                HOLOSCAN_LOG_ERROR("Failed to configure the Advanced Network manager");
                exit(1);
            }
            HOLOSCAN_LOG_INFO("Configured the Advanced Network manager");

            auto vitaConnectorOp = make_operator<ops::Vita49ConnectorOpRx>(
                "vitaConnectorOp",
                from_config("vita_connector"));
            add_operator(vitaConnectorOp);
            inputOp = vitaConnectorOp;
        }

        add_flow(inputOp, fftOp);

#ifdef WRITE_DATA
        auto dataWriterOp = make_operator<ops::DataWriter>(
            "dataWriterOp",
            from_config("data_writer"),
            make_condition<CountCondition>(2));
        add_operator(dataWriterOp);
        add_flow(vitaConnectorOp, dataWriterOp);
#endif
    }
};

int main(int argc, char** argv) {
    holoscan::set_log_pattern("FULL");
    auto app = holoscan::make_application<PsdPipeline>();

    // Get the configuration file
    if (argc < 1) {
        HOLOSCAN_LOG_ERROR("Usage: {} [config.yaml]", argv[0]);
        return -1;
    }

    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/" + std::string(argv[1]);

    // Check if the file exists
    if (!std::filesystem::exists(config_path)) {
        HOLOSCAN_LOG_ERROR("Configuration file '{}' does not exist",
                static_cast<std::string>(config_path));
        return -1;
    }

    // Run
    app->enable_metadata(true);
    app->config(config_path);
    app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
          "event-based-scheduler", app->from_config("scheduler")));
    app->run();

    shutdown();
    return 0;
}
