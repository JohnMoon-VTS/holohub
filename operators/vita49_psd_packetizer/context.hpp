/**
 * SPDX-FileCopyrightText: 2024 Geon Technologies, LLC
 * SPDX-License-Identifier: LGPL-3.0-only
 * Originally generated by vrtgen: https://github.com/Geontech/vrtgen
 */

#pragma once

#include <cstddef>
#include <cstring>
#include <map>
#include <optional>
#include <span>
#include <vector>
#include <vrtgen/vrtgen.hpp>

namespace vrt::packets {

/**
 * @class Context
 */
class Context {
 public:
    /**
     * @brief Context constructor
     */
    Context();

    /**
     * @brief Context unpack constructor
     */
    explicit Context(std::span<const uint8_t> data);

    /**
     * @brief Destructor
     */
    ~Context();

    /**
     * @brief Match the data span against known values for this packet
     * @retval nullopt if packet is a match, otherwise an error string is returned
     */
    static auto match(std::span<const uint8_t> data) -> std::optional<std::string>;

    /**
     * @brief Return the packet's name
     * @retval Name of the packet
     */
    auto name() const -> std::string;

    /**
     * @brief Returns a const reference to header
     * @return A const reference to header
     */
    [[nodiscard]]
    auto header() const -> const vrtgen::packing::ContextHeader&;

    /**
     * @brief Returns the value of packet_count
     * @return packet_count's value
     */
    auto packet_count() const -> uint8_t;

    /**
     * @brief Set the value of packet_count
     * @param value Value to assign to packet_count
     */
    auto packet_count(const uint8_t value) -> void;

    /**
     * @brief Returns the value of stream_id
     * @return stream_id's value
     */
    auto stream_id() const -> uint32_t;

    /**
     * @brief Set the value of stream_id
     * @param value Value to assign to stream_id
     */
    auto stream_id(const uint32_t value) -> void;

    /**
     * @brief Returns the value of integer_timestamp
     * @return integer_timestamp's value
     */
    auto integer_timestamp() const -> uint32_t;

    /**
     * @brief Set the value of integer_timestamp
     * @param value Value to assign to integer_timestamp
     */
    auto integer_timestamp(const uint32_t value) -> void;

    /**
     * @brief Returns the value of fractional_timestamp
     * @return fractional_timestamp's value
     */
    auto fractional_timestamp() const -> uint64_t;

    /**
     * @brief Set the value of fractional_timestamp
     * @param value Value to assign to fractional_timestamp
     */
    auto fractional_timestamp(const uint64_t value) -> void;

    /**
     * @brief Returns a const reference to cif_0
     * @return A const reference to cif_0
     */
    [[nodiscard]]
    auto cif_0() const -> const vrtgen::packing::CIF0&;

    /**
     * @brief Returns the value of change_indicator
     * @return change_indicator's value
     */
    auto change_indicator() const -> bool;

    /**
     * @brief Set the value of change_indicator
     * @param value Value to assign to change_indicator
     */
    auto change_indicator(const bool value) -> void;

    /**
     * @brief Returns the value of bandwidth
     * @return bandwidth's value
     */
    auto bandwidth() const -> std::optional<long double>;

    /**
     * @brief Set the value of bandwidth
     * @param value Value to assign to bandwidth
     */
    auto bandwidth(const long double value) -> void;

    /**
     * @brief If bandwidth contains a value, destroy it
     */
    auto reset_bandwidth() -> void;

    /**
     * @brief Returns the value of rf_ref_frequency
     * @return rf_ref_frequency's value
     */
    auto rf_ref_frequency() const -> std::optional<long double>;

    /**
     * @brief Set the value of rf_ref_frequency
     * @param value Value to assign to rf_ref_frequency
     */
    auto rf_ref_frequency(const long double value) -> void;

    /**
     * @brief If rf_ref_frequency contains a value, destroy it
     */
    auto reset_rf_ref_frequency() -> void;

    /**
     * @brief Returns the value of reference_level
     * @return reference_level's value
     */
    auto reference_level() const -> std::optional<double>;

    /**
     * @brief Set the value of reference_level
     * @param value Value to assign to reference_level
     */
    auto reference_level(const double value) -> void;

    /**
     * @brief If reference_level contains a value, destroy it
     */
    auto reset_reference_level() -> void;

    /**
     * @brief Returns a reference to gain
     * @return A optional reference to gain
     */
    [[nodiscard]]
    auto gain() -> std::optional<vrtgen::packing::Gain>&;

    /**
     * @brief Set the value of gain
     * @param value Value to assign to gain
     */
    auto gain(const vrtgen::packing::Gain& value) -> void;

    /**
     * @brief If gain contains a value, destroy it
     */
    auto reset_gain() -> void;

    /**
     * @brief Returns the value of sample_rate
     * @return sample_rate's value
     */
    auto sample_rate() const -> std::optional<long double>;

    /**
     * @brief Set the value of sample_rate
     * @param value Value to assign to sample_rate
     */
    auto sample_rate(const long double value) -> void;

    /**
     * @brief If sample_rate contains a value, destroy it
     */
    auto reset_sample_rate() -> void;

    /**
     * @brief Returns a reference to device_id
     * @return A optional reference to device_id
     */
    [[nodiscard]]
    auto device_id() -> std::optional<vrtgen::packing::DeviceIdentifier>&;

    /**
     * @brief Set the value of device_id
     * @param value Value to assign to device_id
     */
    auto device_id(const vrtgen::packing::DeviceIdentifier& value) -> void;

    /**
     * @brief If device_id contains a value, destroy it
     */
    auto reset_device_id() -> void;

    /**
     * @brief Returns a reference to signal_data_format
     * @return A optional reference to signal_data_format
     */
    [[nodiscard]]
    auto signal_data_format() -> std::optional<vrtgen::packing::PayloadFormat>&;

    /**
     * @brief Set the value of signal_data_format
     * @param value Value to assign to signal_data_format
     */
    auto signal_data_format(const vrtgen::packing::PayloadFormat& value) -> void;

    /**
     * @brief If signal_data_format contains a value, destroy it
     */
    auto reset_signal_data_format() -> void;

    /**
     * @brief Returns a reference to formatted_gps
     * @return A optional reference to formatted_gps
     */
    [[nodiscard]]
    auto formatted_gps() -> std::optional<vrtgen::packing::Geolocation>&;

    /**
     * @brief Set the value of formatted_gps
     * @param value Value to assign to formatted_gps
     */
    auto formatted_gps(const vrtgen::packing::Geolocation& value) -> void;

    /**
     * @brief If formatted_gps contains a value, destroy it
     */
    auto reset_formatted_gps() -> void;

    /**
     * @brief Returns a const reference to cif_1
     * @return A const optional reference to cif_1
     */
    [[nodiscard]]
    auto cif_1() const -> const std::optional<vrtgen::packing::CIF1>&;

    /**
     * @brief If cif_1 contains a value, destroy it and all underlying fields
     */
    auto reset_cif_1() -> void;

    /**
     * @brief Returns a reference to spectrum
     * @return A optional reference to spectrum
     */
    [[nodiscard]]
    auto spectrum() -> std::optional<vrtgen::packing::Spectrum>&;

    /**
     * @brief Set the value of spectrum
     * @param value Value to assign to spectrum
     */
    auto spectrum(const vrtgen::packing::Spectrum& value) -> void;

    /**
     * @brief If spectrum contains a value, destroy it
     */
    auto reset_spectrum() -> void;

    /**
     * @brief Returns the value of health_status
     * @return health_status's value
     */
    auto health_status() const -> std::optional<uint16_t>;

    /**
     * @brief Set the value of health_status
     * @param value Value to assign to health_status
     */
    auto health_status(const uint16_t value) -> void;

    /**
     * @brief If health_status contains a value, destroy it
     */
    auto reset_health_status() -> void;

    /**
     * @brief Returns the value of v49_spec_compliance
     * @return v49_spec_compliance's value
     */
    auto v49_spec_compliance() const -> std::optional<vrtgen::packing::V49StandardCompliance>;

    /**
     * @brief Set the value of v49_spec_compliance
     * @param value Value to assign to v49_spec_compliance
     */
    auto v49_spec_compliance(const vrtgen::packing::V49StandardCompliance value) -> void;

    /**
     * @brief If v49_spec_compliance contains a value, destroy it
     */
    auto reset_v49_spec_compliance() -> void;

    /**
     * @brief Returns a reference to version_information
     * @return A optional reference to version_information
     */
    [[nodiscard]]
    auto version_information() -> std::optional<vrtgen::packing::VersionInformation>&;

    /**
     * @brief Set the value of version_information
     * @param value Value to assign to version_information
     */
    auto version_information(const vrtgen::packing::VersionInformation& value) -> void;

    /**
     * @brief If version_information contains a value, destroy it
     */
    auto reset_version_information() -> void;

    /**
     * @brief Returns a const reference to cif_2
     * @return A const optional reference to cif_2
     */
    [[nodiscard]]
    auto cif_2() const -> const std::optional<vrtgen::packing::CIF2>&;

    /**
     * @brief If cif_2 contains a value, destroy it and all underlying fields
     */
    auto reset_cif_2() -> void;

    /**
     * @brief Returns the value of platform_class
     * @return platform_class's value
     */
    auto platform_class() const -> std::optional<uint32_t>;

    /**
     * @brief Set the value of platform_class
     * @param value Value to assign to platform_class
     */
    auto platform_class(const uint32_t value) -> void;

    /**
     * @brief If platform_class contains a value, destroy it
     */
    auto reset_platform_class() -> void;

    /**
     * @brief Returns the value of platform_instance
     * @return platform_instance's value
     */
    auto platform_instance() const -> std::optional<uint32_t>;

    /**
     * @brief Set the value of platform_instance
     * @param value Value to assign to platform_instance
     */
    auto platform_instance(const uint32_t value) -> void;

    /**
     * @brief If platform_instance contains a value, destroy it
     */
    auto reset_platform_instance() -> void;

    /**
     * @brief Returns the value of platform_display
     * @return platform_display's value
     */
    auto platform_display() const -> std::optional<uint32_t>;

    /**
     * @brief Set the value of platform_display
     * @param value Value to assign to platform_display
     */
    auto platform_display(const uint32_t value) -> void;

    /**
     * @brief If platform_display contains a value, destroy it
     */
    auto reset_platform_display() -> void;

    /**
     * @brief Returns the value of function_id
     * @return function_id's value
     */
    auto function_id() const -> std::optional<uint32_t>;

    /**
     * @brief Set the value of function_id
     * @param value Value to assign to function_id
     */
    auto function_id(const uint32_t value) -> void;

    /**
     * @brief If function_id contains a value, destroy it
     */
    auto reset_function_id() -> void;

    /**
     * @brief Return a span of the underlying packed data
     * @retval Pointer to underlying packed data
     */
    auto data() -> std::span<const uint8_t>;

    /**
     * @brief Return the size of the packet in bytes
     * @retval Number of packet bytes
     */
    auto size() -> std::size_t;

 private:
    auto min_bytes() const -> std::size_t;
    auto update_packet_size() -> void;
    auto update_positions() -> void;
    auto sync() -> void;

    std::string m_name{"Context"};
    vrtgen::packing::ContextHeader m_header;
    uint32_t m_stream_id{0};
    vrtgen::packing::CIF0 m_cif_0;
    std::optional<vrtgen::packing::CIF1> m_cif_1;
    std::optional<vrtgen::packing::CIF2> m_cif_2;
    std::optional<vrtgen::packing::Gain> m_gain;
    std::optional<vrtgen::packing::DeviceIdentifier> m_device_id;
    std::optional<vrtgen::packing::PayloadFormat> m_signal_data_format;
    std::optional<vrtgen::packing::Geolocation> m_formatted_gps;
    std::optional<vrtgen::packing::Spectrum> m_spectrum;
    std::optional<vrtgen::packing::VersionInformation> m_version_information;
    std::vector<uint8_t> m_data;
    std::map<std::string, std::size_t> m_positions;
};  // end class Context

}  // end namespace vrt::packets