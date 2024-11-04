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
 * @class SignalData
 */
class SignalData {
 public:
    /**
     * @brief SignalData constructor
     */
    SignalData();

    /**
     * @brief SignalData unpack constructor
     */
    explicit SignalData(std::span<const uint8_t> data);

    /**
     * @brief Destructor
     */
    ~SignalData();

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
    auto header() const -> const vrtgen::packing::DataHeader&;

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
     * @brief Returns a const reference to class_id
     * @return A const reference to class_id
     */
    [[nodiscard]]
    auto class_id() const -> const vrtgen::packing::ClassIdentifier&;

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
     * @brief Get a span to the underlying payload vector
     * @return A const pointer to the underlying payload vector
     */
    auto payload() const -> std::span<const uint8_t>;

    /**
     * @brief Set the underlying payload vector data
     * @param data Data to assign to the payload vector
     * @param len Length of @param data in bytes
     */
    auto payload(std::span<const uint8_t> data) -> void;

    /**
     * @brief Get the size of the underlying payload vector data
     * @return Number of bytes in the payload data
     */
    auto payload_size() const -> std::size_t;

    /**
     * @brief Returns a reference to trailer
     * @return A reference to trailer
     */
    [[nodiscard]]
    auto trailer() -> vrtgen::packing::Trailer&;

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

    std::string m_name{"SignalData"};
    vrtgen::packing::DataHeader m_header;
    vrtgen::packing::ClassIdentifier m_class_id;
    uint32_t m_stream_id{0};
    vrtgen::packing::Trailer m_trailer;
    std::vector<uint8_t> m_data;
    std::map<std::string, std::size_t> m_positions;
};  // end class SignalData

}  // end namespace vrt::packets