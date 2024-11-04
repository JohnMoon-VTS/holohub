/**
 * SPDX-FileCopyrightText: 2024 Geon Technologies, LLC
 * SPDX-License-Identifier: LGPL-3.0-only
 * Originally generated by vrtgen: https://github.com/Geontech/vrtgen
 */

#include "signal_data.hpp"

namespace vrt::packets {

/**
 * SignalData class functions
 */
SignalData::SignalData() {
    m_data.resize(min_bytes());
    auto curr_pos = std::size_t{};
    m_header.packet_type(vrtgen::packing::PacketType::SIGNAL_DATA_STREAM_ID);
    m_header.class_id_enable(true);
    m_header.tsi(vrtgen::packing::TSI::UTC);
    m_header.tsf(vrtgen::packing::TSF::REAL_TIME);
    m_header.trailer_included(true);
    m_header.spectrum_or_time(vrtgen::packing::SPECTRUM_OR_TIME::SPECTRUM);
    m_header.pack_into(m_data.data());
    curr_pos += m_header.size();
    m_positions["stream_id"] = curr_pos;
    curr_pos += sizeof(uint32_t);  // stream_id
    m_positions["class_id"] = curr_pos;
    m_class_id.oui(0xFF5654);
    m_class_id.packet_code(0x1);
    m_class_id.pack_into(m_data.data() + m_positions.at("class_id"));
    curr_pos += m_class_id.size();
    m_positions["integer_timestamp"] = curr_pos;
    curr_pos += sizeof(uint32_t);  // integer_timestamp
    m_positions["fractional_timestamp"] = curr_pos;
    curr_pos += sizeof(uint64_t);  // fractional_timestamp
    m_positions["payload"] = curr_pos;
    m_positions["trailer"] = curr_pos;
    update_packet_size();
}

SignalData::SignalData(std::span<const uint8_t> data) {
    m_header.unpack_from(data.data());
    m_data.resize(m_header.packet_size() * sizeof(uint32_t));
    std::memcpy(m_data.data(), data.data(), m_data.size());
    auto curr_pos = m_header.size();
    m_positions["stream_id"] = curr_pos;
    curr_pos += sizeof(uint32_t);  // stream_id
    m_positions["class_id"] = curr_pos;
    m_class_id.unpack_from(m_data.data() + m_positions.at("class_id"));
    curr_pos += m_class_id.size();
    m_positions["integer_timestamp"] = curr_pos;
    curr_pos += sizeof(uint32_t);  // integer_timestamp
    m_positions["fractional_timestamp"] = curr_pos;
    curr_pos += sizeof(uint64_t);  // fractional_timestamp
    m_positions["payload"] = curr_pos;
    m_positions["trailer"] = m_data.size() - m_trailer.size();
    m_trailer.unpack_from(m_data.data() + m_positions.at("trailer"));
}

SignalData::~SignalData() {}

auto SignalData::name() const -> std::string {
    return m_name;
}

auto SignalData::match(std::span<const uint8_t> data) -> std::optional<std::string> {
    auto header = vrtgen::packing::DataHeader{};
    header.unpack_from(data.data());
    [[maybe_unused]]
    auto curr_pos = header.size();
    if (header.packet_type() != vrtgen::packing::PacketType::SIGNAL_DATA_STREAM_ID) {
        std::ostringstream msg;
        msg << "Failed to match header field packet_type." << " Expected "
            << std::to_string(static_cast<int>(vrtgen::packing::PacketType::SIGNAL_DATA_STREAM_ID))
            << " but got " << std::to_string(static_cast<int>(header.packet_type()));
        return {msg.str()};
    }
    if (header.class_id_enable() != true) {
        std::ostringstream msg;
        msg << "Failed to match header field class_id_enable." << " Expected "
            << std::to_string(true) << " but got " << std::to_string(header.class_id_enable());
        return {msg.str()};
    }
    if (header.tsi() != vrtgen::packing::TSI::UTC) {
        std::ostringstream msg;
        msg << "Failed to match header field tsi." << " Expected "
            << std::to_string(static_cast<int>(vrtgen::packing::TSI::UTC)) << " but got "
            << std::to_string(static_cast<int>(header.tsi()));
        return {msg.str()};
    }
    if (header.tsf() != vrtgen::packing::TSF::REAL_TIME) {
        std::ostringstream msg;
        msg << "Failed to match header field tsf." << " Expected "
            << std::to_string(static_cast<int>(vrtgen::packing::TSF::REAL_TIME)) << " but got "
            << std::to_string(static_cast<int>(header.tsf()));
        return {msg.str()};
    }
    if (header.trailer_included() != true) {
        std::ostringstream msg;
        msg << "Failed to match header field trailer_included." << " Expected "
            << std::to_string(true) << " but got " << std::to_string(header.trailer_included());
        return {msg.str()};
    }
    if (header.not_v49d0() != false) {
        std::ostringstream msg;
        msg << "Failed to match header field not_v49d0." << " Expected " << std::to_string(false)
            << " but got " << std::to_string(header.not_v49d0());
        return {msg.str()};
    }
    if (header.spectrum_or_time() != vrtgen::packing::SPECTRUM_OR_TIME::SPECTRUM) {
        std::ostringstream msg;
        msg << "Failed to match header field spectrum_or_time." << " Expected "
            << std::to_string(static_cast<int>(vrtgen::packing::SPECTRUM_OR_TIME::SPECTRUM))
            << " but got " << std::to_string(static_cast<int>(header.spectrum_or_time()));
        return {msg.str()};
    }
    curr_pos += sizeof(uint32_t);  // stream_id
    auto class_id = vrtgen::packing::ClassIdentifier{};
    class_id.unpack_from(data.data() + curr_pos);
    if (class_id.pad_bits() != 0) {
        std::ostringstream msg;
        msg << "Failed to match class_id field pad_bits." << " Expected " << std::to_string(0)
            << " but got " << std::to_string(class_id.pad_bits());
        return {msg.str()};
    }
    if (class_id.oui() != 0xFF5654) {
        std::ostringstream msg;
        msg << "Failed to match class_id field oui." << " Expected " << std::to_string(0xFF5654)
            << " but got " << std::to_string(class_id.oui());
        return {msg.str()};
    }
    if (class_id.information_code() != 0) {
        std::ostringstream msg;
        msg << "Failed to match class_id field information_code." << " Expected "
            << std::to_string(0) << " but got " << std::to_string(class_id.information_code());
        return {msg.str()};
    }
    if (class_id.packet_code() != 0x1) {
        std::ostringstream msg;
        msg << "Failed to match class_id field packet_code." << " Expected " << std::to_string(0x1)
            << " but got " << std::to_string(class_id.packet_code());
        return {msg.str()};
    }
    curr_pos += class_id.size();
    return std::nullopt;
}

auto SignalData::header() const -> const vrtgen::packing::DataHeader& {
    return m_header;
}

auto SignalData::packet_count() const -> uint8_t {
    return m_header.packet_count();
}

auto SignalData::packet_count(const uint8_t value) -> void {
    m_header.packet_count(value);
    m_header.pack_into(m_data.data());
}

auto SignalData::stream_id() const -> uint32_t {
    const auto pos{m_positions.at("stream_id")};
    uint32_t retval{};
    std::memcpy(&retval, m_data.data() + pos, sizeof(uint32_t));
    return vrtgen::swap::from_be(retval);
}

auto SignalData::stream_id(const uint32_t value) -> void {
    auto swapped{vrtgen::swap::to_be(value)};
    const auto pos{m_positions.at("stream_id")};
    std::memcpy(m_data.data() + pos, &swapped, sizeof(swapped));
}

auto SignalData::class_id() const -> const vrtgen::packing::ClassIdentifier& {
    return m_class_id;
}

auto SignalData::integer_timestamp() const -> uint32_t {
    const auto pos{m_positions.at("integer_timestamp")};
    uint32_t retval{};
    std::memcpy(&retval, m_data.data() + pos, sizeof(uint32_t));
    return vrtgen::swap::from_be(retval);
}

auto SignalData::integer_timestamp(const uint32_t value) -> void {
    auto swapped{vrtgen::swap::to_be(value)};
    const auto pos{m_positions.at("integer_timestamp")};
    std::memcpy(m_data.data() + pos, &swapped, sizeof(swapped));
}

auto SignalData::fractional_timestamp() const -> uint64_t {
    const auto pos{m_positions.at("fractional_timestamp")};
    uint64_t retval{};
    std::memcpy(&retval, m_data.data() + pos, sizeof(uint64_t));
    return vrtgen::swap::from_be(retval);
}

auto SignalData::fractional_timestamp(const uint64_t value) -> void {
    auto swapped{vrtgen::swap::to_be(value)};
    const auto pos{m_positions.at("fractional_timestamp")};
    std::memcpy(m_data.data() + pos, &swapped, sizeof(swapped));
}

auto SignalData::payload() const -> std::span<const uint8_t> {
    return {m_data.data() + m_positions.at("payload"), payload_size()};
}

auto SignalData::payload(std::span<const uint8_t> data) -> void {
    const auto pos{m_positions.at("payload")};
    auto mod = (data.size() % sizeof(uint32_t)) != 0
                   ? sizeof(uint32_t) - (data.size() % sizeof(uint32_t))
                   : 0;
    m_data.resize(pos + data.size_bytes() + mod + m_trailer.size(), 0);
    if (!data.empty()) {
        std::memcpy(m_data.data() + pos, data.data(), data.size_bytes());
    }
    m_positions.at("trailer") = m_data.size() - m_trailer.size();
    m_trailer.pack_into(m_data.data() + m_positions.at("trailer"));
    update_packet_size();
}

auto SignalData::payload_size() const -> std::size_t {
    return m_positions.at("trailer") - m_positions.at("payload");
}

auto SignalData::trailer() -> vrtgen::packing::Trailer& {
    return m_trailer;
}

auto SignalData::data() -> std::span<const uint8_t> {
    sync();
    return {m_data.data(), m_data.size()};
}

auto SignalData::size() -> std::size_t {
    return data().size();
}

auto SignalData::update_packet_size() -> void {
    m_header.packet_size(static_cast<uint16_t>((m_data.size() + 3) / 4));
    m_header.pack_into(m_data.data());
}

auto SignalData::update_positions() -> void {}

auto SignalData::min_bytes() const -> std::size_t {
    std::size_t bytes{};
    bytes += m_header.size();
    bytes += sizeof(uint32_t);  // stream_id
    bytes += m_class_id.size();
    bytes += sizeof(uint32_t);  // integer_timestamp
    bytes += sizeof(uint64_t);  // fractional_timestamp
    bytes += m_trailer.size();
    return bytes;
}

auto SignalData::sync() -> void {
    m_trailer.pack_into(m_data.data() + m_positions.at("trailer"));
}

}  // end namespace vrt::packets