#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace restock_mapper_evidence {
using Identity = std::array<std::string, 4>;
struct Acquisition {
  Identity identity;
  std::string sensor, frame;
  uint64_t sequence{};
  int64_t stamp{}, steady{};
};

// Pure policy: native code supplies measurements; this class cannot certify a map.
class State {
 public:
  enum class Phase { Closed, Certified, Open, Fault };
  State(std::string nonce, Identity identity, std::map<std::string, std::string> sensors)
  : nonce_(std::move(nonce)), identity_(std::move(identity)), sensors_(std::move(sensors)) {
    if (nonce_.empty() || sensors_.empty() || sensors_.size() > 4) fail("invalid generation/sensors");
    std::map<std::string, bool> frames;
    for (const auto& v : identity_) if (v.empty()) fail("empty identity revision");
    for (const auto& [sensor, frame] : sensors_) {
      if (sensor.empty() || frame.empty() || frames.count(frame)) fail("ambiguous optical frame");
      frames[frame] = true;
    }
  }
  Phase phase() const { return phase_; }
  const std::string& nonce() const { return nonce_; }
  const Identity& identity() const { return identity_; }
  const std::map<std::string, Acquisition>& ledger() const { return ledger_; }
  bool open() const { return phase_ == Phase::Open; }
  [[noreturn]] void fail(const std::string& reason) {
    phase_ = Phase::Fault;
    throw std::runtime_error(reason);
  }
  void certify(const std::string& challenge, const Identity& identity,
               const std::vector<uint64_t>& blocks, bool queues_empty, bool watermarks_empty) {
    if (phase_ != Phase::Closed || identity != identity_ || challenge.empty()) fail("invalid empty challenge");
    verify_empty(blocks, queues_empty, watermarks_empty);
    challenge_ = challenge;
    phase_ = Phase::Certified;
  }
  void admit(const std::string& nonce, const std::string& challenge, const Identity& identity,
             const std::vector<uint64_t>& blocks, bool queues_empty, bool watermarks_empty,
             int64_t now, int64_t steady) {
    if (phase_ != Phase::Certified || nonce != nonce_ || challenge != challenge_ || identity != identity_)
      fail("empty certificate mismatch");
    verify_empty(blocks, queues_empty, watermarks_empty);
    if (now < 0 || steady <= 0) fail("invalid opening clocks");
    opened_ = last_now_ = now;
    opened_steady_ = last_steady_ = steady;
    phase_ = Phase::Open;
  }
  void clocks(int64_t now, int64_t steady) {
    if (!open()) fail("admission closed");
    if (now < last_now_ || steady < last_steady_ || now-opened_ > epoch || steady-opened_steady_ > epoch)
      fail("clock regression or expired epoch");
    last_now_ = now; last_steady_ = steady;
  }
  void register_acquisition(const std::string& nonce, const Acquisition& a, int64_t now, int64_t steady) {
    clocks(now, steady);
    if (nonce != nonce_ || a.identity != identity_ || !sensors_.count(a.sensor) || sensors_.at(a.sensor) != a.frame)
      fail("acquisition identity mismatch");
    if (a.stamp <= opened_ || a.stamp > now || a.steady <= opened_steady_ || a.steady > steady ||
        now-a.stamp > freshness || steady-a.steady > freshness) fail("invalid acquisition clocks");
    auto old = registered_.find(a.sensor);
    if (old != registered_.end() && (a.sequence <= old->second.sequence || a.stamp <= old->second.stamp ||
        a.steady <= old->second.steady)) fail("duplicate/out-of-order acquisition");
    if (pending_.size() >= 64) fail("pending acquisition bound");
    const auto key = std::make_pair(a.frame, a.stamp);
    if (pending_.count(key)) fail("ambiguous acquisition key");
    pending_[key] = a;
    registered_[a.sensor] = a;
  }
  const Acquisition& match(const std::string& frame, int64_t stamp, int64_t now, int64_t steady) {
    clocks(now, steady);
    auto it = pending_.find({frame, stamp});
    if (it == pending_.end()) fail("native frame has no registered acquisition");
    if (stamp > now || now-stamp > freshness || it->second.steady > steady || steady-it->second.steady > freshness)
      fail("native acquisition expired");
    return it->second;
  }
  void finish(Acquisition a, bool integrated) {
    if (!open() || !pending_.erase({a.frame, a.stamp})) fail("unknown native completion");
    if (integrated) {
      auto it = ledger_.find(a.sensor);
      if (it != ledger_.end() && a.sequence <= it->second.sequence) fail("native completion reordered");
      ledger_[a.sensor] = a;
      ++revision_;
    }
  }
  uint64_t snapshot(int64_t now, int64_t steady) {
    clocks(now, steady);
    if (ledger_.size() != sensors_.size()) throw std::runtime_error("waiting for all sensors");
    for (const auto& [sensor, a] : ledger_) {
      (void)sensor;
      if (a.stamp > now || now-a.stamp > freshness || a.steady > steady || steady-a.steady > freshness)
        throw std::runtime_error("stale incorporated sensor");
    }
    if (!revision_) fail("no native integration");
    return ++snapshot_;
  }
  uint64_t revision() const { return revision_; }
 private:
  void verify_empty(const std::vector<uint64_t>& blocks, bool queues_empty, bool watermarks_empty) {
    if (blocks.empty() || !queues_empty || !watermarks_empty || !ledger_.empty() || !pending_.empty())
      fail("empty evidence incomplete");
    for (auto n : blocks) if (n) fail("native layer not empty");
  }
  static constexpr int64_t freshness = 300000000, epoch = 600000000000;
  std::string nonce_, challenge_;
  Identity identity_;
  std::map<std::string, std::string> sensors_;
  Phase phase_{Phase::Closed};
  std::map<std::string, Acquisition> registered_, ledger_;
  std::map<std::pair<std::string, int64_t>, Acquisition> pending_;
  int64_t opened_{}, opened_steady_{}, last_now_{}, last_steady_{};
  uint64_t revision_{}, snapshot_{};
};
}  // namespace restock_mapper_evidence
