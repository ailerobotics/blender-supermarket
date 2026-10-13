// SPDX-License-Identifier: Apache-2.0
// Navigation-only evidence producer: a task-local subclass of the installed
// nvblox::NvbloxNode. It overrides the real virtual hooks to bind every
// published map slice to the exact incorporated-acquisition ledger that
// produced it. The pure policy (state.hpp) decides admission/identity; this
// adapter supplies the native measurements the policy cannot invent.
//
// NOTHING in this translation unit is instantiated by the CPU build: the
// NvbloxNode constructor may allocate GPU resources, so construction and
// spinning belong to the integration worker under its GPU guards. Building the
// component library only proves the overridden signatures and symbol/ABI
// closure against the installed library.
#ifndef RESTOCK_MAPPER_EVIDENCE__PRODUCER_NODE_HPP_
#define RESTOCK_MAPPER_EVIDENCE__PRODUCER_NODE_HPP_

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <nvblox_ros/nvblox_node.hpp>

#include "restock_mapper_evidence/state.hpp"
#include "restock_mapper_evidence/msg/evidence_snapshot.hpp"
#include "restock_mapper_evidence/srv/empty_generation.hpp"
#include "restock_mapper_evidence/srv/open_admission.hpp"
#include "restock_mapper_evidence/srv/register_acquisition.hpp"

namespace restock_mapper_evidence
{

/// Navigation evidence producer. One generation per native instance; a reset is
/// a new instance with a new measured certificate, never a relabeled map.
class EvidenceProducerNode : public nvblox::NvbloxNode
{
public:
  explicit EvidenceProducerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~EvidenceProducerNode() override = default;

protected:
  // Real installed virtual hooks. Both run inside the base node's mutually
  // exclusive processing callback group, so the frozen ledger and the copied
  // slice are assembled atomically with respect to each other and the services.
  bool processDepthImage(const ImageTypeVariant & depth_mask_msg) override;
  void processEsdf() override;

private:
  // Native measurement helpers (not caller assertions).
  std::vector<msg::LayerCount> measure_layers();
  bool queues_empty();
  bool watermarks_empty();
  int64_t sim_now_ns() const;
  static int64_t steady_now_ns();
  void fill_identity(msg::MapIdentity & out) const;
  void fill_ledger(std::vector<msg::Acquisition> & out) const;
  bool identity_matches(const msg::MapIdentity & id) const;

  // Service handlers, each serialized on the processing group.
  void handle_empty(
    const std::shared_ptr<srv::EmptyGeneration::Request> request,
    std::shared_ptr<srv::EmptyGeneration::Response> response);
  void handle_open(
    const std::shared_ptr<srv::OpenAdmission::Request> request,
    std::shared_ptr<srv::OpenAdmission::Response> response);
  void handle_register(
    const std::shared_ptr<srv::RegisterAcquisition::Request> request,
    std::shared_ptr<srv::RegisterAcquisition::Response> response);

  // Configuration frozen for this producer lifetime.
  Identity identity_;
  std::map<std::string, std::string> sensors_;
  std::string producer_nonce_;
  std::string steady_domain_;
  std::string issued_token_;
  float slice_height_{0.0f};
  float unknown_value_{1000.0f};

  // The pure admission/ledger policy. Protected by the processing group plus
  // this mutex, because converters and base hooks touch shared GPU state.
  std::mutex mutex_;
  std::unique_ptr<State> policy_;

  // Certified publishers/services. The base node's raw static_map_slice output
  // is remapped away at launch; only this envelope carries certified geometry.
  rclcpp::Publisher<msg::EvidenceSnapshot>::SharedPtr snapshot_publisher_;
  rclcpp::Service<srv::EmptyGeneration>::SharedPtr empty_service_;
  rclcpp::Service<srv::OpenAdmission>::SharedPtr open_service_;
  rclcpp::Service<srv::RegisterAcquisition>::SharedPtr register_service_;
};

}  // namespace restock_mapper_evidence

#endif  // RESTOCK_MAPPER_EVIDENCE__PRODUCER_NODE_HPP_
