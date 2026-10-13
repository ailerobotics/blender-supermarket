// SPDX-License-Identifier: Apache-2.0
#include "restock_mapper_evidence/producer_node.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <utility>

#include <builtin_interfaces/msg/time.hpp>
#include <nvblox/nvblox.h>
#include <rclcpp_components/register_node_macro.hpp>

namespace restock_mapper_evidence
{
namespace
{
constexpr char kLayerInventoryIdentity[] = "restock_mapper_evidence/static_multimapper/v1";

builtin_interfaces::msg::Time to_ros_time(int64_t ns)
{
  builtin_interfaces::msg::Time t;
  t.sec = static_cast<int32_t>(ns / 1000000000LL);
  t.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
  return t;
}
}  // namespace

EvidenceProducerNode::EvidenceProducerNode(const rclcpp::NodeOptions & options)
: nvblox::NvbloxNode(options, "nvblox_node")
{
  const std::string session = declare_parameter<std::string>("evidence.session", "");
  const std::string scene = declare_parameter<std::string>("evidence.scene", "");
  const std::string cameras = declare_parameter<std::string>("evidence.cameras", "");
  const std::string localization = declare_parameter<std::string>("evidence.localization", "");
  identity_ = Identity{session, scene, cameras, localization};

  // The published slice must carry the mapper's global frame: the stock nvblox
  // publish path sets slice.header.frame_id to global_frame, but the ESDF slice
  // converter leaves the header default, so without this the landed consumer
  // rejects the snapshot as an "invalid native distance slice" (empty frame_id).
  // global_frame is declared by the base NvbloxNode; read it, never redeclare it.
  global_frame_ = has_parameter("global_frame") ? get_parameter("global_frame").as_string()
                                                 : std::string("map");
  if (global_frame_.empty()) {global_frame_ = "map";}

  producer_nonce_ = declare_parameter<std::string>("evidence.producer_nonce", "");
  steady_domain_ = declare_parameter<std::string>("evidence.steady_clock_domain", "");
  slice_height_ = static_cast<float>(declare_parameter<double>("evidence.slice_height", 0.0));
  unknown_value_ = static_cast<float>(declare_parameter<double>("evidence.unknown_value", 1000.0));

  // "sensor=optical_frame" bindings, one required source per optical frame.
  const auto bindings =
    declare_parameter<std::vector<std::string>>("evidence.sensors", std::vector<std::string>{});
  for (const auto & binding : bindings) {
    const auto sep = binding.find('=');
    if (sep == std::string::npos) {
      throw std::runtime_error("evidence.sensors entry must be 'sensor=optical_frame': " + binding);
    }
    sensors_.emplace(binding.substr(0, sep), binding.substr(sep + 1));
  }

  // The nonce identifies this measured native instance for its whole lifetime.
  // Integration supplies a unique generation nonce; the steady-clock fallback
  // only keeps a standalone launch from colliding with itself.
  if (producer_nonce_.empty()) {
    producer_nonce_ = "native-" + std::to_string(steady_now_ns());
  }
  if (steady_domain_.empty()) {
    throw std::runtime_error("evidence.steady_clock_domain must name the shared CLOCK_MONOTONIC domain");
  }

  // State's constructor validates identity revisions and sensor/frame uniqueness;
  // a misconfigured generation refuses to construct rather than certifying later.
  policy_ = std::make_unique<State>(producer_nonce_, identity_, sensors_);

  // Record nvblox's post-construction allocation as the emptiness baseline, before
  // any sensor frame can arrive. A freshly constructed nvblox layer is non-empty
  // from construction (preallocated/derived blocks independent of sensor input), so
  // "empty" means no growth beyond this baseline, never zero allocated blocks.
  std::vector<uint64_t> baseline;
  for (const auto & lc : measure_layers()) {baseline.push_back(lc.allocated_blocks);}
  policy_->set_construction_baseline(std::move(baseline));

  snapshot_publisher_ = create_publisher<msg::EvidenceSnapshot>(
    "~/evidence_snapshot", rclcpp::QoS(1).reliable().transient_local());

  using std::placeholders::_1;
  using std::placeholders::_2;
  const auto qos = rclcpp::ServicesQoS();
  empty_service_ = create_service<srv::EmptyGeneration>(
    "~/empty_generation", std::bind(&EvidenceProducerNode::handle_empty, this, _1, _2),
    qos, group_processing_);
  open_service_ = create_service<srv::OpenAdmission>(
    "~/open_admission", std::bind(&EvidenceProducerNode::handle_open, this, _1, _2),
    qos, group_processing_);
  register_service_ = create_service<srv::RegisterAcquisition>(
    "~/register_acquisition", std::bind(&EvidenceProducerNode::handle_register, this, _1, _2),
    qos, group_processing_);
}

int64_t EvidenceProducerNode::sim_now_ns() const
{
  return this->get_clock()->now().nanoseconds();
}

int64_t EvidenceProducerNode::steady_now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

void EvidenceProducerNode::fill_identity(msg::MapIdentity & out) const
{
  out.session = identity_[0];
  out.scene = identity_[1];
  out.cameras = identity_[2];
  out.localization = identity_[3];
}

void EvidenceProducerNode::fill_ledger(std::vector<msg::Acquisition> & out) const
{
  out.clear();
  for (const auto & entry : policy_->ledger()) {
    const Acquisition & a = entry.second;
    msg::Acquisition m;
    m.identity.session = a.identity[0];
    m.identity.scene = a.identity[1];
    m.identity.cameras = a.identity[2];
    m.identity.localization = a.identity[3];
    m.sensor = a.sensor;
    m.optical_frame = a.frame;
    m.sequence = a.sequence;
    m.acquisition_stamp = to_ros_time(a.stamp);
    m.acquisition_steady_ns = static_cast<uint64_t>(a.steady);
    m.steady_clock_domain = steady_domain_;
    out.push_back(std::move(m));
  }
}

std::vector<msg::LayerCount> EvidenceProducerNode::measure_layers()
{
  std::vector<msg::LayerCount> out;
  auto add = [&out](const char * name, const char * type, uint64_t blocks) {
    msg::LayerCount lc;
    lc.layer_name = name;
    lc.layer_type = type;
    lc.allocated_blocks = blocks;
    out.push_back(std::move(lc));
  };
  // Every maintained layer, not the ESDF alone: an omitted layer is not "zero".
  add("tsdf", "TsdfLayer", static_mapper_->tsdf_layer().numAllocatedBlocks());
  add("esdf", "EsdfLayer", static_mapper_->esdf_layer().numAllocatedBlocks());
  add("color", "ColorLayer", static_mapper_->color_layer().numAllocatedBlocks());
  add("freespace", "FreespaceLayer", static_mapper_->freespace_layer().numAllocatedBlocks());
  add("occupancy", "OccupancyLayer", static_mapper_->occupancy_layer().numAllocatedBlocks());
  return out;
}

bool EvidenceProducerNode::queues_empty()
{
  auto empty = [](const auto & q) { return !q || q->empty(); };
  return empty(depth_image_queue_) && empty(color_image_queue_) &&
         empty(pointcloud_queue_) && empty(esdf_service_queue_) &&
         empty(file_path_service_queue_);
}

bool EvidenceProducerNode::watermarks_empty()
{
  return integrate_depth_last_times_.empty() && integrate_color_last_times_.empty() &&
         newest_integrated_depth_time_.nanoseconds() == 0;
}

bool EvidenceProducerNode::processDepthImage(const ImageTypeVariant & depth_mask_msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::string frame;
  int64_t stamp = 0;
  try {
    const auto decomposed = decomposeImageTypeVariant(depth_mask_msg);
    const auto & info = std::get<1>(decomposed);
    frame = info->header.frame_id;
    stamp = rclcpp::Time(info->header.stamp).nanoseconds();
  } catch (const std::exception & e) {
    RCLCPP_WARN(get_logger(), "evidence: undecodable depth frame: %s", e.what());
  }

  const bool tracking = policy_ && policy_->open() && !frame.empty();
  int64_t prior = 0;
  if (tracking) {
    const auto it = integrate_depth_last_times_.find(frame);
    if (it != integrate_depth_last_times_.end()) {
      prior = it->second.nanoseconds();
    }
  }

  // Run the real integration; a true return from the base is NOT incorporation.
  const bool ok = NvbloxNode::processDepthImage(depth_mask_msg);

  if (tracking) {
    try {
      const int64_t now = sim_now_ns();
      const int64_t steady = steady_now_ns();
      const Acquisition acq = policy_->match(frame, stamp, now, steady);
      const auto it = integrate_depth_last_times_.find(frame);
      const int64_t after = (it != integrate_depth_last_times_.end()) ? it->second.nanoseconds() : 0;
      // Record only when this exact frame's watermark advanced to this stamp.
      const bool integrated = ok && after == stamp && after > prior;
      policy_->finish(acq, integrated);
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "evidence depth not certified: %s", e.what());
    }
  }
  return ok;
}

void EvidenceProducerNode::processEsdf()
{
  // Real native ESDF update (and the base raw slice publication, which launch
  // remaps to a dead internal topic so only the envelope carries geometry).
  NvbloxNode::processEsdf();

  std::lock_guard<std::mutex> lock(mutex_);
  if (!policy_ || !policy_->open() || !snapshot_publisher_) {
    return;
  }
  try {
    const int64_t now = sim_now_ns();
    const int64_t steady = steady_now_ns();
    cuda_stream_->synchronize();

    const nvblox::EsdfLayer & layer = static_mapper_->esdf_layer();
    nvblox::AxisAlignedBoundingBox aabb;
    nvblox::Image<float> slice_image(nvblox::MemoryType::kDevice);
    esdf_slicer_.sliceLayerToDistanceImage(layer, slice_height_, unknown_value_, &aabb, &slice_image);
    cuda_stream_->synchronize();

    auto env = std::make_unique<msg::EvidenceSnapshot>();
    esdf_slice_converter_.distanceMapSliceMsgFromSliceImage(
      slice_image, aabb, slice_height_, static_mapper_->voxel_size_m(), unknown_value_, &env->slice);

    // The converter fills geometry but leaves the header default. Mirror the stock
    // nvblox publish contract the consumer validates: the slice frame is the mapper
    // global frame, and the stamp is the newest incorporated acquisition's sim time
    // (provenance-consistent; the publication stamp is not an incorporation ack).
    env->slice.header.frame_id = global_frame_;
    int64_t newest_stamp = 0;
    for (const auto & entry : policy_->ledger()) {
      newest_stamp = std::max(newest_stamp, entry.second.stamp);
    }
    env->slice.header.stamp = to_ros_time(newest_stamp);

    // snapshot() throws unless every configured sensor is incorporated and fresh
    // in both clocks; the copied slice above is frozen with this exact ledger.
    const uint64_t index = policy_->snapshot(now, steady);
    env->producer_nonce = producer_nonce_;
    env->certificate_token = issued_token_;
    fill_identity(env->identity);
    env->snapshot_index = index;
    env->esdf_revision = policy_->revision();
    fill_ledger(env->incorporated);
    env->completed_steady_ns = static_cast<uint64_t>(steady);
    env->steady_clock_domain = steady_domain_;
    env->admission_state = msg::EvidenceSnapshot::OPEN;
    env->empty_state = msg::EvidenceSnapshot::POPULATED;
    // Provenance only; full-height observed-column coverage is a separate gate.
    env->full_height_observation_verified = false;
    snapshot_publisher_->publish(std::move(env));
  } catch (const std::exception & e) {
    RCLCPP_WARN(get_logger(), "evidence snapshot withheld: %s", e.what());
  }
}

bool EvidenceProducerNode::identity_matches(const msg::MapIdentity & id) const
{
  return id.session == identity_[0] && id.scene == identity_[1] &&
         id.cameras == identity_[2] && id.localization == identity_[3];
}

void EvidenceProducerNode::handle_empty(
  const std::shared_ptr<srv::EmptyGeneration::Request> request,
  std::shared_ptr<srv::EmptyGeneration::Response> response)
{
  std::lock_guard<std::mutex> lock(mutex_);
  response->challenge = request->challenge;
  response->producer_nonce = producer_nonce_;
  fill_identity(response->identity);
  response->steady_clock_domain = steady_domain_;
  response->admission_open = false;
  response->success = false;
  // Report the measured per-layer allocation (and the inventory identity) on EVERY
  // outcome: a non-emptiness refusal must name the exact non-empty layer and its
  // count so the caller can diagnose the map, and the policy refuses that case
  // WITHOUT faulting the one-shot producer, so this probe never bricks the map.
  const auto layers = measure_layers();
  response->layers = layers;
  response->layer_inventory_identity = kLayerInventoryIdentity;
  try {
    if (!policy_) {throw std::runtime_error("producer not configured");}
    if (!identity_matches(request->expected_identity)) {
      throw std::runtime_error("expected identity mismatch");
    }
    std::vector<uint64_t> blocks;
    blocks.reserve(layers.size());
    for (const auto & lc : layers) {blocks.push_back(lc.allocated_blocks);}
    const bool q = queues_empty();
    const bool w = watermarks_empty();
    policy_->certify(request->challenge, identity_, blocks, q, w);  // refuses (no fault) unless empty vs baseline
    const int64_t steady = steady_now_ns();
    issued_token_ = request->challenge;
    response->certificate_token = issued_token_;
    response->ledger_count = 0;
    response->pending_sensor_work_count = 0;
    response->synchronized_steady_ns = static_cast<uint64_t>(steady);
    response->success = true;
  } catch (const std::exception & e) {
    response->failure_reason = e.what();
  }
}

void EvidenceProducerNode::handle_open(
  const std::shared_ptr<srv::OpenAdmission::Request> request,
  std::shared_ptr<srv::OpenAdmission::Response> response)
{
  std::lock_guard<std::mutex> lock(mutex_);
  response->producer_nonce = producer_nonce_;
  fill_identity(response->identity);
  response->certificate_token = request->certificate_token;
  response->steady_clock_domain = steady_domain_;
  response->acquisition_cutoff_stamp = request->acquisition_cutoff_stamp;
  response->acquisition_cutoff_steady_ns = request->acquisition_cutoff_steady_ns;
  response->admission_open = false;
  response->success = false;
  try {
    if (!policy_) {throw std::runtime_error("producer not configured");}
    if (request->producer_nonce != producer_nonce_) {throw std::runtime_error("producer nonce mismatch");}
    if (request->certificate_token != issued_token_) {throw std::runtime_error("certificate mismatch");}
    if (!identity_matches(request->identity)) {throw std::runtime_error("identity mismatch");}
    if (request->steady_clock_domain != steady_domain_) {throw std::runtime_error("steady domain mismatch");}
    // Required sensor/frame bindings must match exactly what this generation maintains.
    std::map<std::string, std::string> required;
    for (const auto & b : request->required_sensors) {required.emplace(b.sensor, b.optical_frame);}
    if (required != sensors_) {throw std::runtime_error("required sensor bindings mismatch");}
    const auto layers = measure_layers();
    std::vector<uint64_t> blocks;
    blocks.reserve(layers.size());
    for (const auto & lc : layers) {blocks.push_back(lc.allocated_blocks);}
    const int64_t cutoff_stamp = rclcpp::Time(request->acquisition_cutoff_stamp).nanoseconds();
    const int64_t cutoff_steady = static_cast<int64_t>(request->acquisition_cutoff_steady_ns);
    if (cutoff_stamp < 0 || cutoff_steady <= 0) {throw std::runtime_error("invalid acquisition cutoff");}
    // The cutoff becomes the admission boundary; registered acquisitions must be
    // strictly newer than it in both clocks (same-tick replay stays rejected).
    policy_->admit(request->producer_nonce, request->certificate_token, identity_,
      blocks, queues_empty(), watermarks_empty(), cutoff_stamp, cutoff_steady);
    response->opened_steady_ns = static_cast<uint64_t>(steady_now_ns());
    response->admission_open = true;
    response->success = true;
  } catch (const std::exception & e) {
    response->failure_reason = e.what();
  }
}

void EvidenceProducerNode::handle_register(
  const std::shared_ptr<srv::RegisterAcquisition::Request> request,
  std::shared_ptr<srv::RegisterAcquisition::Response> response)
{
  std::lock_guard<std::mutex> lock(mutex_);
  response->success = false;
  try {
    if (!policy_) {throw std::runtime_error("producer not configured");}
    if (request->producer_nonce != producer_nonce_) {throw std::runtime_error("producer nonce mismatch");}
    if (request->certificate_token != issued_token_) {throw std::runtime_error("certificate mismatch");}
    const auto & a = request->acquisition;
    if (a.steady_clock_domain != steady_domain_) {throw std::runtime_error("steady domain mismatch");}
    Acquisition acq;
    acq.identity = Identity{a.identity.session, a.identity.scene, a.identity.cameras, a.identity.localization};
    acq.sensor = a.sensor;
    acq.frame = a.optical_frame;
    acq.sequence = a.sequence;
    acq.stamp = rclcpp::Time(a.acquisition_stamp).nanoseconds();
    acq.steady = static_cast<int64_t>(a.acquisition_steady_ns);
    policy_->register_acquisition(request->producer_nonce, acq, sim_now_ns(), steady_now_ns());
    response->success = true;
  } catch (const std::exception & e) {
    response->failure_reason = e.what();
  }
}

}  // namespace restock_mapper_evidence

RCLCPP_COMPONENTS_REGISTER_NODE(restock_mapper_evidence::EvidenceProducerNode)
