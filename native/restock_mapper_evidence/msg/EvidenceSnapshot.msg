# No producer exists yet. These fields must be populated from synchronized native work.
uint8 CLOSED=0
uint8 OPEN=1
uint8 FAULT=2
uint8 UNKNOWN=0
uint8 VERIFIED_EMPTY=1
uint8 POPULATED=2
nvblox_msgs/DistanceMapSlice slice
string producer_nonce
string certificate_token
restock_mapper_evidence/MapIdentity identity
uint64 snapshot_index
uint64 esdf_revision
restock_mapper_evidence/Acquisition[] incorporated
uint64 completed_steady_ns
string steady_clock_domain
uint8 admission_state
uint8 empty_state
# Provenance evidence cannot establish full-height observed-column coverage.
bool full_height_observation_verified
