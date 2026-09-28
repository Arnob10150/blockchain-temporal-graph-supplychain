// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title DataProvenance
/// @notice Off-chain data, on-chain hash provenance for supply-chain snapshots.
///         Only the SHA-256 digest of each daily snapshot is stored on-chain;
///         the raw records remain off-chain. A snapshot is identified by its
///         date string (e.g. "2023-11-20"). Hashes are write-once to make the
///         registered record immutable.
contract DataProvenance {
    struct Snapshot {
        bytes32 hash;      // SHA-256 digest of the canonical snapshot
        uint256 timestamp; // block time at registration
        bool exists;
    }

    mapping(bytes32 => Snapshot) private snapshots; // keccak256(date) => Snapshot
    uint256 public count;

    event SnapshotRegistered(string date, bytes32 hash, uint256 timestamp);

    function _key(string memory date) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(date));
    }

    /// @notice Register the hash of a snapshot. Reverts if already registered.
    function registerSnapshot(string calldata date, bytes32 hash) external {
        bytes32 k = _key(date);
        require(!snapshots[k].exists, "already registered");
        snapshots[k] = Snapshot(hash, block.timestamp, true);
        count += 1;
        emit SnapshotRegistered(date, hash, block.timestamp);
    }

    /// @notice Return the registered hash for a date (0x0 if absent).
    function getSnapshotHash(string calldata date) external view returns (bytes32) {
        return snapshots[_key(date)].hash;
    }

    /// @notice True iff the supplied hash matches the registered one.
    function verifySnapshot(string calldata date, bytes32 hash) external view returns (bool) {
        Snapshot memory s = snapshots[_key(date)];
        return s.exists && s.hash == hash;
    }
}
