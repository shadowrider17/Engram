import Testing
import EngramKit
import EngramModels
import Lattice
import Foundation
import SQLite3

/// WAL read locks left behind by short-lived Lattice handles.
///
/// Each test opens a fresh database, performs some reads, then writes filler
/// frames from a plain SQLite connection and asks that connection for a
/// PASSIVE checkpoint. A complete checkpoint means no read lock was holding the
/// WAL; a partial one means a read snapshot outlived the code that took it.
/// The idle timeout is 1 s, so every test waits past it before checking.
@Suite("WAL read-lock leaks", .serialized)
struct WalReadLockLeakTests {

    struct CheckpointOutcome: CustomStringConvertible {
        let busy: Int32, logFrames: Int32, checkpointed: Int32
        var complete: Bool { busy == 0 && checkpointed == logFrames }
        var description: String { "busy=\(busy) log=\(logFrames) checkpointed=\(checkpointed)" }
    }

    static func configuration(_ url: URL) -> Lattice.Configuration {
        var config = Lattice.Configuration(fileURL: url, migration: engramMigrations)
        config.resultsTuning.generationTTLSeconds = 1
        return config
    }

    static func open(_ url: URL) throws -> Lattice {
        try Lattice(Memory.self, Edge.self, Checkpoint.self, HookState.self,
                    SessionState.self, SyncConfig.self, configuration: configuration(url))
    }

    static func freshDatabase() throws -> URL {
        try testFixtureDirectory().appending(path: "wal-leak-\(UUID().uuidString).sqlite")
    }

    static func removeDatabase(_ url: URL) {
        for suffix in ["", "-wal", "-shm"] {
            try? FileManager.default.removeItem(atPath: url.path + suffix)
        }
    }

    /// The backstop's read: one marker row through a live Results access.
    static func readMarker(_ lattice: Lattice) -> Int {
        EmbeddingMigration.storedVersion(in: lattice)
    }

    /// Write filler frames from a plain connection, wait past the idle
    /// timeout, then try a PASSIVE checkpoint from the same connection.
    static func writeThenCheckpoint(_ url: URL, wait: Double = 3) throws -> CheckpointOutcome {
        var db: OpaquePointer?
        guard sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READWRITE, nil) == SQLITE_OK else {
            throw CocoaError(.fileReadUnknown)
        }
        defer { sqlite3_close_v2(db) }
        sqlite3_busy_timeout(db, 2000)
        func exec(_ sql: String) throws {
            guard sqlite3_exec(db, sql, nil, nil, nil) == SQLITE_OK else {
                throw CocoaError(.fileWriteUnknown)
            }
        }
        try exec("PRAGMA wal_autocheckpoint=0")
        try exec("CREATE TABLE IF NOT EXISTS _filler(x BLOB)")
        try exec("BEGIN; INSERT INTO _filler SELECT randomblob(3900) FROM "
                 + "(WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i<200) "
                 + "SELECT i FROM n); COMMIT;")
        Thread.sleep(forTimeInterval: wait)
        var log: Int32 = -1, ckpt: Int32 = -1
        let rc = sqlite3_wal_checkpoint_v2(db, nil, SQLITE_CHECKPOINT_PASSIVE, &log, &ckpt)
        return CheckpointOutcome(busy: rc == SQLITE_BUSY ? 1 : 0, logFrames: log, checkpointed: ckpt)
    }

    /// Regression: the MCP server's startup backstop must not leave a read
    /// lock behind on an idle server (one that never receives a tool call).
    @Test func serverBackstop_releasesItsReadLock() async throws {
        let url = try Self.freshDatabase()
        defer { Self.removeDatabase(url) }
        // Mark the space current through a separate handle that closes, so the
        // backstop only reads the marker, as on an already-migrated hub.
        let setup = try Self.open(url)
        try EmbeddingMigration.setStoredVersion(EmbeddingSpace.currentVersion, in: setup)
        setup.close()

        let owner = try Self.open(url)  // stands in for MemoryTools' long-lived handle
        let ref = owner.sendableReference
        await Task.detached {
            await EmbeddingMigration.runBackstop(from: ref, embedder: EmbeddingService(), log: { _ in })
        }.value
        let result = try Self.writeThenCheckpoint(url)
        print("serverBackstop: \(result)")
        withExtendedLifetime(owner) {}
        #expect(result.logFrames > 200, "the filler write must reach the WAL")
        #expect(result.complete)
    }
}
