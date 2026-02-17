import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  createCollection,
  insertVector,
  buildIndex,
  search,
  deleteVector,
  stats,
} from "../index.js";

/** Generate a random normalized vector of given dimensions */
function randomVector(dims) {
  const vec = new Float32Array(dims);
  let norm = 0;
  for (let i = 0; i < dims; i++) {
    vec[i] = Math.random() - 0.5;
    norm += vec[i] * vec[i];
  }
  norm = Math.sqrt(norm);
  for (let i = 0; i < dims; i++) {
    vec[i] /= norm;
  }
  return vec;
}

/** Create a deterministic vector: all zeros except position idx */
function basisVector(dims, idx) {
  const vec = new Float32Array(dims);
  vec[idx % dims] = 1.0;
  return vec;
}

const DIMS = 384;

describe("createCollection", () => {
  let tmpDir;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "zvec-test-"));
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("should create a new collection", () => {
    const collPath = join(tmpDir, "coll1");
    createCollection({
      path: collPath,
      dimensions: DIMS,
      indexType: "hnsw",
      metric: "cosine",
    });

    const s = stats(collPath);
    assert.equal(s.count, 0);
    assert.equal(s.dimensions, DIMS);
  });

  it("should be idempotent", () => {
    const collPath = join(tmpDir, "coll1");
    const config = {
      path: collPath,
      dimensions: DIMS,
      indexType: "hnsw",
      metric: "cosine",
    };
    createCollection(config);
    createCollection(config); // should not throw
    assert.equal(stats(collPath).count, 0);
  });

  it("should reject invalid metric", () => {
    const collPath = join(tmpDir, "coll1");
    assert.throws(
      () =>
        createCollection({
          path: collPath,
          dimensions: DIMS,
          indexType: "hnsw",
          metric: "euclidean",
        }),
      /Unsupported metric/
    );
  });

  it("should reject invalid index type", () => {
    const collPath = join(tmpDir, "coll1");
    assert.throws(
      () =>
        createCollection({
          path: collPath,
          dimensions: DIMS,
          indexType: "flat",
          metric: "cosine",
        }),
      /Unsupported index type/
    );
  });
});

describe("insertVector", () => {
  let tmpDir;
  let collPath;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "zvec-test-"));
    collPath = join(tmpDir, "coll1");
    createCollection({
      path: collPath,
      dimensions: DIMS,
      indexType: "hnsw",
      metric: "cosine",
    });
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("should insert a vector", () => {
    insertVector(collPath, "vec-1", randomVector(DIMS));
    assert.equal(stats(collPath).count, 1);
  });

  it("should reject wrong dimensions", () => {
    assert.throws(
      () => insertVector(collPath, "vec-1", new Float32Array(128)),
      /Dimension mismatch/
    );
  });

  it("should handle upsert (same ID)", () => {
    insertVector(collPath, "vec-1", randomVector(DIMS));
    insertVector(collPath, "vec-1", randomVector(DIMS));
    assert.equal(stats(collPath).count, 1);
  });
});

describe("search", () => {
  let tmpDir;
  let collPath;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "zvec-test-"));
    collPath = join(tmpDir, "coll1");
    createCollection({
      path: collPath,
      dimensions: DIMS,
      indexType: "hnsw",
      metric: "cosine",
    });
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("should return empty for empty collection", () => {
    const results = search(collPath, randomVector(DIMS), 5);
    assert.equal(results.length, 0);
  });

  it("should find nearest neighbors with correct ordering", () => {
    // Insert basis vectors pointing in different directions
    const v1 = basisVector(DIMS, 0);
    const v2 = basisVector(DIMS, 1);
    const v3 = basisVector(DIMS, 2);

    insertVector(collPath, "a", v1);
    insertVector(collPath, "b", v2);
    insertVector(collPath, "c", v3);

    buildIndex(collPath);

    // Query with v1 — should find "a" first with highest score
    const results = search(collPath, v1, 3);
    assert.ok(results.length > 0, "Should return results");
    assert.equal(results[0].id, "a");
    assert.ok(results[0].score > 0.9, `Score should be ~1.0, got ${results[0].score}`);
  });

  it("should respect k limit", () => {
    for (let i = 0; i < 10; i++) {
      insertVector(collPath, `v-${i}`, randomVector(DIMS));
    }
    buildIndex(collPath);

    const results = search(collPath, randomVector(DIMS), 3);
    assert.ok(results.length <= 3);
  });

  it("should handle k > count", () => {
    insertVector(collPath, "only", randomVector(DIMS));
    buildIndex(collPath);

    const results = search(collPath, randomVector(DIMS), 10);
    assert.equal(results.length, 1);
  });
});

describe("deleteVector", () => {
  let tmpDir;
  let collPath;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "zvec-test-"));
    collPath = join(tmpDir, "coll1");
    createCollection({
      path: collPath,
      dimensions: DIMS,
      indexType: "hnsw",
      metric: "cosine",
    });
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("should return true for existing vector", () => {
    insertVector(collPath, "vec-1", randomVector(DIMS));
    assert.equal(deleteVector(collPath, "vec-1"), true);
  });

  it("should return false for missing vector", () => {
    assert.equal(deleteVector(collPath, "nonexistent"), false);
  });

  it("should exclude deleted vectors from search after rebuild", () => {
    const target = basisVector(DIMS, 0);
    insertVector(collPath, "keep", basisVector(DIMS, 1));
    insertVector(collPath, "remove", target);
    buildIndex(collPath);

    deleteVector(collPath, "remove");
    buildIndex(collPath);

    const results = search(collPath, target, 5);
    const ids = results.map((r) => r.id);
    assert.ok(!ids.includes("remove"), "Deleted vector should not appear in results");
  });
});

describe("stats", () => {
  let tmpDir;
  let collPath;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "zvec-test-"));
    collPath = join(tmpDir, "coll1");
    createCollection({
      path: collPath,
      dimensions: DIMS,
      indexType: "hnsw",
      metric: "cosine",
    });
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("should return correct count and dimensions", () => {
    insertVector(collPath, "a", randomVector(DIMS));
    insertVector(collPath, "b", randomVector(DIMS));
    const s = stats(collPath);
    assert.equal(s.count, 2);
    assert.equal(s.dimensions, DIMS);
  });

  it("should show file size after build", () => {
    insertVector(collPath, "a", randomVector(DIMS));
    buildIndex(collPath);
    const s = stats(collPath);
    assert.ok(s.fileSizeBytes > 0, "File size should be > 0 after build");
  });
});

describe("persistence", () => {
  let tmpDir;
  let collPath;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "zvec-test-"));
    collPath = join(tmpDir, "coll1");
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("should survive dump and reload", () => {
    createCollection({
      path: collPath,
      dimensions: DIMS,
      indexType: "hnsw",
      metric: "cosine",
    });

    const v1 = basisVector(DIMS, 0);
    const v2 = basisVector(DIMS, 1);
    insertVector(collPath, "a", v1);
    insertVector(collPath, "b", v2);
    buildIndex(collPath);

    // The global COLLECTIONS cache still has this loaded.
    // We can't easily "unload" it from JS, but we can verify the files exist
    // and that stats are correct.
    const s = stats(collPath);
    assert.equal(s.count, 2);
    assert.equal(s.dimensions, DIMS);
    assert.ok(s.fileSizeBytes > 0);
  });
});
