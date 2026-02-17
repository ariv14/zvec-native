# @moltmind/zvec-native

HNSW approximate nearest neighbor search for [MoltMind](https://github.com/ariv14/moltmind). Native Rust addon via napi-rs.

## Install

```bash
npm install @moltmind/zvec-native
```

Pre-built binaries are available for:
- macOS (ARM64, x64)
- Linux (x64, ARM64 glibc)
- Windows (x64)

## API

```js
const {
  createCollection,
  insertVector,
  buildIndex,
  search,
  deleteVector,
  stats,
} = require("@moltmind/zvec-native");

// Create or load a collection
createCollection({
  path: "/tmp/my-vectors",
  dimensions: 384,
  indexType: "hnsw",
  metric: "cosine",
});

// Insert vectors (Float32Array)
insertVector("/tmp/my-vectors", "doc-1", new Float32Array(384));

// Build/persist the index
buildIndex("/tmp/my-vectors");

// Search (returns [{ id, score }])
const results = search("/tmp/my-vectors", queryVector, 10);

// Delete a vector (soft delete until next buildIndex)
deleteVector("/tmp/my-vectors", "doc-1");

// Get collection stats
const { count, dimensions, fileSizeBytes } = stats("/tmp/my-vectors");
```

## How it works

- Uses [hnsw_rs](https://crates.io/crates/hnsw_rs) for the HNSW algorithm (M=16, ef_construction=200)
- Collections are file-based directories with `metadata.json` for persistence
- In-process cache avoids reloading the index on every call
- Cosine similarity scores (0-1, higher = more similar)
- Vectors must be L2-normalized before insertion (the MiniLM-L6-v2 model used by MoltMind already produces normalized vectors)

## Building from source

Requires Rust 1.70+ and Node.js 18+.

```bash
npm install
npm run build
npm test
```

## License

MIT
