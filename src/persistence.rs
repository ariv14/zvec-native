use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use crate::collection::Collection;

#[derive(Serialize, Deserialize)]
struct Metadata {
    dimensions: usize,
    next_id: usize,
    id_map: HashMap<String, usize>,
    deleted_ids: HashSet<String>,
    /// Vectors stored as base64-encoded f32 arrays keyed by internal ID
    vectors: HashMap<String, String>,
}

const METADATA_FILE: &str = "metadata.json";

pub fn save_collection(collection: &Collection) -> Result<(), String> {
    let path = &collection.path;
    fs::create_dir_all(path).map_err(|e| format!("Failed to create directory: {}", e))?;

    // Encode vectors as base64
    let mut encoded_vectors: HashMap<String, String> = HashMap::new();
    for (&internal_id, vec) in &collection.vectors {
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
        encoded_vectors.insert(internal_id.to_string(), BASE64.encode(&bytes));
    }

    let metadata = Metadata {
        dimensions: collection.dimensions,
        next_id: collection.next_id,
        id_map: collection.id_map.clone(),
        deleted_ids: collection.deleted_ids.clone(),
        vectors: encoded_vectors,
    };

    let json = serde_json::to_string_pretty(&metadata)
        .map_err(|e| format!("Failed to serialize metadata: {}", e))?;

    let metadata_path = path.join(METADATA_FILE);
    fs::write(&metadata_path, json)
        .map_err(|e| format!("Failed to write metadata: {}", e))?;

    Ok(())
}

pub fn load_collection(path: &Path) -> Result<Option<Collection>, String> {
    let metadata_path = path.join(METADATA_FILE);

    if !metadata_path.exists() {
        return Ok(None);
    }

    let json = fs::read_to_string(&metadata_path)
        .map_err(|e| format!("Failed to read metadata: {}", e))?;

    let metadata: Metadata = serde_json::from_str(&json)
        .map_err(|e| format!("Failed to parse metadata: {}", e))?;

    let mut collection = Collection::new(path.to_path_buf(), metadata.dimensions);
    collection.next_id = metadata.next_id;
    collection.id_map = metadata.id_map;
    collection.deleted_ids = metadata.deleted_ids;

    // Decode vectors from base64
    for (id_str, b64) in &metadata.vectors {
        let internal_id: usize = id_str.parse()
            .map_err(|e| format!("Invalid internal ID '{}': {}", id_str, e))?;

        let bytes = BASE64.decode(b64)
            .map_err(|e| format!("Failed to decode vector: {}", e))?;

        let vec: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        collection.vectors.insert(internal_id, vec);
    }

    // Build reverse map from id_map
    for (uuid, &internal_id) in &collection.id_map {
        collection.reverse_map.insert(internal_id, uuid.clone());
    }

    // Rebuild HNSW from stored vectors
    collection.rebuild_from_vectors();

    Ok(Some(collection))
}

pub fn collection_file_size(path: &Path) -> u64 {
    let metadata_path = path.join(METADATA_FILE);
    if metadata_path.exists() {
        fs::metadata(&metadata_path).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    }
}
