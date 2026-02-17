mod collection;
mod persistence;

use collection::Collection;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

static COLLECTIONS: Lazy<RwLock<HashMap<String, Collection>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

#[napi(object)]
pub struct CollectionConfig {
    pub path: String,
    pub dimensions: u32,
    pub index_type: String,
    pub metric: String,
}

#[napi(object)]
pub struct SearchResult {
    pub id: String,
    pub score: f64,
}

#[napi(object)]
pub struct CollectionStats {
    pub count: u32,
    pub dimensions: u32,
    pub file_size_bytes: u32,
}

#[napi]
pub fn create_collection(config: CollectionConfig) -> Result<()> {
    if config.metric != "cosine" {
        return Err(Error::from_reason(format!(
            "Unsupported metric '{}'. Only 'cosine' is supported.",
            config.metric
        )));
    }
    if config.index_type != "hnsw" {
        return Err(Error::from_reason(format!(
            "Unsupported index type '{}'. Only 'hnsw' is supported.",
            config.index_type
        )));
    }
    if config.dimensions == 0 {
        return Err(Error::from_reason("Dimensions must be > 0".to_string()));
    }

    let path = PathBuf::from(&config.path);
    let key = config.path.clone();

    let mut collections = COLLECTIONS
        .write()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    // Idempotent: if already loaded, skip
    if collections.contains_key(&key) {
        return Ok(());
    }

    // Try to load existing collection from disk
    match persistence::load_collection(&path) {
        Ok(Some(existing)) => {
            if existing.dimensions != config.dimensions as usize {
                return Err(Error::from_reason(format!(
                    "Dimension mismatch: existing collection has {} dims, requested {}",
                    existing.dimensions, config.dimensions
                )));
            }
            collections.insert(key, existing);
        }
        Ok(None) => {
            let coll = Collection::new(path, config.dimensions as usize);
            collections.insert(key, coll);
        }
        Err(e) => {
            return Err(Error::from_reason(format!(
                "Failed to load collection: {}",
                e
            )));
        }
    }

    Ok(())
}

#[napi]
pub fn insert_vector(path: String, id: String, vector: Float32Array) -> Result<()> {
    let mut collections = COLLECTIONS
        .write()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let coll = collections
        .get_mut(&path)
        .ok_or_else(|| Error::from_reason(format!("Collection not found at '{}'", path)))?;

    if vector.len() != coll.dimensions {
        return Err(Error::from_reason(format!(
            "Dimension mismatch: expected {}, got {}",
            coll.dimensions,
            vector.len()
        )));
    }

    let vec: Vec<f32> = vector.to_vec();
    coll.insert_vector(&id, vec);

    Ok(())
}

#[napi]
pub fn build_index(path: String) -> Result<()> {
    let mut collections = COLLECTIONS
        .write()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let coll = collections
        .get_mut(&path)
        .ok_or_else(|| Error::from_reason(format!("Collection not found at '{}'", path)))?;

    // If deletions are pending, rebuild the HNSW from scratch
    if !coll.deleted_ids.is_empty() {
        // Remove deleted vectors and id mappings
        let deleted: Vec<String> = coll.deleted_ids.iter().cloned().collect();
        for uuid in &deleted {
            if let Some(&internal_id) = coll.id_map.get(uuid) {
                coll.vectors.remove(&internal_id);
                coll.reverse_map.remove(&internal_id);
            }
            coll.id_map.remove(uuid);
        }
        coll.deleted_ids.clear();

        coll.rebuild_from_vectors();
    }

    // Persist to disk
    persistence::save_collection(coll)
        .map_err(|e| Error::from_reason(e))?;

    coll.dirty = false;

    Ok(())
}

#[napi]
pub fn search(path: String, query: Float32Array, k: u32) -> Result<Vec<SearchResult>> {
    let collections = COLLECTIONS
        .read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let coll = collections
        .get(&path)
        .ok_or_else(|| Error::from_reason(format!("Collection not found at '{}'", path)))?;

    if query.len() != coll.dimensions {
        return Err(Error::from_reason(format!(
            "Query dimension mismatch: expected {}, got {}",
            coll.dimensions,
            query.len()
        )));
    }

    if coll.active_count() == 0 {
        return Ok(Vec::new());
    }

    let results = coll.search_vectors(query.as_ref(), k as usize);

    Ok(results
        .into_iter()
        .map(|(id, score)| SearchResult {
            id,
            score: score as f64,
        })
        .collect())
}

#[napi]
pub fn delete_vector(path: String, id: String) -> Result<bool> {
    let mut collections = COLLECTIONS
        .write()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let coll = collections
        .get_mut(&path)
        .ok_or_else(|| Error::from_reason(format!("Collection not found at '{}'", path)))?;

    Ok(coll.delete_vector(&id))
}

#[napi]
pub fn stats(path: String) -> Result<CollectionStats> {
    let collections = COLLECTIONS
        .read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let coll = collections
        .get(&path)
        .ok_or_else(|| Error::from_reason(format!("Collection not found at '{}'", path)))?;

    let file_size = persistence::collection_file_size(&coll.path);

    Ok(CollectionStats {
        count: coll.active_count() as u32,
        dimensions: coll.dimensions as u32,
        file_size_bytes: file_size as u32,
    })
}
