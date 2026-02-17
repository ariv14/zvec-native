use hnsw_rs::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// HNSW parameters
const MAX_NB_CONNECTION: usize = 16; // M parameter
const MAX_ELEMENTS: usize = 100_000;
const MAX_LAYER: usize = 16;
const EF_CONSTRUCTION: usize = 200;

pub struct Collection {
    pub hnsw: Hnsw<'static, f32, DistCosine>,
    pub id_map: HashMap<String, usize>,
    pub reverse_map: HashMap<usize, String>,
    pub deleted_ids: HashSet<String>,
    pub next_id: usize,
    pub dimensions: usize,
    pub path: PathBuf,
    pub dirty: bool,
    pub vectors: HashMap<usize, Vec<f32>>,
}

impl Collection {
    pub fn new(path: PathBuf, dimensions: usize) -> Self {
        let hnsw = Hnsw::<f32, DistCosine>::new(
            MAX_NB_CONNECTION,
            MAX_ELEMENTS,
            MAX_LAYER,
            EF_CONSTRUCTION,
            DistCosine,
        );

        Collection {
            hnsw,
            id_map: HashMap::new(),
            reverse_map: HashMap::new(),
            deleted_ids: HashSet::new(),
            next_id: 0,
            dimensions,
            path,
            dirty: false,
            vectors: HashMap::new(),
        }
    }

    /// Rebuild HNSW index from stored vectors (excluding deleted).
    /// Used after loading from persistence or after deletions.
    pub fn rebuild_from_vectors(&mut self) {
        self.hnsw = Hnsw::<f32, DistCosine>::new(
            MAX_NB_CONNECTION,
            MAX_ELEMENTS,
            MAX_LAYER,
            EF_CONSTRUCTION,
            DistCosine,
        );

        // Re-insert all non-deleted vectors
        for (&internal_id, vec) in &self.vectors {
            if let Some(uuid) = self.reverse_map.get(&internal_id) {
                if !self.deleted_ids.contains(uuid) {
                    self.hnsw.insert((vec.as_slice(), internal_id));
                }
            }
        }
    }

    pub fn insert_vector(&mut self, id: &str, vector: Vec<f32>) {
        // Handle upsert: if ID already exists, mark old one as deleted
        if let Some(&old_internal) = self.id_map.get(id) {
            self.deleted_ids.insert(id.to_string());
            self.vectors.remove(&old_internal);
            self.reverse_map.remove(&old_internal);
        }

        let internal_id = self.next_id;
        self.next_id += 1;

        self.id_map.insert(id.to_string(), internal_id);
        self.reverse_map.insert(internal_id, id.to_string());
        self.vectors.insert(internal_id, vector.clone());

        // Remove from deleted if it was previously deleted
        self.deleted_ids.remove(id);

        self.hnsw.insert((vector.as_slice(), internal_id));
        self.dirty = true;
    }

    pub fn search_vectors(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(String, f32)> {
        let ef = std::cmp::max(ef_search, k);
        let results = self.hnsw.search(query, k + self.deleted_ids.len(), ef);

        let mut output: Vec<(String, f32)> = Vec::new();

        for neighbour in results {
            if output.len() >= k {
                break;
            }
            let internal_id = neighbour.d_id;
            if let Some(uuid) = self.reverse_map.get(&internal_id) {
                if !self.deleted_ids.contains(uuid) {
                    // Convert distance to similarity: score = 1.0 - distance
                    let score = 1.0 - neighbour.distance;
                    output.push((uuid.clone(), score));
                }
            }
        }

        output
    }

    pub fn delete_vector(&mut self, id: &str) -> bool {
        if self.id_map.contains_key(id) && !self.deleted_ids.contains(id) {
            self.deleted_ids.insert(id.to_string());
            self.dirty = true;
            true
        } else {
            false
        }
    }

    pub fn active_count(&self) -> usize {
        self.id_map.len() - self.deleted_ids.len()
    }
}
