import numpy as np
import json

def _cosine_sim(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

class SemanticMemory:
    def __init__(self):
        """ Stores generalized schemas/skills/options. """
        self.schemas = {} # map from schema_id to schema dict
        self.version = "1.0"

    def upsert(self, schema_id, schema):
        """
        Inserts a new schema or updates an existing one.
        A schema could be:
        {
          "precondition": vector,
          "action_model": {action: score},
          "expected_reward": float,
          "confidence": float,
          "use_count": int
        }
        """
        # A simple upsert. More sophisticated logic could involve merging.
        self.schemas[schema_id] = schema

    def query(self, state_signature, k=1):
        """
        Find the k most similar schemas to a given state signature vector.
        """
        if not self.schemas:
            return []

        scored_schemas = []
        for sid, schema in self.schemas.items():
            sim = _cosine_sim(state_signature, schema.get("precondition", []))
            scored_schemas.append((sim, sid, schema))

        scored_schemas.sort(key=lambda x: x[0], reverse=True)
        return scored_schemas[:k]

    def export(self):
        """ Returns all schemas. """
        return self.schemas

    def save(self, path):
        # Convert numpy arrays to lists for JSON serialization
        exportable_schemas = {}
        for sid, schema in self.schemas.items():
            s_copy = schema.copy()
            if 'precondition' in s_copy and isinstance(s_copy['precondition'], np.ndarray):
                s_copy['precondition'] = s_copy['precondition'].tolist()
            exportable_schemas[sid] = s_copy

        with open(path, 'w') as f:
            json.dump({'version': self.version, 'schemas': exportable_schemas}, f, indent=2)

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            data = json.load(f)

        mem = SemanticMemory()
        if mem.version != data.get('version'):
            print(f"Warning: Loading semantic memory from a different version. Loaded: {data.get('version')}, Current: {mem.version}")

        mem.schemas = data.get('schemas', {})
        # Convert preconditions back to numpy arrays
        for sid, schema in mem.schemas.items():
            if 'precondition' in schema:
                schema['precondition'] = np.array(schema['precondition'])
        return mem
