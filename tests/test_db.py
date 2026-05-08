"""
test_db.py — Unit tests for the face database.
"""

import json
import os

# Resolve database.json relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "database.json")


def test_database_loads_and_has_persons():
    """Assert that database.json loads correctly and contains at least one person."""
    assert os.path.exists(DB_PATH), f"database.json not found at {DB_PATH}"

    with open(DB_PATH, "r") as f:
        db = json.load(f)

    assert "persons" in db, "database.json must contain a 'persons' key"
    assert isinstance(db["persons"], list), "'persons' must be a list"
    assert len(db["persons"]) >= 1, "Database must contain at least one person"

    # Verify each person has required fields
    for person in db["persons"]:
        assert "name" in person, f"Person entry missing 'name': {person}"
        assert "embedding" in person, f"Person entry missing 'embedding': {person}"
        assert isinstance(person["embedding"], list), "Embedding must be a list"
        assert len(person["embedding"]) == 128, (
            f"Embedding for '{person['name']}' has {len(person['embedding'])} "
            f"dimensions, expected 128"
        )
