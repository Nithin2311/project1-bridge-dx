# Data

EHRShot class extract (13 OMOP CSVs). Lives **outside** this repo at `../FinalProject/data/` on Nithin's machine.
Set `EHRSHOT_DATA` to point at the folder on any other machine. Never copy it into the repo.

Tables present: person, visit_occurrence, condition_occurrence, drug_exposure, procedure_occurrence, death,
observation_period, concept, concept_ancestor, concept_relationship, concept_class, domain, relationship, vocabulary.
Not in the class extract: measurement, observation (available via Redivis full access if ever needed; not needed for Option A).

Loader: `bridgedx.data.connect()` opens a DuckDB database at `.cache/ehrshot.duckdb` and registers each CSV as a
view the first time it is used. `concept.csv` is 1.1 GB and `concept_relationship.csv` is 3.7 GB; DuckDB reads them
lazily, so first queries on them take ~10–60 s and are cached afterwards.

Columns we rely on:
- `condition_occurrence`: person_id, condition_concept_id, condition_start_DATE, condition_source_value (ICD code), visit_occurrence_id
- `person`: person_id, gender_concept_id, year_of_birth, birth_DATETIME
- `death`: person_id, death_DATE
- `observation_period`: person_id, observation_period_start_DATE, observation_period_end_DATE
- `concept`: concept_id, concept_name, vocabulary_id, concept_code, domain_id
