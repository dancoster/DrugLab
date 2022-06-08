SELECT DISTINCT ON (subject_id) * FROM admissions
WHERE has_chartevents_data=1
ORDER BY subject_id, admittime ASC
LIMIT 100;