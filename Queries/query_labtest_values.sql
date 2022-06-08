SELECT * FROM (
    SELECT * FROM labevents INNER JOIN (SELECT * FROM d_labitems) k 
    ON labevents.itemid=k.itemid
) AS l 
WHERE 
l.hadm_id IS NOT NULL 
AND
(
    l.hadm_id IN (
        SELECT DISTINCT ON (subject_id) hadm_id FROM admissions
        WHERE has_chartevents_data=1
        ORDER BY subject_id, admittime ASC
    )
)
AND l.valuenum IS NOT NULL
LIMIT 1000;